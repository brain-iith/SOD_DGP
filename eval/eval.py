# Copyright 2021 Ayush Jain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
# from tensorflow.python.summary import event_accumulator
from tensorboard.backend.event_processing import event_accumulator

def get_min(event_file, scalar_str):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    minval = 999999
    minstep=0
    for scalar in ea.Scalars(scalar_str):
        if scalar.value< minval:
            minstep = scalar.step
            minval = scalar.value
    return minstep, minval

def get_max(event_file, scalar_str):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    maxval = -999999
    optstep=0
    for scalar in ea.Scalars(scalar_str):
        if scalar.value> maxval:
            optstep = scalar.step
            maxval = scalar.value
    return optstep, maxval

def get_opt_step_using_window(event_file, window_size):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    epsilon = 0.0001
    opt_lik=-99999.0
    optstep=0
    scalars = ea.Scalars('train_lik')
    prev_avg = -99999.0
    for i in range(0,len(scalars),window_size):
        cur_avg=0.0
        for scalar in scalars[i:i+window_size]: 
            cur_avg += scalar.value
        cur_avg /= window_size
        if cur_avg<(prev_avg-epsilon):
            #find max in prev window to set as optimal
            maxval = -999999.0
            maxstep=0
            for s in scalars[i-window_size:i]:
                if s.value> maxval:
                    maxstep = s.step
                    maxval = s.value
            optstep = maxstep
            opt_lik = maxval
            break
        prev_avg = cur_avg
    
    #compare max in last window with optimal if optstep is 0 i.e. increasing optimisation curve , peak not found
    if optstep==0:
        for s in scalars[len(scalars)-window_size:]: #[-1:-1-window_size:-1]:
            if s.value> opt_lik:
                optstep = s.step
                opt_lik = s.value
    return optstep, opt_lik

def get_opt_step(event_file):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    optrmse = 999999
    optlik = -999999
    optstep=0
    for scalar in ea.Scalars('train_lik'):
        rmse = get_val_at_step(event_file, scalar.step, 'train_rmse')
        if scalar.value> optlik and rmse< optrmse:
            optstep = scalar.step
            optrmse = rmse
            optlik = scalar.value
    return optstep
    lc = np.stack(
      [np.asarray([scalar.step, scalar.value])
      for scalar in ea.Scalars(scalar_str)])
    return(lc)

def get_val_at_step(event_file, step, scalar_str):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    for scalar in ea.Scalars(scalar_str):
        if scalar.step == step:
            return scalar.value
def get_last_step(event_file):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    return ea.Scalars('train_lik')[-1].step

import argparse
import os, glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', '-dir', nargs = 1, required = True, help = 'Event files folder name')
    parser.add_argument('-window_size', '-ws', nargs = 1, type = int, default = 0, help = 'window size:(0 for get max)(-1 to report last iteration value)(>0 for window over train_lik, usually >=5)')
    args = parser.parse_args()
    folder = args.folder[0]
    window_size = args.window_size[0]
    os.chdir('./'+folder)
    files = glob.glob('model-tensorboard*')
    rmse_list = []
    lik_list = []
    train_lik_list = []
    train_rmse_list = []
    folder = folder.split('/')[-1]
    layer = int(folder.split('_')[1])
    n=0;
    for f in files:
        n = n+ 1
    kv = np.zeros((layer,n))
    kls = np.zeros((layer,n))
    i =0
    for f in files:
#         step,_ = get_min(f,'val_rmse')
#         step = get_opt_step(f)
        if window_size ==0:
            step, opt_train_lik = get_max(f, 'train_lik')
            opt_train_rmse = get_val_at_step(f,step,'train_rmse')
        elif window_size <0:
            step = get_last_step(f)
            opt_train_lik = get_val_at_step(f,step,'train_lik')
            opt_train_rmse = get_val_at_step(f,step,'train_rmse')
        else:
            step, opt_train_lik = get_opt_step_using_window(f,window_size)
            opt_train_rmse = get_val_at_step(f,step,'train_rmse')
        rmse = get_val_at_step(f,step,'test_rmse')
        lik = get_val_at_step(f,step,'test_lik')
        
        train_rmse_list.append(opt_train_rmse)
        train_lik_list.append(opt_train_lik)
        rmse_list.append(rmse)
        lik_list.append(lik)
        print(f)
        print(str(step)+'\n')
        for l in range(layer):
            kls[l][i] = get_val_at_step(f,step,folder+'/layers/'+str(l)+'/kern/lengthscales_1')
            kv[l][i] = get_val_at_step(f,step,folder+'/layers/'+str(l)+'/kern/variance_1')
        i = i+1
    file = open('Test_Result'+str(window_size)+'.txt','w')
    file.write('Train RMSE: mean:{}, dev:{}'.format(np.mean(train_rmse_list),np.std(train_rmse_list)))
    file.write('\nTrain Likelihood: mean:{}, dev:{}'.format(np.mean(train_lik_list),np.std(train_lik_list)))
    file.write('\nTest rmse: mean:{}, dev:{}'.format(np.mean(rmse_list),np.std(rmse_list)))
    file.write('\nTest log likelihood: mean:{}, dev:{}'.format(np.mean(lik_list),np.std(lik_list)))
    for l in range(layer):
        file.write('\nLayer:{}'.format(l))
        file.write('\nkernel lengthscale:')
        file.write('mean:{}, dev:{}\n'.format(np.mean(kls[l]),np.std(kls[l])))
        for j in range(n):
            file.write('{},'.format(kls[l][j]))
        file.write('\nkernel variance:')
        file.write('mean:{}, dev:{}\n'.format(np.mean(kv[l]),np.std(kv[l])))
        for j in range(n):
            file.write('{},'.format(kv[l][j]))
    file.close()

if __name__ == "__main__":    
    main()