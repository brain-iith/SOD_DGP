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
np.random.seed(0)
import matplotlib.pyplot as plt
import time
from math import ceil
import collections
from csv import DictWriter

import os
import io


import tensorflow as tf
tf.set_random_seed(0)
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import gpflow as gpflow
#Allow GPU memory growth 
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
gpflow.reset_default_session(config=config)

from gpflow import autoflow, params_as_tensors
from gpflow.params import DataHolder
from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF, White
from gpflow.mean_functions import Constant
# from gpflow.models.sgpr import SGPR, GPRFITC
# from gpflow.models.svgp import SVGP
# from gpflow.models.gpr import GPR
from gpflow.training import AdamOptimizer, ScipyOptimizer, NatGradOptimizer
import gpflow.training.monitor as mon
from gpflow.actions import Action, Loop
from gpflow import settings,session_manager
from core.utils import reparameterize
# from scipy.interpolate import make_interp_spline, BSpline


from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from scipy.special import logsumexp

from core.sod_dgp import SOD_DGP
from core.utils import get_subsets
from datasets import Datasets

from numba import cuda


def load_data(name,split):
    datasets = Datasets(data_path='./data/')
    dataset_name =  name
    data = datasets.all_datasets[dataset_name].get_data(split=split)
    X, Y, Xs, Ys, Y_std = [data[_] for _ in ['X', 'Y', 'Xs', 'Ys', 'Y_std']]
    
    ind = np.arange(X.shape[0])

    np.random.seed(0)
    np.random.shuffle(ind)

    n = int(X.shape[0]* 0.889)

    Xval = X[ind[n:], :]
    X = X[ind[:n], :]
    
    Yval = Y[ind[n:], :]
    Y = Y[ind[:n], :]
    
    print('N: {}, D: {}, Ns: {}, Nval: {}'.format(X.shape[0], X.shape[1], Xs.shape[0],Xval.shape[0]))
    
    return X, Y, Xs, Ys, Xval, Yval, Y_std

def select_subset(X, Y, subset_size):
    X, X_sub, Y, Y_sub = get_subsets(X,  Y, subset_size)
    #do auto subset selction
    return X, X_sub, Y, Y_sub

def make_dgp_models(dataset_name, X, Y, X_sub,Y_sub,subset_size,minibatch_size, kern_var, kern_length_scale, layers, lik_var, num_samples_train, train_full_cov,activation, back_constraint,full_mean_field):
    models, names = [], []
    for L in layers:
        D = min(30,X.shape[1])
        with gpflow.defer_build():
            # the layer shapes are defined by the kernel dims, so here all hidden layers are D dimensional 
            kernels = []
            kernels.append(RBF(X.shape[1],variance =kern_var[0], lengthscales=kern_length_scale[0]))
            for l in range(1,L):
                kernels.append(RBF(D,variance = kern_var[l], lengthscales=kern_length_scale[l])) 

            name = 'DGP_{}'.format(L) +'_'+ dataset_name + '_'+ str(subset_size)

            model = SOD_DGP(X, Y, X_sub, Y_sub,subset_size, kernels, Gaussian(lik_var), layer_variance = 10**(-5), num_outputs = Y.shape[1],back_constraint = back_constraint, activation = activation, full_mean_field = full_mean_field,full_cov = train_full_cov, num_samples=num_samples_train, minibatch_size=minibatch_size,name = name)       

        model.compile()

        models.append(model)
        names.append(name)
        print(name+":")
        print(model)
        print(model.compute_log_likelihood())            
    
    return models, names

#---------------Tensorboard Tasks---------------------------
class CustomTestPerformanceTensorBoardTask(mon.BaseTensorBoardTask):
    def __init__(self, file_writer, model, X, Y, Xt, Yt, Xval, Yval, Y_std,S, minibatch_size,pred_full_cov):
        super().__init__(file_writer, model)
        self.Xt = Xt
        self.Yt = Yt
        self.X = X
        self.Y = Y
        self.Y_std = Y_std
        self.Xval = Xval
        self.Yval = Yval
        self.full_cov = pred_full_cov
        self._full_test_err = tf.placeholder(gpflow.settings.float_type, shape=())
        self._full_test_lik = tf.placeholder(gpflow.settings.float_type, shape=())
        
        self._full_train_err = tf.placeholder(gpflow.settings.float_type, shape=())
        self._full_train_lik = tf.placeholder(gpflow.settings.float_type, shape=())
        
        self._full_val_err = tf.placeholder(gpflow.settings.float_type, shape=())
        self._full_val_lik = tf.placeholder(gpflow.settings.float_type, shape=())
        
        self._summary = tf.summary.merge([tf.summary.scalar("test_rmse", self._full_test_err),
                                         tf.summary.scalar("test_lik", self._full_test_lik),
                                         tf.summary.scalar("train_rmse", self._full_train_err),
                                         tf.summary.scalar("train_lik", self._full_train_lik),
                                         tf.summary.scalar("val_rmse", self._full_val_err),
                                         tf.summary.scalar("val_lik", self._full_val_lik)])
        self.S = S
        self.minibatch_size = minibatch_size
    
    def batch_assess(self,assess_model, X, Y):
        n_batches = max(ceil(X.shape[0]/self.minibatch_size), 1)
        lik, sq_diff = [], []
        for X_batch, Y_batch in zip(np.array_split(X, n_batches), np.array_split(Y, n_batches)):
            l, sq = assess_model(X_batch, Y_batch)
            lik.append(l)
            sq_diff.append(sq)
        lik = np.concatenate(lik, 0)
        sq_diff = np.array(np.concatenate(sq_diff, 0), dtype=float)
        return np.average(lik), np.average(sq_diff)**0.5

    def assess_sampled(self,X_batch, Y_batch):
        if self.full_cov:
            m, v = self.model.predict_y_full_cov(X_batch, self.S)
        else:
            m, v = self.model.predict_y(X_batch, self.S)
        S_lik = np.sum(norm.logpdf(Y_batch*self.Y_std, loc=m*self.Y_std, scale=self.Y_std*v**0.5), 2)
        lik = logsumexp(S_lik, 0, b=1/float(self.S))

        mean = np.average(m, 0)
        sq_diff = self.Y_std**2*((mean - Y_batch)**2)
        return lik, sq_diff
    
    def run(self, context: mon.MonitorContext, *args, **kwargs) -> None:        
        test_lik, test_err = self.batch_assess(self.assess_sampled, self.Xt, self.Yt)            
        train_lik, train_err = self.batch_assess(self.assess_sampled,self.X,self.Y)
        val_lik, val_err = self.batch_assess(self.assess_sampled, self.Xval, self.Yval)
        self._eval_summary(context, {self._full_test_err: test_err, self._full_test_lik: test_lik,self._full_train_err: train_err, self._full_train_lik: train_lik, self._full_val_lik: val_lik, self._full_val_err: val_err})

class CustomLmlToTensorBoardTask(mon.BaseTensorBoardTask):
    """
    Monitoring task that creates a TensorBoard with just one scalar value -
    the unbiased estimator of the evidence lower bound (ELBO or LML).
    The LML is averaged over a number of minimatches. The input dataset is split into the specified
    number of sequential minibatches such that every datapoint is used exactly once. The set of
    minibatches doesn't change from one iteration to another.
    The task can display the progress of calculating LML at each iteration (how many minibatches
    are left to compute). For that the `tqdm' progress bar should be installed (pip install tqdm).
    """

    def __init__(self, file_writer, model):
        """
        :param model: Model tensor
        :param file_writer: Event file writer object.
        
        """

        super().__init__(file_writer, model)
        self._full_lml = tf.placeholder(settings.float_type, shape=())
        self._pred_term = tf.placeholder(settings.float_type, shape=())
        self._lik_term = tf.placeholder(settings.float_type, shape=())
        self._kl_term = tf.placeholder(settings.float_type, shape=())

        self._summary = tf.summary.merge([tf.summary.scalar(model.name + '/full_lml', self._full_lml),
                                          tf.summary.scalar(model.name + '/pred_term', self._pred_term),
                                         tf.summary.scalar(model.name + '/lik_term', self._lik_term),
                                         tf.summary.scalar(model.name + '/kl_term', self._kl_term)])

    def run(self, context: mon.MonitorContext, *args, **kwargs) -> None:
        
        lml = self._model.compute_log_likelihood()
        predterm ,likterm,klterm = self._model.get_ELBO_terms()
        
        self._eval_summary(context, {self._full_lml: lml,self._pred_term: predterm,self._lik_term : likterm,self._kl_term:klterm })
    
class LogAction(Action):
    def __init__(self, model, logfile, X, Y, Xt, Yt, Xval, Yval, Y_std,S, minibatch_size,pred_full_cov):
        self.model = model
        self.logfile = logfile
        self.header = ['iter', 'elbo', 'train_lik', 'train_rmse', 'test_lik', 'test_rmse', 'val_lik', 'val_rmse']
        self.Xt = Xt
        self.Yt = Yt
        self.X = X
        self.Y = Y
        self.Y_std = Y_std
        self.Xval = Xval
        self.Yval = Yval
        self.full_cov = pred_full_cov
        self.S = S
        self.minibatch_size = minibatch_size
        with open(self.logfile, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            dict_writer = DictWriter(write_obj, fieldnames=self.header)
            dict_writer.writeheader()

    def batch_assess(self,assess_model, X, Y):
        n_batches = max(ceil(X.shape[0]/self.minibatch_size), 1)
        lik, sq_diff = [], []
        for X_batch, Y_batch in zip(np.array_split(X, n_batches), np.array_split(Y, n_batches)):
            l, sq = assess_model(X_batch, Y_batch)
            lik.append(l)
            sq_diff.append(sq)
        lik = np.concatenate(lik, 0)
        sq_diff = np.array(np.concatenate(sq_diff, 0), dtype=float)
        return np.average(lik), np.average(sq_diff)**0.5

    def assess_sampled(self,X_batch, Y_batch):
        if self.full_cov:
            m, v = self.model.predict_y_full_cov(X_batch, self.S)
        else:
            m, v = self.model.predict_y(X_batch, self.S)
        S_lik = np.sum(norm.logpdf(Y_batch*self.Y_std, loc=m*self.Y_std, scale=self.Y_std*v**0.5), 2)
        lik = logsumexp(S_lik, 0, b=1/float(self.S))

        mean = np.average(m, 0)
        sq_diff = self.Y_std**2*((mean - Y_batch)**2)
        return lik, sq_diff

    def run(self, ctx):
        row = collections.defaultdict(lambda : '-')
        likelihood = ctx.session.run(self.model.likelihood_tensor)
        row['elbo'] = likelihood
        row['iter'] = ctx.iteration
        test_lik, test_err = self.batch_assess(self.assess_sampled, self.Xt, self.Yt)            
        train_lik, train_err = self.batch_assess(self.assess_sampled,self.X,self.Y)
        val_lik, val_err = self.batch_assess(self.assess_sampled, self.Xval, self.Yval)
        row['train_lik'] = train_lik
        row['train_rmse'] = train_err
        row['test_lik'] = test_lik
        row['test_rmse'] = test_err
        row['val_lik'] = val_lik
        row['val_rmse'] = val_err
        with open(self.logfile, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            dict_writer = DictWriter(write_obj, fieldnames=self.header)                        
            # Add dictionary as wor in the csv
            dict_writer.writerow(row)


#----------------------------------------------
def train(dataset,split,models_dgp, names_dgp, X, Y, Xs, Ys, Xval, Yval, Y_std, Stest, minibatch_size, lr,gamma, max_iterations, iteration_steps,suffix,pred_full_cov):
    for m, name in zip(models_dgp, names_dgp):

        print(name)

#         print_task = mon.PrintTimingsTask().with_name('print') .with_condition(mon.PeriodicIterationCondition(iteration_steps)).with_exit_condition(True)

#         sleep_task = mon.SleepTask(0.01).with_name('sleep').with_name('sleep')

# #         saver_task = mon.CheckpointTask('./'+name+'/monitor-saves-'+name+filename_suffix).with_name('saver').with_condition(mon.PeriodicIterationCondition(iteration_steps)).with_exit_condition(True)


#         file_writer = mon.LogdirWriter(dataset+'/'+suffix+'/'+name+'/model-tensorboard-'+name+suffix+str(split),filename_suffix ='.txt')

#         model_tboard_task = mon.ModelToTensorBoardTask(file_writer, m).with_name('model_tboard')        .with_condition(mon.PeriodicIterationCondition(iteration_steps)).with_exit_condition(True)

# #         qmean_norm_task = mon.VectorFuncToTensorBoardTask(file_writer, m.get_qMeanNorm, name+'/qMeanNorm' + '/layer',len(m.layers)).with_condition(mon.PeriodicIterationCondition(iteration_steps)).with_exit_condition(True)

# #         qvar_norm_task = mon.VectorFuncToTensorBoardTask(file_writer, m.get_qVarNorm, name+'/qVarNorm' + '/layer',len(m.layers)).with_condition(mon.PeriodicIterationCondition(iteration_steps)).with_exit_condition(True)

#         lml_tboard_task = CustomLmlToTensorBoardTask(file_writer, m).with_name('lml_tboard')        .with_condition(mon.PeriodicIterationCondition(iteration_steps)).with_exit_condition(True)

#         test_performance_tboard_task = CustomTestPerformanceTensorBoardTask(file_writer, m,X,Y, Xs, Ys, Xval, Yval,Y_std, Stest,minibatch_size,pred_full_cov).with_name('custom_tboard').with_condition(mon.PeriodicIterationCondition(iteration_steps))    .with_exit_condition(True)

#         monitor_tasks = [ print_task, model_tboard_task, lml_tboard_task, test_performance_tboard_task, sleep_task]#,qmean_norm_task,qvar_norm_task,saver_task]

#         session = m.enquire_session()
#         global_step = mon.create_global_step(session)

#         optimiser = AdamOptimizer(lr)
#         with mon.Monitor(monitor_tasks, session,global_step, print_summary=True) as monitor:
#             optimiser.minimize(m, step_callback=monitor, maxiter=max_iterations, global_step=global_step)

#         file_writer.close()
        logfilepath = dataset+'/'+suffix+'/'+name
        if not os.path.exists(logfilepath):
            os.makedirs(logfilepath)
        logfile = logfilepath+'/model-adam_natgrad-log-'+name+suffix+str(split)+'.csv'
        callback = LogAction(m, logfile, X,Y, Xs, Ys, Xval, Yval,Y_std, Stest,minibatch_size,pred_full_cov)
        
        ng_vars = [[ m.layers[-1].qF_S_mean, m.layers[-1].qF_S_var_sqrt]]
        for v in ng_vars[0]:
                v.set_trainable(False)
        
#         ng_vars = [[l.qF_S_mean, l.qF_S_var_sqrt] for l in m.layers]
#         for var in ng_vars:
#             for v in var:
#                 v.set_trainable(False)
        ng_action = NatGradOptimizer(gamma).make_optimize_action(m, var_list=ng_vars)
        adam_action = AdamOptimizer(lr).make_optimize_action(m)

        actions = [ng_action, adam_action, callback]
#         actions = actions if callback is None else actions + [callback]

        Loop(actions, stop = max_iterations, step = iteration_steps)()

#---------------------------------------------------------        
        
# To compute Test Lik, Test RMSE
def batch_assess(assess_model, model, X, Y, Ystd, minibatch_size, S,pred_full_cov):
        n_batches = max(ceil(X.shape[0]/minibatch_size), 1)
        lik, sq_diff = [], []
        for X_batch, Y_batch in zip(np.array_split(X, n_batches), np.array_split(Y, n_batches)):
            l, sq = assess_model(model, X_batch, Y_batch, Ystd, S,pred_full_cov)
            lik.append(l)
            sq_diff.append(sq)
        lik = np.concatenate(lik, 0)
        sq_diff = np.array(np.concatenate(sq_diff, 0), dtype=float)
        return np.average(lik), np.average(sq_diff)**0.5

def assess_sampled(model, X_batch, Y_batch, Ystd,S,pred_full_cov):
        if pred_full_cov:
            m, v = model.predict_y_full_cov(X_batch, S)
        else:
            m, v = model.predict_y(X_batch, S)
        S_lik = np.sum(norm.logpdf(Y_batch*Ystd, loc=m*Ystd, scale=Ystd*v**0.5), 2)
        lik = logsumexp(S_lik, 0, b=1/float(S))

        mean = np.average(m, 0)
        sq_diff = Ystd**2*((mean - Y_batch)**2)
        return lik, sq_diff

        

def test(model, name, Xs, Ys, Y_std, num_samples_test, minibatch_size,pred_full_cov):
    test_lik, test_err = batch_assess(assess_sampled, model, Xs, Ys, Y_std, minibatch_size, num_samples_test,pred_full_cov) 
    print(name+'Test Lik:',test_lik)
    print(name+'Test RMSE:',test_err)
    
#----------------------------------------------------    
import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '-d', nargs = 1, required = True, help = 'Name of Dataset')
    parser.add_argument('-max_iter', '-m', nargs = 1, type = int, required = True, help = 'Max number of training iterations')
#     parser.add_argument('-gpu', nargs =1, required = True, help = 'comma seprated list of required GPU devices')
    parser.add_argument('-layers', '-l', nargs=2,type =int,help = 'Range of Layers for multiple models Ex: input:2 4 for range(2,5),2 2 for range(2,3)',required =True)
    parser.add_argument('-subset', '-s', nargs = 1, type = int, required = True, help = 'subset size')
    parser.add_argument('-batch', '-b', nargs = 1, type = float, default = [1000.], help = 'Batch size(default:1000)')
    parser.add_argument('-Strain', '-s1', nargs = 1, type = int, default = [10], help = '# Samples for MC sampling during training(default:10)')
    parser.add_argument('-Stest', '-s2', nargs = 1, type = int, default = [50], help = '# Samples for MC sampling during testing(default:50)')

    parser.add_argument('-lik_var', '-lv', nargs = 1, type = float, default = [0.001], help = 'Likelihood variance(default:0.01)')

#     parser.add_argument('-layer_var', '-lrv', nargs = 1, type = float, default = [10**(-5)], help = 'Layer variance')
    parser.add_argument('-kern_var','-kv', nargs = "*", type = float, required = True, help = 'RBF Kernel variance(default:2.)')
    parser.add_argument('-kern_lengthscale','-kls', nargs = "*", type = float, required = True, help = 'RBF Kernel lengthscale(default:2.)')
    parser.add_argument('-iter_steps','-iter', nargs = 1, type = int, default = [100], help = 'Iteration steps(default:100)')
    parser.add_argument('-learning_rate','-lr', nargs = 1, type = float, default = [0.01], help = 'learning rate(default:0.01)')
    parser.add_argument('-gamma','-gm', nargs = 1, type = float, default = [0.1], help = 'gamma for natgrad (default:0.1)')
    parser.add_argument('-suffix','-sx', nargs = 1, default = [''], help = 'log filename suffix(default:'')')
    parser.add_argument('-trainfullcovar','-Tfcv',action = 'store_true', help = 'Train Full covariance(default: F)')
    parser.add_argument('-predfullcovar','-Pfcv',action = 'store_true', help = 'Prediction Full covariance(default: F)')
    parser.add_argument('-back_constraint','-bc',action = 'store_true', help = 'Back constraint(T/F)default F')
    parser.add_argument('-full_mean_field','-fmf',action = 'store_true', help = 'FullMean Feild assumption(T/F)default F')
    parser.add_argument('-activation','-act', nargs = 1, default = [None], help = 'Activation function if back constraint True(tanh,softplus)')
    parser.add_argument('-split','-split', nargs = 1, type = int,required = True, help = 'split number')
    
    args = parser.parse_args()
    
#     print('catch!!')
    
#     from tensorflow.python.client import device_lib
#     os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu[0]
#     print(device_lib.list_local_devices())
    
    if tf.test.is_built_with_cuda():
        print("The installed version of TensorFlow includes GPU support.")
    else:
        print("The installed version of TensorFlow does not include GPU support.")

    tf.logging.set_verbosity(0)
    
    
    subset_size = args.subset[0]
    split = args.split[0]
    dataset = args.dataset[0]
    if not os.path.exists(dataset):
        os.mkdir(dataset)
    
    
    if args.trainfullcovar:
        print('Train Full covariance True')
    else:
        print('Train full covariance False')

    X, Y, Xs, Ys, Xval, Yval, Y_std = load_data(args.dataset[0],split)
    X, X_sub, Y, Y_sub = select_subset(X,  Y, subset_size)
    minibatch_size = args.batch[0]-subset_size


    custom_config = settings.get_settings()
    custom_config.numerics.jitter = 10**(-5)
    custom_config.float_type = np.float64

    with settings.temp_settings(custom_config):
        models_dgp, names_dgp = make_dgp_models(args.dataset[0], X, Y, X_sub,Y_sub,subset_size,minibatch_size, args.kern_var, args.kern_lengthscale, range(args.layers[0],args.layers[1]+1), args.lik_var[0], args.Strain[0],train_full_cov = args.trainfullcovar,activation = args.activation[0], back_constraint =args.back_constraint,full_mean_field = args.full_mean_field)

    train(dataset,split,models_dgp, names_dgp, X, Y, Xs, Ys, Xval, Yval, Y_std, args.Stest[0], minibatch_size, args.learning_rate[0], args.gamma[0], args.max_iter[0], args.iter_steps[0], args.suffix[0],args.predfullcovar)

    for m,n in zip(models_dgp,names_dgp):
        test(m,n, Xs, Ys, Y_std, args.Stest[0], minibatch_size,args.predfullcovar)
        #release gpu memory
#         device = cuda.get_current_device()
#         device.reset()

if __name__ == "__main__":
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:
#             # Currently, memory growth needs to be the same across GPUs
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#         except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#             print(e)
    main()