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

import tensorflow as tf
tf.set_random_seed(0)
import numpy as np
np.random.seed(0)

import gpflow
from gpflow.params import DataHolder, Minibatch
from gpflow import autoflow, params_as_tensors, ParamList
from gpflow.models.model import Model
from gpflow.mean_functions import Identity, Linear
from gpflow.mean_functions import Zero
from gpflow.quadrature import mvhermgauss
from gpflow.likelihoods import Gaussian
from gpflow import logdensities
from gpflow import settings
float_type = settings.float_type

from core.utils import reparameterize
from core.utils import get_subsets
from core.utils import BroadcastingLikelihood
from core.sod_layers_initialization import init_layers


class DGP_Base(Model):
    """
    The base class for Deep Gaussian process models.

    Implements a Monte-Carlo variational bound and convenience functions.
    
    Credits : https://github.com/ICL-SML/Doubly-Stochastic-DGP/blob/master/doubly_stochastic_dgp/dgp.py

    """
    def __init__(self, X, Y, likelihood, layers,full_cov = False,
                 minibatch_size=None,
                 num_samples=1, num_data=None,
                 **kwargs):
        Model.__init__(self, **kwargs)
        self.num_samples = num_samples
        self.full_cov = full_cov
        self.num_data = num_data or X.shape[0]
        self.minibatch_size = minibatch_size if X.shape[0] > minibatch_size else None 
        if self.minibatch_size:
            print("using mini batches!")  
            self.X = Minibatch(X, minibatch_size, seed=0)                   
            self.Y = Minibatch(Y, minibatch_size, seed=0)            
        else:
            print("using full batch!")
            self.X = DataHolder(X, fix_shape = True)
            self.Y = DataHolder(Y, fix_shape = True)      
        
        self.likelihood = BroadcastingLikelihood(likelihood)
        self.layers = ParamList(layers)

    @params_as_tensors
    def propagate(self, X, full_cov=False, S=1, zs=None):
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1])
        Fs, Fmeans, Fvars = [], [], []
        F = sX
        zs = zs or [None, ] * len(self.layers)
        
        for layer, z in zip(self.layers, zs):
            F, Fmean, Fvar = layer.sample_from_conditional(F, z=z, full_cov=full_cov)

            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)

        return Fs, Fmeans, Fvars

    @params_as_tensors
    def _build_predict(self, X, full_cov = False, S=1):
        Fs, Fmeans, Fvars = self.propagate(X, full_cov=full_cov, S=S)
        return Fmeans[-1], Fvars[-1]

    def E_log_p_Y(self, X, Y):
        """
        Calculate the expectation of the data log likelihood under the variational distribution
         with MC samples
        """
        print('start E log py')
        Fmean, Fvar = self._build_predict(X, full_cov=self.full_cov, S=self.num_samples)
        #fmean and fvar has to be SND 
        var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y)  # S, N, D
        # print("E log py output:",var_exp)
        return tf.reduce_mean(var_exp, 0)  # S, N, D -mean-> N, D, mean over S

    @params_as_tensors
    def _build_likelihood(self):
        L = tf.reduce_sum(self.E_log_p_Y(self.X, self.Y))
        KL = tf.reduce_sum([layer.KL() for layer in self.layers])
        scale = tf.cast(self.num_data, float_type)
        scale /= tf.cast(tf.shape(self.X)[0], float_type)  # minibatch size
        return L * scale - KL

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_f(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=False, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_f_full_cov(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=True, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_all_layers(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=False, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_all_layers_full_cov(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=True, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_y(self, Xnew, num_samples):  
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        #fmean and fvar has to be SND to likelihood
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)
    
    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_y_full_cov(self, Xnew, num_samples):  
        Fmean, Fvar = self._build_predict(Xnew, full_cov=True, S=num_samples)
        #fmean and fvar has to be SND to likelihood
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    @autoflow((float_type, [None, None]), (float_type, [None, None]), (tf.int32, []))
    def predict_density(self, Xnew, Ynew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        #fmean and fvar has to be SND while passing to likelihood
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew)
        log_num_samples = tf.log(tf.cast(num_samples, float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)
    
    @autoflow((float_type, [None, None]), (float_type, [None, None]), (tf.int32, []))
    def predict_density_full_cov(self, Xnew, Ynew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=True, S=num_samples)
        #fmean and fvar has to be SND while passing to likelihood
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew)
        log_num_samples = tf.log(tf.cast(num_samples, float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)

class SOD_DGP(DGP_Base):
    def __init__(self, X, Y, X_sub, Y_sub,subset_size, kernels, likelihood, layer_variance = 10**(-5), 
                 num_outputs = None,
                 mean_function = Zero(), white = False,back_constraint = False, activation = None, full_mean_field = False,**kwargs):

        layers = init_layers(kernels, num_outputs=num_outputs, variance = layer_variance,subset_size = subset_size, mean_function=mean_function, white=white,back_constraint = back_constraint, activation = activation, full_mean_field = full_mean_field )       
        DGP_Base.__init__(self, X, Y, likelihood, layers,**kwargs)

        self.X_sub = DataHolder(X_sub, fix_shape = True)
        self.Y_sub = DataHolder(Y_sub, fix_shape = True) 
        
        self.back_constraint = back_constraint

    @params_as_tensors
    def compute_KL_terms(self,S = 1):
        result =0
        Xs = tf.expand_dims(self.X_sub ,0)
        Ys = self.Y_sub
        KL = []       
        count = 0  
        
        for layer in self.layers:
            count = count +1
            if(count < len(self.layers)):
                KL.append(layer.KL(Xs))
                Xs = layer.get_latent_subset(self.full_cov,S=S)
            else:
                KL.append(layer.KL(Xs,Ys,final_layer=True))
        
        result = tf.reduce_sum(KL)
        return result

    @params_as_tensors
    def propagate(self, X, full_cov=False, S=1, zs=None):      # need to update this obtain q(Fsbar)
        print('propogate start')        
        # print(X)
        # print(self.X_sub)
        temp = tf.shape(X)[0]
        # print(temp)
        X = tf.concat([self.X_sub,X],0)       #stacking subset of data to training or test data             
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1])
        # print(sX)
        
        D= tf.shape(X)[1]
        # print("D",D)
        
        Fs, Fmeans, Fvars = [], [], []        
        F = sX
        zs = zs or [None, ] * len(self.layers)
        ctr=0                   
        for layer, z in zip(self.layers, zs):
            ctr = ctr + 1
            D= tf.shape(F)[2] # dim of latent representation
            if(ctr < len(self.layers)):  
                F_sub = tf.slice(F, [0, 0, 0], [S, tf.shape(self.X_sub)[0], D])
                #F[:][:X_sub shape][:]
                # print("F_sub.shape",F_sub.shape)
                F = tf.slice(F, [0, tf.shape(self.X_sub)[0], 0], [S, temp, D])
                #F[:][xsub shape:355][:]               
                # print("F.shape",F.shape)
                F, Fmean, Fvar = layer.sample_from_conditional(F, z,F_sub, full_cov=full_cov)
            else:               
                F_sub = tf.slice(F, [0, 0, 0], [S, tf.shape(self.X_sub)[0], D])
                #F[:][:X_sub shape][:]
                # print("F_sub",F_sub)
                F = tf.slice(F, [0, tf.shape(self.X_sub)[0], 0], [S, temp, D])
                #F[:][xsub shape:355][:]                             
                sY_sub =tf.expand_dims(self.Y_sub,0)
                sY_sub = tf.tile(sY_sub,[S, 1, 1])
                # after final layer fmean and fvar has to be SND to pass it to likelihood, hence passing full_cov = False        
                F, Fmean, Fvar = layer.sample_from_conditional(F, z, F_sub, sY_sub, full_cov=False, final_layer = True)
            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)

        print('propogate end')
        return Fs, Fmeans, Fvars
    
    @params_as_tensors
    def _build_likelihood(self):   # final elbo expression result
        print('build likelihood start')        
        if self.back_constraint:
            qmean = self.Y_sub
            qvar = self.Y_sub
            for l in self.layers[-1::-1]:
                qmean, qvar = l.propogate_back_constraint(qmean,qvar)
            
        logPredProb = tf.reduce_sum(self.E_log_p_Y(self.X, self.Y))     #N, D -sum-> scalar     
        
        # keep check on dimensions***
        print("start logLik computation")
        #fmean and fvar has to be SND to pass it to likelihood, hence passing full_cov = False
        var,mean = self.layers[-1].computePosteriorQhat(final_layer=True,full_cov=False,Y_sub=self.Y_sub, meanSND = True, varSNND=True)
        print(mean)
        print(var)
        print(self.Y_sub)
        var_exp = self.likelihood.variational_expectations(mean, var, self.Y_sub)  # S, N, D 
        print( "var_exp output:",var_exp)
        print( "var_exp shape:",var_exp.shape)
        
        logLik = tf.reduce_sum(tf.reduce_mean(var_exp, 0))  # S,N,D -mean-> N, D -sum-> scalar ,  reduce mean over S

        KL = self.compute_KL_terms(S = self.num_samples)
        
        scale = tf.cast(self.num_data, float_type)
        scale /= tf.cast(tf.shape(self.X)[0], float_type)  # minibatch size

        print('build likelihood end')
        
        self.predterm = logPredProb* scale
        self.likterm = logLik
        self.KLterm = KL

        return logPredProb * scale + logLik - KL 

    @autoflow()
    def get_ELBO_terms(self):
        return self.predterm ,self.likterm, self.KLterm
    
    @autoflow()
    @params_as_tensors
    def get_qMeanNorm(self):
        qMeanNorm=[]        
        for l in self.layers:
            qMeanNorm.append(tf.norm(l.qF_S_mean))        
        return tf.convert_to_tensor(qMeanNorm)
    @autoflow()
    @params_as_tensors
    def get_qVarNorm(self):
        qVarNorm=[]        
        for l in self.layers:        
            qVarNorm.append(tf.norm(l.qF_S_var_sqrt))
        return tf.convert_to_tensor(qVarNorm)
    




