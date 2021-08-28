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

import gpflow as gpflow
from gpflow.params import Parameter, Parameterized
from gpflow.conditionals import conditional
from gpflow.features import InducingPoints
from gpflow.kullback_leiblers import gauss_kl
from gpflow.priors import Gaussian as Gaussian_prior
from gpflow import transforms , ParamList
from gpflow import settings
from gpflow.models.gplvm import BayesianGPLVM
from gpflow.expectations import expectation
from gpflow.probability_distributions import DiagonalGaussian
from gpflow import params_as_tensors
from gpflow.logdensities import multivariate_normal
from gpflow.mean_functions import Zero


from core.utils import reparameterize , multivariate_gauss_cross_entropy , multivariate_gauss_KL


class Layer(Parameterized):
    def __init__(self, input_prop_dim=None, **kwargs): 
        """
        A base class for GP layers. Basic functionality for multisample conditional, and input propagation
        
        Credits : https://github.com/ICL-SML/Doubly-Stochastic-DGP/blob/master/doubly_stochastic_dgp/layers.py
        
        :param input_prop_dim: the first dimensions of X to propagate. If None (or zero) then no input prop
        :param kwargs:       
        """
        Parameterized.__init__(self, **kwargs)
        self.input_prop_dim = input_prop_dim
    @params_as_tensors
    def conditional_ND(self, X, X_sub = None, Y_sub = None, full_cov=False,final_layer = False):
        raise NotImplementedError

    def KL(self):
        return tf.cast(0., dtype=settings.float_type)

    def conditional_SND(self, X, X_sub, Y_sub, full_cov=False,final_layer = False):
        """
        A multisample conditional, where X is shape (S,N,D_in), independent over samples S

        if full_cov is True
            mean is (S,N,D_out), var is (S,N,N,D_out)

        if full_cov is False
            mean and var are both (S,N,D_out)

        :param X:  The input locations (S,N,D_in)
        :param X_sub, Y_sub: subset
        :param full_cov: Whether to calculate full covariance or just diagonal
        :return: mean (S,N,D_out), var (S,N,D_out or S,N,N,D_out)
        """
         
        if(final_layer):
            if X_sub is not None and Y_sub is not None:  
                f = lambda a: self.conditional_ND(a[0], a[1], a[2], full_cov=full_cov, final_layer = final_layer)
                mean, var = tf.map_fn(f, (X, X_sub, Y_sub), dtype=(tf.float64, tf.float64))
                return tf.stack(mean), tf.stack(var)
            else:
                raise ValueError('X_sub and Y_sub should be not None')
            
        else:
            if X_sub is not None:   
                f = lambda a: self.conditional_ND(X = a[0], X_sub = a[1], Y_sub = None, full_cov=full_cov, final_layer=final_layer)
                mean, var = tf.map_fn(f, (X, X_sub), dtype=(tf.float64, tf.float64))
                return tf.stack(mean), tf.stack(var)
            else:
                raise ValueError('X_sub should be not None')
   
    def sample_from_conditional(self, X, z=None, X_sub = None, Y_sub = None, full_cov =False,final_layer = False):
        """
        Calculates self.conditional and also draws a sample, adding input propagation if necessary

        If z=None then the tensorflow random_normal function is used to generate the
        N(0, 1) samples, otherwise z are used for the whitened sample points

        :param X: Input locations (S,N,D_in)
        :param full_cov: Whether to compute correlations between outputs
        :param z: None, or the sampled points in whitened representation
        :param X_sub, Y_sub : subset
        :return: mean (S,N,D), var (S,N,N,D or S,N,D), samples (S,N,D)
        """
        mean, var = self.conditional_SND(X, X_sub, Y_sub, full_cov,final_layer)

        # set shapes
        S = tf.shape(X)[0]
        if(final_layer):
            N = tf.shape(X)[1]
        else:
            N = tf.shape(X)[1] + tf.shape(X_sub)[1]
        
        D = self.num_outputs

#         print("sample from conditional")
#         print(X.shape)
#         print(mean.shape)
#         print(var.shape)
#         print(S,N,D)

        mean = tf.reshape(mean, (S, N, D))
        # print(mean.shape)

        if full_cov : 
            var = tf.reshape(var, (S, N, N, D))
        else:
            
            var = tf.reshape(var, (S, N, D))

        if z is None:
            z = tf.random_normal(tf.shape(mean), dtype=settings.float_type)
        samples = reparameterize(mean, var, z, full_cov=full_cov)

        if self.input_prop_dim:
            shape = [tf.shape(X)[0], tf.shape(X)[1], self.input_prop_dim]
            X_prop = tf.reshape(X[:, :, :self.input_prop_dim], shape)

            samples = tf.concat([X_prop, samples], 2)
            mean = tf.concat([X_prop, mean], 2)

            if full_cov:
                shape = (tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[1], tf.shape(var)[3])
                zeros = tf.zeros(shape, dtype=settings.float_type)
                var = tf.concat([zeros, var], 3)
            else:
                var = tf.concat([tf.zeros_like(X_prop), var], 2)

#         print("sample from conditional result")
#         print(X.shape)
#         print(mean.shape)
#         print(var.shape)
#         print(S,N,D)
        return samples, mean, var

class SOD_Layer(Layer):
    def __init__(self, kern, num_outputs, mean_function, subset_size, variance = 1.0, 
                 white=False,qvar_scale_factor = (10**(-3)), input_prop_dim=None, back_constraint = False, wdimin = None, wdimout = None, activation = None, full_mean_field = False, **kwargs):
        Layer.__init__(self, input_prop_dim, **kwargs)
        self.kern = kern
        self.mean_function = mean_function
        self.subset_size = subset_size

        self.num_outputs = num_outputs
        self.white = white
        self.variance = Parameter(variance, transform=transforms.positive, dtype=settings.float_type,name="layer_var") 
        self.activation = activation
        self.Wqmean = None
        self.bqmean = None
        self.Wqvarsqrt = None
        self.bqvarsqrt = None
        self.qF_S_mean = None
        self.qF_S_var_sqrt = None
        self.back_constraint = back_constraint
        np.random.seed(0)
        if self.back_constraint:
            #Diagonal parameterisation, full mean field assumption
            #Xavier Initialisation to weights
            self.Wqmean = Parameter((np.random.rand(wdimout, wdimin)*2-1)*np.sqrt(6./(wdimout+wdimin)), dtype = settings.float_type, name ="Wqmean")
            self.bqmean = Parameter(np.zeros(wdimout), dtype = settings.float_type, name = "bqmean")
            
            
            self.Wqvarsqrt = Parameter((np.random.rand(wdimout, wdimin)*2-1)*np.sqrt(6./(wdimout+wdimin)), dtype = settings.float_type, name ="Wqvarsqrt")
            self.bqvarsqrt = Parameter(np.zeros(wdimout), dtype = settings.float_type, name = "bqvarsqrt")
            
            #implement cholesky factor reparameterisation, non mean field
        else:            
            #ND do tf.transpose(qF_S_mean)[:, :, None] to get DN1
            self.qF_S_mean = Parameter(np.random.rand(self.subset_size,self.num_outputs),dtype=settings.float_type,name="qmean") 
            
            if full_mean_field:
                self.qF_S_var_sqrt = Parameter(np.tile((np.eye(self.subset_size)*qvar_scale_factor)[None, :, :], [self.num_outputs, 1, 1]),transform=transforms.DiagMatrix(self.subset_size),name="qvar")
            else:
                self.qF_S_var_sqrt = Parameter(np.tile((np.eye(self.subset_size)*qvar_scale_factor)[None, :, :], [self.num_outputs, 1, 1]),transform=transforms.LowerTriangular(self.subset_size,num_matrices=self.num_outputs),name="qvar")        

    @params_as_tensors
    def propogate_back_constraint(self, qmean, qvar):
        #qmean, qvar shape N x wdimin to N x wdimout
        qmean = tf.matmul(qmean, self.Wqmean, transpose_b = True) + self.bqmean[None,:]
        qvar = tf.matmul(qvar, self.Wqvarsqrt, transpose_b = True) + self.bqvarsqrt[None,:]
        if self.activation == 'tanh':
            qmean = tf.tanh(qmean)
            qvar = tf.tanh(qvar)
        elif self.activation == 'softplus':
            qmean = tf.math.softplus(qmean)
            qvar = tf.math.softplus(qvar)
        elif self.activation is None:
            #do nothing
            pass
        else:
            raise ValueError('Unsupported activation function')
        
        self.qF_S_mean = qmean #N x D
        self.qF_S_var_sqrt = tf.matrix_diag(tf.transpose(qvar)) # D x N x N
        
        return qmean, qvar
        
    #to obtain Zs samples for layer 1 to L-1
    @params_as_tensors
    def get_latent_subset(self,full_cov,S=1):     
        N_sub = tf.shape(self.qF_S_mean)[0]
        mean = self.qF_S_mean
        mean = tf.tile(tf.expand_dims(mean,0),[S,1,1]) # S,N,D
        # print(mean.shape)

        if full_cov:
            var = tf.stack([tf.matmul(self.qF_S_var_sqrt[i],self.qF_S_var_sqrt[i],transpose_b=True) + tf.eye(N_sub,dtype=settings.float_type)*(self.variance+settings.jitter) for i in range(self.num_outputs)],2)
            var = tf.tile(tf.expand_dims(var,0),[S,1,1,1]) #S,N,N,D
        else:
            var = tf.stack([tf.matrix_diag_part(tf.matmul(self.qF_S_var_sqrt[i],self.qF_S_var_sqrt[i],transpose_b=True) + tf.eye(N_sub,dtype=settings.float_type)*(self.variance+settings.jitter)) for i in range(self.num_outputs)],1)
            var = tf.tile(tf.expand_dims(var,0),[S,1,1]) #S,N,D

        z = tf.random_normal(tf.shape(mean), dtype=settings.float_type)
        samples = reparameterize(mean, var, z, full_cov)
#         print("get latent subset result shapes")
#         print(samples.shape)
#         print(var.shape)
#         print(mean.shape)
        return samples #S,N,D

    @params_as_tensors 
    def computePosteriorQhat(self,final_layer,full_cov,Y_sub,meanSND = False, varSNND = False):
        if(final_layer):
            I = tf.eye(tf.shape(Y_sub)[0],dtype=settings.float_type)
            C = I*(1/self.variance)
            qF_S_var_inverse = tf.cholesky_solve(self.qF_S_var_sqrt[0],I)           
            
            qF_S_var = tf.matmul(self.qF_S_var_sqrt[0],self.qF_S_var_sqrt[0],transpose_b=True)
            qhatF_S_var = self.variance*I - (self.variance**2)*tf.cholesky_solve(tf.cholesky(qF_S_var+self.variance*I),I)
            B = tf.add(tf.matmul(C,Y_sub),tf.matmul(qF_S_var_inverse,(tf.transpose(self.qF_S_mean)[:, :, None])[0]))
            qhatF_S_mean = tf.matmul(qhatF_S_var,B)            

#             print("start posterior compuatation")
#             print("var shape",qhatF_S_var)
#             print("mean shape:",qhatF_S_mean)
            if(not full_cov):
                qhatF_S_var = tf.matrix_diag_part(qhatF_S_var) # NxN -> N
                print("var diag_part shape",qhatF_S_var)
            
            if(meanSND):
                qhatF_S_mean = tf.expand_dims(qhatF_S_mean,0) #1,N,1 SND
            if(varSNND):
                if full_cov:
                    qhatF_S_var = tf.expand_dims(tf.expand_dims(qhatF_S_var,0),3) #1,N,N,1 SNND
                else:
                    qhatF_S_var = tf.expand_dims(tf.expand_dims(qhatF_S_var,0),2) #1,N,1 SND
#             print("expanded dim")
#             print("var shape",qhatF_S_var)
#             print("mean shape:",qhatF_S_mean)

            return qhatF_S_var,qhatF_S_mean
        else:
            return None, None       
    
    @params_as_tensors   
    def conditional_ND(self, X, X_sub = None, Y_sub = None, full_cov=False,final_layer = False):                                       
        if final_layer:                      
            if X_sub is None or Y_sub is None:
                raise ValueError('X_sub and Y_sub should be not None.')                 
            Ksbar = self.kern.K(X) + tf.eye(tf.shape(X)[0],dtype=settings.float_type)*settings.jitter
            Ks = self.kern.K(X_sub) + tf.eye(tf.shape(X_sub)[0],dtype=settings.float_type)*settings.jitter  #is it correct session?                             
            Ksbar_s = self.kern.K(X,X_sub)

            I = tf.eye(tf.shape(Ks)[0],dtype=settings.float_type)
            temp =  tf.matmul(Ksbar_s,tf.cholesky_solve(tf.cholesky(Ks),I))
            V_Sbar_given_S = tf.subtract(Ksbar,tf.matmul(temp,Ksbar_s,transpose_b=True))
            
            qhatF_S_var,qhatF_S_mean = self.computePosteriorQhat(final_layer,True,Y_sub)

            mean = tf.matmul(temp,qhatF_S_mean)
            var = tf.add(V_Sbar_given_S,tf.matmul(temp,tf.matmul(qhatF_S_var,temp,transpose_b=True)))

            if not full_cov:
                var = tf.matrix_diag_part(var) 
        else:
            if X_sub is None:
                raise ValueError('X_sub should be not None.')

            Ksbar = self.kern.K(X)+ tf.eye(tf.shape(X)[0],dtype=settings.float_type)*settings.jitter
            Ks = self.kern.K(X_sub)+ tf.eye(tf.shape(X_sub)[0],dtype=settings.float_type)*settings.jitter                           
            Ksbar_s = self.kern.K(X,X_sub)

            I = tf.eye(tf.shape(Ks)[0],dtype=settings.float_type)
            temp =  tf.matmul(Ksbar_s,tf.cholesky_solve(tf.cholesky(Ks),I))
            V_Sbar_given_S = tf.subtract(Ksbar,tf.matmul(temp,Ksbar_s,transpose_b=True))
                
            mean = [tf.concat([(tf.transpose(self.qF_S_mean)[:, :, None])[i],tf.matmul(temp,(tf.transpose(self.qF_S_mean)[:, :, None])[i])],0)for i in range(self.num_outputs)]

            var = []
            for i in range(self.num_outputs):
                qF_S_var = tf.matmul(self.qF_S_var_sqrt[i],self.qF_S_var_sqrt[i],transpose_b=True)
                var_submat1 = qF_S_var
                var_submat2 = tf.matmul(qF_S_var,temp,transpose_b=True)
                var_submat3 = tf.transpose(var_submat2)
                var_submat4 = tf.add(V_Sbar_given_S,tf.matmul(temp,var_submat2))
                var_temp = tf.concat([tf.concat([var_submat1,var_submat3],0),tf.concat([var_submat2,var_submat4],0)],1)
                var_temp += tf.eye(tf.shape(var_temp)[0],dtype=settings.float_type)*(self.variance+settings.jitter)
                if(full_cov):
                    var.append(var_temp)
                else:
                    var.append(tf.matrix_diag_part(var_temp))

            mean = tf.squeeze(tf.stack(mean,1),2)
            if(full_cov):
                var = tf.stack(var,2)
            else:
                var = tf.stack(var,1)
        # print('conditional ND output')
        # print(mean)
        # print(var)
        return mean, var

    def KL(self, X_sub,Y_sub = None, final_layer=False):
        
        if not final_layer:
            KL = []
            for s in range(X_sub.shape[0]):
                Ks = self.kern.K(X_sub[s])+ tf.eye(tf.shape(X_sub[s])[0],dtype=settings.float_type)*settings.jitter
                pF_S_mean = self.mean_function(X_sub[s])
                
                KL.append(multivariate_gauss_KL(tf.transpose(self.qF_S_mean),self.qF_S_var_sqrt,True,tf.tile(tf.transpose(pF_S_mean),[self.num_outputs,1]),tf.tile(Ks[None,:,:],[self.num_outputs,1,1]),False))
        else:                          
            if(Y_sub is None):
                raise ValueError("Ysub should not be none")
            
            qhatF_S_var,qhatF_S_mean = self.computePosteriorQhat(final_layer,True,Y_sub)           

            KL = []
            for s in range(X_sub.shape[0]):
                    Ks = self.kern.K(X_sub[s])+ tf.eye(tf.shape(X_sub[s])[0],dtype=settings.float_type)*settings.jitter
                    pF_S_mean = self.mean_function(X_sub[s])
                    KL.append(multivariate_gauss_KL(tf.transpose(qhatF_S_mean),tf.expand_dims(qhatF_S_var,0),False,tf.transpose(pF_S_mean),tf.expand_dims(Ks,0),False))
        
        return tf.reduce_mean(KL) # reduce mean vs reduce sum    