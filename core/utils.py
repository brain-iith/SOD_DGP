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
#import tensorflow_probability as tfp

from gpflow import settings
from gpflow import params_as_tensors, Parameterized ,autoflow
from gpflow.likelihoods import Gaussian

from scipy.cluster.vq import kmeans2
import numpy as np
np.random.seed(0)

def compute_closest_point(Centroids,Index_Centroids,X):
    Closest = []
    for c in range(Centroids.shape[0]):
        Indeces = np.where(Index_Centroids==c)[0] #returns the index of training examples which are closer to a particular centroid 'c'
        min_dis = 19349040
        #min index initialised with random point
        np.random.seed(c)
        min_index = np.random.randint(0,Indeces.shape[0])
            
        for i in range(Indeces.shape[0]):
            dis = np.linalg.norm(Centroids[c]-X[Indeces[i]])
            if dis < min_dis:
                min_dis = dis
                min_index = i
        
        Closest.append(Indeces[min_index])
    #default seed
    np.random.seed(0)
            
    return Closest
            
def get_subsets(X, Y, subset_size):
    if(subset_size < (X.shape[0]/2)):
        Z = kmeans2(X, subset_size, minit='points')#returns centroids and also the index of the centroid closest to each training example
        
        Closest = compute_closest_point(Z[0],Z[1],X)

        X_sub = X[Closest][:]
        Y_sub = Y[Closest][:]
        remaining = list(set(range(X.shape[0])) - set(Closest))
        X = X[remaining][:]
        Y = Y[remaining][:]
        return X, X_sub, Y, Y_sub
    else:
        raise ValueError('subset size can be atmost half of total total data!')
        
def get_random_subset(X, Y, subset_size):
    if(subset_size < (X.shape[0]/2)):
        subset_indeces = np.random.randint(0,X.shape[0],subset_size)
        X_sub = X[subset_indeces][:]
        Y_sub = Y[subset_indeces][:]
        remaining = list(set(range(X.shape[0])) - set(subset_indeces))
        X = X[remaining][:]
        Y = Y[remaining][:]
        return X, X_sub, Y, Y_sub
    else:
        raise ValueError('subset size can be atmost half of total total data!')

# following www.cse.iitb.ac.in/~aruniyer/kldivergencenormal.pdf
#@autoflow((settings.float_type,[None,None]),(settings.float_type,[None,None]),(settings.float_type,[None,None]),(settings.float_type,[None,None]))
def multivariate_gauss_cross_entropy2(meanA, covA, meanB, covB):
    K = tf.shape(covA,out_type = tf.float64)[0] 
    # print(K)
    temp = tf.matmul(meanA,meanA,transpose_b=True)
    
    cholB = tf.cholesky(covB)

    traceTerm = 0.5* tf.trace(tf.linalg.cholesky_solve(cholB,tf.add(covA,temp)))
    priorLogDeterminantTerm = 0.5*tf.linalg.slogdet(covB)[1]
    constantTerm = 0.5*K*np.log(2*np.pi)
    delta = meanB - meanA

    term1 = -0.5 * tf.matmul(tf.transpose(meanB), tf.linalg.cholesky_solve(cholB, meanA))
    term2 = 0.5 * tf.matmul(tf.transpose(delta), tf.linalg.cholesky_solve(cholB, meanB))
    

    return (constantTerm + priorLogDeterminantTerm + traceTerm + term1 + term2)


# Credits to GPFlow : gpflow/tests/test_variational.py
#@autoflow((settings.float_type,[None,None]),(settings.float_type,[None,None]),(settings.float_type,[None,None]),(settings.float_type,[None,None]))
def multivariate_gauss_KL2(meanA, covA, meanB, covB):
# KL[ qA | qB ] = E_{qA} \log [qA / qB] where qA and aB are
# K dimensional multivariate normal distributions.
# Analytically tractable and equal to...
# 0.5 * (Tr(covB^{-1} covA) + (meanB - meanA)^T covB^{-1} (meanB - meanA)
#        - K + log(det(covB)) - log (det(covA)))
    K = tf.shape(covA,out_type = tf.float64)[0] 
    cholB = tf.cholesky(covB)
    traceTerm = 0.5 * tf.trace(tf.linalg.cholesky_solve(cholB, covA))
    delta = meanB - meanA
    mahalanobisTerm = 0.5 * tf.matmul(tf.transpose(delta), tf.linalg.cholesky_solve(cholB, delta))
    constantTerm = -0.5 * K
    priorLogDeterminantTerm = 0.5*tf.linalg.slogdet(covB)[1]
    variationalLogDeterminantTerm = -0.5 * tf.linalg.slogdet(covA)[1]
    result = (traceTerm +
            mahalanobisTerm +
            constantTerm +
            priorLogDeterminantTerm +
            variationalLogDeterminantTerm)
    # print("computed KL = ")
    # print(result)
    return result

def multivariate_gauss_KL(meanA, covA, sqrtA, meanB, covB,sqrtB):
    #mean D x N
    #covar D X N X N
    if(sqrtA):
        Q = tf.contrib.distributions.MultivariateNormalTriL( loc=meanA, scale_tril=covA)
    else:
        Q = tf.contrib.distributions.MultivariateNormalFullCovariance( loc=meanA, covariance_matrix=covA)
    if(sqrtB):
        P = tf.contrib.distributions.MultivariateNormalTriL( loc=meanB, scale_tril=covB)
    else:
        P = tf.contrib.distributions.MultivariateNormalFullCovariance( loc=meanB, covariance_matrix=covB)
    
    result = tf.reduce_sum(Q.kl_divergence(P), name = 'kl_divergence_computation')
#     result = tf.reduce_mean(tf.contrib.distributions.kl_divergence( Q, P,allow_nan_stats = False, name = 'kl_divergence_computation'))

    # print("computed KL =")
    # print(result)
    return result

def multivariate_gauss_cross_entropy(meanA, covA, sqrtA, meanB, covB,sqrtB):
    if(sqrtA):
        P = tf.contrib.distributions.MultivariateNormalTriL( loc=meanA, scale_tril=covA)
    else:
        P = tf.contrib.distributions.MultivariateNormalFullCovariance( loc=meanA, covariance_matrix=covA)
    if(sqrtB):
        Q = tf.contrib.distributions.MultivariateNormalTriL( loc=meanB, scale_tril=covB)
    else:
        Q = tf.contrib.distributions.MultivariateNormalFullCovariance( loc=meanB, covariance_matrix=covB)
        
    result = tf.reduce_sum(P.cross_entropy(other = Q, name = 'cross_entropy_computation'))

    # print("computed CE =")
    # print(result)
    return result


def reparameterize(mean, var, z, full_cov=True):
    """
    Implements the 'reparameterization trick' for the Gaussian, either full rank or diagonal

    If the z is a sample from N(0, 1), the output is a sample from N(mean, var)

    If full_cov=True then var must be of shape S,N,N,D and the full covariance is used. Otherwise
    var must be S,N,D and the operation is elementwise
    
    Credits : https://github.com/ICL-SML/Doubly-Stochastic-DGP/blob/master/doubly_stochastic_dgp/utils.py

    :param mean: mean of shape S,N,D
    :param var: covariance of shape S,N,D or S,N,N,D
    :param z: samples form unit Gaussian of shape S,N,D
    :param full_cov: bool to indicate whether var is of shape S,N,N,D or S,N,D
    :return sample from N(mean, var) of shape S,N,D      
    """
    if var is None:
        return mean

    if full_cov is False:
        return mean + z * (var + settings.jitter) ** 0.5

    else:
        # print("reparameterize!")
        # print(mean)
        # print(var)
        S, N, D = tf.shape(mean)[0], tf.shape(mean)[1], tf.shape(mean)[2] # var is SNND
        mean = tf.transpose(mean, (0, 2, 1))  # SND -> SDN
        var = tf.transpose(var, (0, 3, 1, 2))  # SNND -> SDNN
        I = settings.jitter * tf.eye(N, dtype=settings.float_type)[None, None, :, :] # 11NN
        chol = tf.cholesky(var + I)  # SDNN
        z_SDN1 = tf.transpose(z, [0, 2, 1])[:, :, :, None]  # SND->SDN1
        f = mean + tf.matmul(chol, z_SDN1)[:, :, :, 0]  # SDN(1)
        return tf.transpose(f, (0, 2, 1)) # SND


class BroadcastingLikelihood(Parameterized):
    """
    A wrapper for the likelihood to broadcast over the samples dimension. The Gaussian doesn't
    need this, but for the others we can apply reshaping and tiling.

    With this wrapper all likelihood functions behave correctly with inputs of shape S,N,D,
    but with Y still of shape N,D
    
    Credits : https://github.com/ICL-SML/Doubly-Stochastic-DGP/blob/master/doubly_stochastic_dgp/utils.py
    """
    def __init__(self, likelihood):
        Parameterized.__init__(self)
        self.likelihood = likelihood

        if isinstance(likelihood, Gaussian):
            self.needs_broadcasting = False
        else:
            self.needs_broadcasting = True

    def _broadcast(self, f, vars_SND, vars_ND):
        if self.needs_broadcasting is False:
            return f(vars_SND, [tf.expand_dims(v, 0) for v in vars_ND])

        else:
            S, N, D = [tf.shape(vars_SND[0])[i] for i in range(3)]
            vars_tiled = [tf.tile(x[None, :, :], [S, 1, 1]) for x in vars_ND]

            flattened_SND = [tf.reshape(x, [S*N, D]) for x in vars_SND]
            flattened_tiled = [tf.reshape(x, [S*N, -1]) for x in vars_tiled]

            flattened_result = f(flattened_SND, flattened_tiled)
            if isinstance(flattened_result, tuple):
                return [tf.reshape(x, [S, N, -1]) for x in flattened_result]
            else:
                return tf.reshape(flattened_result, [S, N, -1])

    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.variational_expectations(vars_SND[0], vars_SND[1], vars_ND[0])
        return self._broadcast(f,[Fmu, Fvar], [Y])

    @params_as_tensors
    def logp(self, F, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.logp(vars_SND[0], vars_ND[0])
        return self._broadcast(f, [F], [Y])

    @params_as_tensors
    def conditional_mean(self, F):
        f = lambda vars_SND, vars_ND: self.likelihood.conditional_mean(vars_SND[0])
        return self._broadcast(f,[F], [])

    @params_as_tensors
    def conditional_variance(self, F):
        f = lambda vars_SND, vars_ND: self.likelihood.conditional_variance(vars_SND[0])
        return self._broadcast(f,[F], [])

    @params_as_tensors
    def predict_mean_and_var(self, Fmu, Fvar):
        f = lambda vars_SND, vars_ND: self.likelihood.predict_mean_and_var(vars_SND[0], vars_SND[1])
        return self._broadcast(f,[Fmu, Fvar], [])

    @params_as_tensors
    def predict_density(self, Fmu, Fvar, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.predict_density(vars_SND[0], vars_SND[1], vars_ND[0])
        return self._broadcast(f,[Fmu, Fvar], [Y])
