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

from gpflow.params import DataHolder, Minibatch
from gpflow import autoflow, params_as_tensors, ParamList
from gpflow.models.model import Model
from gpflow.mean_functions import Identity, Linear
from gpflow.mean_functions import Zero
from gpflow.quadrature import mvhermgauss
from gpflow import settings
float_type = settings.float_type
from core.sod_layer import SOD_Layer

def init_layers(kernels,
                       num_outputs, variance,subset_size,
                       mean_function=Zero(),
                       Layer=SOD_Layer,
                       white=False, back_constraint = False, activation = None, full_mean_field = False):
    num_outputs = num_outputs   
    layers = []
    
    # final layer
    layers.append(Layer(kernels[-1], num_outputs, mean_function,subset_size, variance, white=white,qvar_scale_factor =1, back_constraint = back_constraint, wdimin = num_outputs, wdimout = num_outputs, activation = activation, full_mean_field = full_mean_field))
    dim_out = num_outputs

    for kern_out, kern_in in zip(kernels[-1:0:-1], kernels[-2::-1]):
        dim_in = kern_in.input_dim
        wdimin = dim_out
        dim_out = kern_out.input_dim
        wdimout = dim_out # wdimout is same as dimout of layer, wdimin is same as dimout of prev layer (from back constraint view)
        # print(dim_in, dim_out)
        mf = Zero()       
        layers.insert(0,Layer(kern_in, dim_out, mf,subset_size, variance, white=white,qvar_scale_factor = (10**(-5)), back_constraint = back_constraint, wdimin = wdimin, wdimout = wdimout, activation = activation, full_mean_field = full_mean_field))   
    
    return layers