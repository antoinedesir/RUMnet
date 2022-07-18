#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Layer
from tensorflow.keras import Model, Input
from functools import reduce
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

class MNLModel(tf.keras.Model):

  def __init__(self,paramsArchitecture, paramsModel):
    super(MNLModel, self).__init__()
    self.paramsArchitecture = paramsArchitecture
    self.paramsModel = paramsModel
    
    regularizer = tf.keras.regularizers.L2(self.paramsModel['regularization'])
    
    self.dense = [Dense(paramsArchitecture['width'], activation="elu", 
                          kernel_regularizer=regularizer,
                          use_bias= True) for i in range(self.paramsArchitecture['depth'])]
    
    self.last = Dense(1, activation="linear",
                          kernel_regularizer=regularizer,
                          use_bias= False)
    
  def call(self,inputs):
    
    tol = self.paramsModel['tol']
    n = self.paramsModel['number_products']
    
    U = inputs[:n]
    Z = inputs[n:]
    
    U = [tf.keras.layers.Concatenate()((U[i],)+ Z) for i in range(n)]

    if self.paramsArchitecture['depth']>0:
        
        for k in range(self.paramsArchitecture['depth']):
            U = [self.dense[k](U[i]) for i in range(n)]            
    
    U = [self.last(U[i]) for i in range(n)]                
    
    combined = tf.keras.layers.Concatenate()(U)
    
    return (tol+tf.keras.layers.Activation(activation=tf.nn.softmax)(combined))\
            /(1+ tol*n)



class TasteNet(tf.keras.Model):

  def __init__(self,paramsArchitecture, paramsModel):
    super(TasteNet, self).__init__()
    self.paramsArchitecture = paramsArchitecture
    self.paramsModel = paramsModel
    
    regularizer = tf.keras.regularizers.L2(self.paramsModel['regularization'])
    
    
    self.dense = [Dense(paramsArchitecture['width'], activation="elu", 
                          kernel_regularizer=regularizer,
                          use_bias= True) for i in range(self.paramsArchitecture['depth'])]
    
    self.last = Dense(paramsModel["number_p_features"], activation="linear",
                          kernel_regularizer=regularizer,
                          use_bias= True)
    
    self.beta = Dense(1, activation="linear",
                          kernel_regularizer=regularizer,
                          use_bias= True)    
    
  def call(self,inputs):
    
    tol = self.paramsModel['tol']
    n = self.paramsModel['number_products']
    
    U = inputs[:n]
    Z = tf.keras.layers.Concatenate()(inputs[n:])   
    
    if self.paramsArchitecture['depth']>0:
        
        for k in range(self.paramsArchitecture['depth']):
            Z = self.dense[k](Z) 

    U = [ tf.keras.layers.Dot(axes=1)([self.last(Z),U[i]])\
         + self.beta(U[i]) for i in range(n)]                
    
    combined = tf.keras.layers.Concatenate()(U) 
    
    return (tol+tf.keras.layers.Activation(activation=tf.nn.softmax)(combined))\
            /(1+ tol*n)


class RUMModel(tf.keras.Model):

  def __init__(self, paramsArchitecture, paramsModel):
    super(RUMModel, self).__init__()
    self.paramsArchitecture = paramsArchitecture
    self.paramsModel = paramsModel

    regularizer = tf.keras.regularizers.L2(self.paramsModel['regularization'])
    
    self.dense_x = [[Dense(self.paramsArchitecture['width_eps_x'], 
                            activation="elu",
                            kernel_regularizer=regularizer, 
                            use_bias= True) \
                     for i in range(self.paramsArchitecture['depth_eps_x'])] \
                    for j in range(self.paramsArchitecture['heterogeneity_x'])]
    self.dense_z = [[Dense(self.paramsArchitecture['width_eps_z'], 
                            activation="elu",
                            kernel_regularizer=regularizer, 
                            use_bias= True) for i in range(self.paramsArchitecture['depth_eps_z'])] for j in range(self.paramsArchitecture['heterogeneity_z'])]
    self.utility = [Dense(self.paramsArchitecture['width_u'], 
                        activation="elu", 
                        kernel_regularizer=regularizer, 
                        use_bias= True) for i in range(self.paramsArchitecture['depth_u'])]
    self.last = Dense(1, 
                    activation="linear",
                    use_bias= False)

    
  def call(self,inputs):
    tol = self.paramsModel['tol']
    n = self.paramsModel['number_products']
    
    y = []
    
    for i in range(self.paramsArchitecture['heterogeneity_x']):
        for j in range(self.paramsArchitecture['heterogeneity_z']):
            X = inputs[:n]
            Z = inputs[n:]              

        
            for k in range(self.paramsArchitecture['depth_eps_x']):
                X = [self.dense_x[i][k](X[a]) for a in range(n)]
                
            z = tf.keras.layers.Concatenate()(Z)

            for k in range(self.paramsArchitecture['depth_eps_z']):
                z = self.dense_z[j][k](z)
            
            z_u = tf.keras.layers.Concatenate()(Z)
            
            X_u = inputs[:n]            
            U = [tf.keras.layers.Concatenate()([X_u[a],X[a],z,z_u]) for a in range(n)]
            
            for k in range(self.paramsArchitecture['depth_u']):
                U = [self.utility[k](u) for u in U]

            U = [self.last(u) for u in U]

            combined = tf.keras.layers.Concatenate()(U) 
            
            combined = (tol+tf.keras.layers.Activation(activation=tf.nn.softmax)(combined))/(1+ tol*n)
            y.append(combined)
    if (self.paramsArchitecture['heterogeneity_x']*self.paramsArchitecture['heterogeneity_z'] > 1):
    	return tf.keras.layers.Average()(y)
    else:
    	return y



class VanillaNN(tf.keras.Model):

  def __init__(self, paramsArchitecture, paramsModel):
    super(VanillaNN, self).__init__()
    self.paramsArchitecture = paramsArchitecture
    self.paramsModel = paramsModel
    regularizer = tf.keras.regularizers.L2(self.paramsModel['regularization'])
    self.dense = [Dense(self.paramsArchitecture['width'], 
                            activation="elu",
                            kernel_regularizer=regularizer, 
                            use_bias= True) for i in range(self.paramsArchitecture['depth'])] 
    
    if self.paramsModel['embedding'] == True:
        self.embed = [Dense(1, activation="linear", use_bias= True) \
                      for i in range(self.paramsModel['number_c_features'])]
  

    self.last = Dense(self.paramsModel['number_products'], 
                    activation="linear",
                    kernel_regularizer=regularizer, 
                    use_bias= False)

  def call(self,inputs):
    tol = self.paramsModel['tol']
    n = self.paramsModel['number_products']  
    
    X = inputs[:n]
    Z = inputs[n:]      
    
    if self.paramsModel['embedding']==True:                
        Z = [self.embed[c](z) for (c,z) in enumerate(Z)]
    
    z = tf.keras.layers.Concatenate()(Z)
    x = tf.keras.layers.Concatenate()(X)
    
    combined = tf.keras.layers.Concatenate()([x,z])    
    for k in range(self.paramsArchitecture['depth']):
        combined = self.dense[k](combined)
    combined = self.last(combined)

    return (tol+tf.keras.layers.Activation(activation=tf.nn.softmax)(combined))/(1+ tol*n)




