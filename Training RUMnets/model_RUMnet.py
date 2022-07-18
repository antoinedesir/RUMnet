#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Layer
from tensorflow.keras import Model, Input
from functools import reduce

'''

'''

class RUMnet(tf.keras.Model):
  '''
  This function implements the RUMnet model in the tensor flow framework. 
  '''

  def __init__(self,paramsArchitecture, paramsModel):
    '''
    paramsArchitecture is a dictionnary containing parameters allowing fine tuning the architecture
        paramsArchitecture['regularization']: the coefficient for L2 regularization 
        paramsArchitecture['width_u']: number of neurons in each layer of the feed-forward neural network representing the utility
        paramsArchitecture['depth_u']: number of layers in the feed-forward neural network representing the utility
        paramsArchitecture['width_eps_x']: number of neurons in each layer of the feed-forward neural network representing epsilon
        paramsArchitecture['depth_eps_x']: number of layers in the feed-forward neural network representing epsilon
        paramsArchitecture['width_eps_z']: number of neurons in each layer of the feed-forward neural network representing nu
        paramsArchitecture['depth_eps_z']: number of layers in the feed-forward neural network representing nu
        paramsArchitecture['heterogeneity_x']: number of samples for epsilon
        paramsArchitecture['heterogeneity_z']: number of samples for nu
            
    paramsModel is a dictionnary containing parameters allowing fine tuning the model
        paramsModel['tol']: small noise added to the choice probabilities
        paramsModel['number_products']: assortment size
    '''

    super(RUMnet, self).__init__()
    self.paramsArchitecture = paramsArchitecture
    self.paramsModel = paramsModel
    
    # Setting up a regularizer
    regularizer = tf.keras.regularizers.L2(self.paramsModel['regularization'])
    

    # Creating the feed-forward neural network for epsilon
    self.dense_x = [[Dense(paramsArchitecture['width_eps_x'], 
                            activation="elu", 
                            kernel_regularizer=regularizer,
                            use_bias= True) for i in range(self.paramsArchitecture['depth_eps_x'])]
                                for j in range(self.paramsArchitecture['heterogeneity_x'])]

    # Creating the feed-forward neural network for nu
    self.dense_z = [[Dense(self.paramsArchitecture['width_eps_z'], 
                            activation="elu",
                            kernel_regularizer=regularizer, 
                            use_bias= True) for i in range(self.paramsArchitecture['depth_eps_z'])] 
                                for j in range(self.paramsArchitecture['heterogeneity_z'])]
    
    # Creating the feed-forward neural network for the main utility 
    self.utility = [Dense(self.paramsArchitecture['width_u'], 
                        activation="elu", 
                        kernel_regularizer=regularizer, 
                        use_bias= True) for i in range(self.paramsArchitecture['depth_u'])]


    # Last linear layer to convert the output into a single number
    self.last = Dense(1, activation="linear",
                          kernel_regularizer=regularizer,
                          use_bias= False)
    
  def call(self,inputs):
    
    y = []
    
    for i in range(self.paramsArchitecture['heterogeneity_x']):
        for j in range(self.paramsArchitecture['heterogeneity_z']):
            # Product features
            X = inputs[:self.paramsModel['assortmentSize']]

            # Customer features
            Z = inputs[self.paramsModel['assortmentSize']:]

            # Computing epsilon
            for k in range(self.paramsArchitecture['depth_eps_x']):
                X = [self.dense_x[i][k](X[a]) for a in range(self.paramsModel['assortmentSize'])]
            
            # Computing nu
            z = tf.keras.layers.Concatenate()(Z)
            for k in range(self.paramsArchitecture['depth_eps_z']):
                z = self.dense_z[j][k](z)

            # Concatenating input for utility
            z_u = tf.keras.layers.Concatenate()(Z)
            X_u = inputs[:self.paramsModel['assortmentSize']]            
            U = [tf.keras.layers.Concatenate()([X_u[a],X[a],z,z_u]) for a in range(self.paramsModel['assortmentSize'])]
            
            # Computing utility
            for k in range(self.paramsArchitecture['depth_u']):
                U = [self.utility[k](u) for u in U]
            U = [self.last(u) for u in U]

            # Applying a Softmax layer
            combined = tf.keras.layers.Concatenate()(U)
            combined = tf.keras.layers.Activation(activation=tf.nn.softmax)(combined)
            combined = (self.paramsModel['tol']+combined)/(1+self.paramsModel['tol']*self.paramsModel['assortmentSize'])

            y.append(combined)

    if (self.paramsArchitecture['heterogeneity_x']*self.paramsArchitecture['heterogeneity_z'] > 1):
        return tf.keras.layers.Average()(y)
    else:
        return y




