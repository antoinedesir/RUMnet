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

class DeepMNL(tf.keras.Model):
  '''
  This function implements the Deep MNL model in the tensor flow framework. 
  '''

  def __init__(self,paramsArchitecture, paramsModel):
    '''
    paramsArchitecture is a dictionnary containing parameters allowing fine tuning the architecture
        paramsArchitecture['regularization']: the coefficient for L2 regularization 
        paramsArchitecture['width']: number of neurons in each layer of the feed-forward neural network
        paramsArchitecture['rdepth']: number of layers in the feed-forward neural network
        
    paramsModel is a dictionnary containing parameters allowing fine tuning the model
        paramsModel['tol']: small noise added to the choice probabilities
        paramsModel['number_products']: assortment size
    '''

    super(DeepMNL, self).__init__()
    self.paramsArchitecture = paramsArchitecture
    self.paramsModel = paramsModel
    
    # Setting up a regularizer
    regularizer = tf.keras.regularizers.L2(self.paramsModel['regularization'])
    
    # Creating the feed-forward neural network
    self.dense = [Dense(paramsArchitecture['width'], 
                            activation="elu", 
                            kernel_regularizer=regularizer,
                            use_bias= True) for i in range(self.paramsArchitecture['depth'])]
    
    # Last linear layer to convert the output into a single number
    self.last = Dense(1, activation="linear",
                          kernel_regularizer=regularizer,
                          use_bias= False)
    
  def call(self,inputs):
    
    # Product features
    U = inputs[:self.paramsModel['assortmentSize']]

    # Customer features
    Z = inputs[self.paramsModel['assortmentSize']:]
    
    # Appending customer features to each product
    U = [tf.keras.layers.Concatenate()((U[i],)+ Z) for i in range(self.paramsModel['assortmentSize'])]

    # Passing the features through the feed-forward neural network
    if self.paramsArchitecture['depth']>0:
        for k in range(self.paramsArchitecture['depth']):
            U = [self.dense[k](U[i]) for i in range(self.paramsModel['assortmentSize'])]            
    U = [self.last(U[i]) for i in range(self.paramsModel['assortmentSize'])]                
    
    # Applying a Softmax layer
    combined = tf.keras.layers.Concatenate()(U)
    combined = tf.keras.layers.Activation(activation=tf.nn.softmax)(combined)

    return (self.paramsModel['tol']+combined)/(1+self.paramsModel['tol']*self.paramsModel['assortmentSize'])





