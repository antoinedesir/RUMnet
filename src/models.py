#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Dense

def create_model(paramsArchitecture, paramsGeneral, paramsExperiment):
    if paramsArchitecture['modelName'] == 'MNL':
        model = MNLModel(paramsArchitecture['param'],paramsGeneral, paramsExperiment)
    elif paramsArchitecture['modelName'] == 'mixDeepMNL':
        model = mixDeepMNL(paramsArchitecture['param'],paramsGeneral, paramsExperiment)
    elif paramsArchitecture['modelName'] == 'RUMnet':
        model = RUMModel(paramsArchitecture['param'],paramsGeneral, paramsExperiment)
    elif paramsArchitecture['modelName'] == 'TasteNet':
        model = TasteNet(paramsArchitecture['param'],paramsGeneral, paramsExperiment)
    elif paramsArchitecture['modelName'] == 'NN':
        model = VanillaNN(paramsArchitecture['param'],paramsGeneral, paramsExperiment)
    return model


class MNLModel(tf.keras.Model):

  def __init__(self,paramsArchitecture, paramsGeneral, paramsExperiment):
    super(MNLModel, self).__init__()
    self.paramsArchitecture = paramsArchitecture
    self.paramsGeneral = paramsGeneral
    self.paramsExperiment = paramsExperiment
    
    regularizer = tf.keras.regularizers.L2(self.paramsGeneral['training']['regularization'])
    
    self.dense = [Dense(paramsArchitecture['width'], activation="elu", 
                          kernel_regularizer=regularizer,
                          use_bias= True) for i in range(self.paramsArchitecture['depth'])]
    
    self.last = Dense(1, activation="linear",
                          kernel_regularizer=regularizer,
                          use_bias= False)
    
  def call(self,inputs):
    
    tol = self.paramsGeneral['training']['tol']
    n = self.paramsExperiment['assortmentSettings']['assortment_size']
    
    U = inputs[:n]
    Z = inputs[n:]
    
    U = [tf.keras.layers.Concatenate()((U[i],)+ Z) for i in range(n)]

    if self.paramsArchitecture['depth']>0:
        
        for k in range(self.paramsArchitecture['depth']):
            U = [self.dense[k](U[i]) for i in range(n)]            
    
    U = [self.last(U[i]) for i in range(n)]                
    
    combined = tf.keras.layers.Concatenate()(U)
    
    return (1-tol)*tf.keras.layers.Activation(activation=tf.nn.softmax)(combined) + tol*tf.ones_like(combined)/n 


class TasteNet(tf.keras.Model):

  def __init__(self,paramsArchitecture, paramsGeneral, paramsExperiment):
    super(TasteNet, self).__init__()
    self.paramsArchitecture = paramsArchitecture
    self.paramsGeneral = paramsGeneral
    self.paramsExperiment = paramsExperiment
    
    regularizer = tf.keras.regularizers.L2(self.paramsGeneral['training']['regularization'])
    
    
    self.dense = [Dense(paramsArchitecture['width'], activation="elu", 
                          kernel_regularizer=regularizer,
                          use_bias= True) for i in range(self.paramsArchitecture['depth'])]
    
    self.last = Dense(paramsExperiment["number_p_features"], activation="linear",
                          kernel_regularizer=regularizer,
                          use_bias= True)
    
    self.beta = Dense(1, activation="linear",
                          kernel_regularizer=regularizer,
                          use_bias= True)    
    
  def call(self,inputs):
    
    tol = self.paramsGeneral['training']['tol']
    n = self.paramsExperiment['assortmentSettings']['assortment_size']
    
    U = inputs[:n]
    Z = tf.keras.layers.Concatenate()(inputs[n:])   
    
    if self.paramsArchitecture['depth']>0:
        
        for k in range(self.paramsArchitecture['depth']):
            Z = self.dense[k](Z) 

    U = [ tf.keras.layers.Dot(axes=1)([self.last(Z),U[i]])\
         + self.beta(U[i]) for i in range(n)]                
    
    combined = tf.keras.layers.Concatenate()(U) 
    
    return (1-tol)*tf.keras.layers.Activation(activation=tf.nn.softmax)(combined) + tol*tf.ones_like(combined)/n 


class mixDeepMNL(tf.keras.Model):

  def __init__(self, paramsArchitecture, paramsGeneral, paramsExperiment):
    super(mixDeepMNL, self).__init__()
    
    self.paramsArchitecture = paramsArchitecture
    self.paramsGeneral = paramsGeneral
    self.paramsExperiment = paramsExperiment

    regularizer = tf.keras.regularizers.L2(self.paramsGeneral['training']['regularization'])
   
    self.dense = [[Dense(self.paramsArchitecture['width'], 
                            activation="elu",
                            kernel_regularizer=regularizer, 
                            use_bias= True) \
                     for i in range(self.paramsArchitecture['depth'])] \
                    for j in range(self.paramsArchitecture['number_mixture'])]
     
    self.last = [Dense(1, 
                    activation="linear",
                    use_bias= False) for j in range(self.paramsArchitecture['number_mixture'])]

  def call(self,inputs):
    
    tol = self.paramsGeneral['training']['tol']
    n = self.paramsExperiment['assortmentSettings']['number_products']
    
    y = []
    
    X = inputs[:n]
    Z = inputs[n:]

    for j in range(self.paramsArchitecture['number_mixture']):
            
      U =  [tf.keras.layers.Concatenate()((X[i],)+ Z) for i in range(n)]             
        
      if self.paramsArchitecture['depth']>0:
            
        for k in range(self.paramsArchitecture['depth']):
          U = [self.dense[j][k](U[i]) for i in range(n)]
        U = [self.last[j](U[i]) for i in range(n)]             
        combined = tf.keras.layers.Concatenate()(U)
        combined = (1-tol)*tf.keras.layers.Activation(activation=tf.nn.softmax)(combined) + tol*tf.ones_like(combined)/n
        y.append(combined)

    if (self.paramsArchitecture['number_mixture'] > 1):
        return tf.keras.layers.Average()(y)
    else:
        return y


class RUMModel(tf.keras.Model):

  def __init__(self, paramsArchitecture, paramsGeneral, paramsExperiment):
    super(RUMModel, self).__init__()
    self.paramsArchitecture = paramsArchitecture
    self.paramsGeneral = paramsGeneral
    self.paramsExperiment = paramsExperiment

    regularizer = tf.keras.regularizers.L2(self.paramsGeneral['training']['regularization'])
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
    tol = self.paramsGeneral['training']['tol']
    n = self.paramsExperiment['assortmentSettings']['assortment_size']
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
        combined = (1-tol)*tf.keras.layers.Activation(activation=tf.nn.softmax)(combined) + tol*tf.ones_like(combined)/n
        y.append(combined)
    if (self.paramsArchitecture['heterogeneity_x']*self.paramsArchitecture['heterogeneity_z'] > 1):
      return tf.keras.layers.Average()(y)
    else:
      return y


class RUMModel_xonly(tf.keras.Model):

  def __init__(self, paramsArchitecture, paramsGeneral, paramsExperiment):
    super(RUMModel_xonly, self).__init__()
    self.paramsArchitecture = paramsArchitecture
    self.paramsGeneral = paramsGeneral
    self.paramsExperiment = paramsExperiment

    regularizer = tf.keras.regularizers.L2(self.paramsGeneral['training']['regularization'])
    
    self.dense_x = [[Dense(self.paramsArchitecture['width_eps_x'], 
                            activation="elu",
                            kernel_regularizer=regularizer, 
                            use_bias= True) \
                     for i in range(self.paramsArchitecture['depth_eps_x'])] \
                    for j in range(self.paramsArchitecture['heterogeneity_x'])]
    self.last_x = [Dense(self.paramsArchitecture['last_x'],
                            activation="linear",
                            use_bias= True) for j in range(self.paramsArchitecture['heterogeneity_x'])]
    
    self.utility = [Dense(self.paramsArchitecture['width_u'], 
                        activation="elu", 
                        kernel_regularizer=regularizer, 
                        use_bias= True) for i in range(self.paramsArchitecture['depth_u'])]
    self.last = Dense(1, 
                    activation="linear",
                    use_bias= False)

  def call(self,inputs):
    tol = self.paramsGeneral['training']['tol']
    n = self.paramsExperiment['assortmentSettings']['assortment_size']
    
       
    
    y = []
    
    for i in range(self.paramsArchitecture['heterogeneity_x']):
        X = inputs[:n]
        for k in range(self.paramsArchitecture['depth_eps_x']):
            X = [self.dense_x[i][k](X[a]) for a in range(n)]
        X = [self.last_x[i](X[a]) for a in range(n)]  
        X_u = inputs[:n]            
        U = [tf.keras.layers.Concatenate()([X_u[a],X[a]]) for a in range(n)]
             
        for k in range(self.paramsArchitecture['depth_u']):
            U = [self.utility[k](U[a]) for a in range(n)]

        U = [self.last(U[a]) for a in range(n)]
        combined = tf.keras.layers.Concatenate()(U) 
            
        combined = (1-tol)*tf.keras.layers.Activation(activation=tf.nn.softmax)(combined) + tol*tf.ones_like(combined)/n
        y.append(combined)
    if (self.paramsArchitecture['heterogeneity_x'] > 1):
        return tf.keras.layers.Average()(y)
    else:
        return y


class VanillaNN(tf.keras.Model):

  def __init__(self, paramsArchitecture, paramsGeneral, paramsExperiment):
    super(VanillaNN, self).__init__()
    self.paramsArchitecture = paramsArchitecture
    self.paramsGeneral = paramsGeneral
    self.paramsExperiment = paramsExperiment
    regularizer = tf.keras.regularizers.L2(self.paramsGeneral['training']['regularization'])
    self.dense = [Dense(self.paramsArchitecture['width'], 
                            activation="elu",
                            kernel_regularizer=regularizer, 
                            use_bias= True) for i in range(self.paramsArchitecture['depth'])] 
    
    self.last = Dense(self.paramsExperiment["assortmentSettings"]['number_products'], 
                    activation="linear",
                    kernel_regularizer=regularizer, 
                    use_bias= False)

  def call(self,inputs):
    tol = self.paramsGeneral['training']['tol']
    n = self.paramsExperiment["assortmentSettings"]['assortment_size'] 
    
    X = inputs[:n]
    Z = inputs[n:]      
    
    z = tf.keras.layers.Concatenate()(Z)
    x = tf.keras.layers.Concatenate()(X)
    
    combined = tf.keras.layers.Concatenate()([x,z])    
    
    for k in range(self.paramsArchitecture['depth']):
        combined = self.dense[k](combined)
    combined = self.last(combined)

    return (1-tol)*tf.keras.layers.Activation(activation=tf.nn.softmax)(combined) + tol*tf.ones_like(combined)/n 


