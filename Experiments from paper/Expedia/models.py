#!/usr/bin/env python3
# -*- coding: utf-8 -*-




import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Layer
from tensorflow.keras import Model, Input
# from class_mymodel import *
# from class_taste_net import *
from functools import reduce
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


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
                          use_bias= True)
    
  def call(self,inputs):
    
    tol = self.paramsModel['tol']
    n = self.paramsModel['number_products']
    
    U = inputs[:n]
    Z = inputs[n:]   
    
    # u1 = inputs[0]
    # u2 = inputs[1]
    # u3 = inputs[2]
    # z_group = inputs[3]
    # z_purpose = inputs[4]
    # z_first = inputs[5]
    # z_ticket = inputs[6]
    # z_who = inputs[7]
    # z_luggage = inputs[8]
    # z_age = inputs[9]
    # z_male = inputs[10]
    # z_income = inputs[11]
    # z_ga = inputs[12]
    # z_origin = inputs[13]
    # z_dest = inputs[14]     
    # print(len(Z),type(Z),type(U),len(inputs))
    
    U = [tf.keras.layers.Concatenate()((U[i],)+ Z) for i in range(n)]
    
    if self.paramsArchitecture['depth']>0:
        
        for k in range(self.paramsArchitecture['depth']):
            U = [self.dense[k](U[i]) for i in range(n)]            
                  
            # u1 = tf.keras.layers.Concatenate()([u1,z_group,z_purpose,z_first,z_ticket,z_who,z_luggage,z_age,z_male,z_income,z_ga,z_origin,z_dest])
            # u2 = tf.keras.layers.Concatenate()([u2,z_group,z_purpose,z_first,z_ticket,z_who,z_luggage,z_age,z_male,z_income,z_ga,z_origin,z_dest])
            # u3 = tf.keras.layers.Concatenate()([u3,z_group,z_purpose,z_first,z_ticket,z_who,z_luggage,z_age,z_male,z_income,z_ga,z_origin,z_dest])
            # for k in range(self.paramsArchitecture['depth']):
            #     u1 = self.dense[k](u1)
            #     u2 = self.dense[k](u2)
            #     u3 = self.dense[k](u3)
    
    U = [ self.last(U[i]) for i in range(n)]                
    
    combined = tf.keras.layers.Concatenate()(U)
    
    return (tol+tf.keras.layers.Activation(activation=tf.nn.softmax)(combined))\
            /(1+ tol*n)


class RUMModel(tf.keras.Model):

  def __init__(self, paramsArchitecture, paramsModel):
    super(RUMModel, self).__init__()
    self.paramsArchitecture = paramsArchitecture
    self.paramsModel = paramsModel

    regularizer = tf.keras.regularizers.L2(self.paramsModel['regularization'])
    #initializer = tf.keras.initializers.TruncatedNormal(mean=0.0001, stddev=0.0002)
    self.dense_x = [[Dense(self.paramsArchitecture['width_eps_x'], 
                            activation="elu",
                            kernel_regularizer=regularizer, 
                            use_bias= False) \
                     for i in range(self.paramsArchitecture['depth_eps_x'])] \
                    for j in range(self.paramsArchitecture['heterogeneity_x'])]
    self.last_x = [Dense(self.paramsArchitecture['last_x'],
                            activation="linear",
                            #kernel_regularizer=regularizer, 
                            use_bias= False) for j in range(self.paramsArchitecture['heterogeneity_x'])]
    self.dense_z = [[Dense(self.paramsArchitecture['width_eps_z'], 
                            activation="elu",
                            kernel_regularizer=regularizer, 
                            use_bias= False) for i in range(self.paramsArchitecture['depth_eps_z'])] for j in range(self.paramsArchitecture['heterogeneity_z'])]
    self.last_z = [Dense(self.paramsArchitecture['last_z'], 
                        activation="linear",
                        #kernel_regularizer=regularizer, 
                        use_bias= False) for j in range(self.paramsArchitecture['heterogeneity_z'])]
    self.utility = [Dense(self.paramsArchitecture['width_u'], 
                        activation="elu", 
                        kernel_regularizer=regularizer, 
                        use_bias= False) for i in range(self.paramsArchitecture['depth_u'])]
    self.last = Dense(1, 
                    activation="linear",
                    #kernel_regularizer=regularizer, 
                    use_bias= False)

    if self.paramsModel['embedding'] == True:
        self.embed = [Dense(1, activation="linear", use_bias= True) \
                      for i in range(self.paramsModel['number_c_features'])]
        # self.embed_group = Dense(1, activation="linear", use_bias= True)
        # self.embed_purpose = Dense(1, activation="linear", use_bias= True)
        # self.embed_first = Dense(1, activation="linear", use_bias= True)
        # self.embed_ticket = Dense(1, activation="linear", use_bias= True)
        # self.embed_who = Dense(1, activation="linear", use_bias= True)
        # self.embed_luggage = Dense(1, activation="linear", use_bias= True)
        # self.embed_age = Dense(1, activation="linear", use_bias= True)
        # self.embed_male = Dense(1, activation="linear", use_bias= True)
        # self.embed_income = Dense(1, activation="linear", use_bias= True)
        # self.embed_ga = Dense(1, activation="linear", use_bias= True)
        # self.embed_origin = Dense(1, activation="linear", use_bias= True)
        # self.embed_dest = Dense(1, activation="linear", use_bias= True)

  def call(self,inputs):
    tol = self.paramsModel['tol']
    n = self.paramsModel['number_products']
    
       
    
    y = []
    
    for i in range(self.paramsArchitecture['heterogeneity_x']):
        for j in range(self.paramsArchitecture['heterogeneity_z']):
            X = inputs[:n]
            Z = inputs[n:]              
            
            # x1 = inputs[0]
            # x2 = inputs[1]
            # x3 = inputs[2]
            # z_group = inputs[3]
            # z_purpose = inputs[4]
            # z_first = inputs[5]
            # z_ticket = inputs[6]
            # z_who = inputs[7]
            # z_luggage = inputs[8]
            # z_age = inputs[9]
            # z_male = inputs[10]
            # z_income = inputs[11]
            # z_ga = inputs[12]
            # z_origin = inputs[13]
            # z_dest = inputs[14] 
        
            for k in range(self.paramsArchitecture['depth_eps_x']):
                X = [self.dense_x[i][k](X[a]) for a in range(n)]
                
                # x1 = self.dense_x[i][k](x1)
                # x2 = self.dense_x[i][k](x2)
                # x3 = self.dense_x[i][k](x3)
            
            X = [self.last_x[k](x) for x in X]
            
            # x1 = self.last_x[i](x1)
            # x2 = self.last_x[i](x2)
            # x3 = self.last_x[i](x3)
            
            if self.paramsModel['embedding']==True:                
                Z = [self.embed[c](z) for (c,z) in enumerate(Z)]
                
                # z_group = self.embed_group(z_group)
                # z_purpose = self.embed_purpose(z_purpose)
                # z_first = self.embed_first(z_first)
                # z_ticket = self.embed_ticket(z_ticket)
                # z_who = self.embed_who(z_who)
                # z_luggage = self.embed_luggage(z_luggage)
                # z_age = self.embed_age(z_age)
                # z_male = self.embed_male(z_male)
                # z_income = self.embed_income(z_income)
                # z_ga = self.embed_ga(z_ga)
                # z_origin = self.embed_origin(z_origin)
                # z_dest = self.embed_dest(z_dest)
                
            z = tf.keras.layers.Concatenate()(Z)

            # z = tf.keras.layers.Concatenate()([z_group,z_purpose,z_first,z_ticket,z_who,z_luggage,z_age,z_male,z_income,z_ga,z_origin,z_dest])
            
            for k in range(self.paramsArchitecture['depth_eps_z']):
                z = self.dense_z[j][k](z)
            z = self.last_z[j](z)
            
            z_u = tf.keras.layers.Concatenate()(Z)
            
            # z_u = tf.keras.layers.Concatenate()([z_group,z_purpose,z_first,z_ticket,z_who,z_luggage,z_age,z_male,z_income,z_ga,z_origin,z_dest])

            X_u = inputs[:n]            
            U = [tf.keras.layers.Concatenate()([X_u[a],X[a],z,z_u]) for a in range(n)]
            
            # u1 = tf.keras.layers.Concatenate()([inputs[0],x1,z,z_u])
            # u2 = tf.keras.layers.Concatenate()([inputs[1],x2,z,z_u])
            # u3 = tf.keras.layers.Concatenate()([inputs[2],x3,z,z_u])
            
            for k in range(self.paramsArchitecture['depth_u']):
                U = [self.utility[k](u) for u in U]

                # u1 = self.utility[k](u1)
                # u2 = self.utility[k](u2)
                # u3 = self.utility[k](u3)

            U = [self.last(u) for u in U]

            # u1 = self.last(u1)
            # u2 = self.last(u2)
            # u3 = self.last(u3)
            
            combined = tf.keras.layers.Concatenate()(U) 
            
            # combined = tf.keras.layers.Concatenate()([u1,u2,u3]) 
            
            combined = (tol+tf.keras.layers.Activation(activation=tf.nn.softmax)(combined))/(1+ tol*n)
            y.append(combined)
    
    return tf.keras.layers.Average()(y)

class VanillaNN(tf.keras.Model):

  def __init__(self, paramsArchitecture, paramsModel):
    super(VanillaNN, self).__init__()
    self.paramsArchitecture = paramsArchitecture    
    self.paramsModel = paramsModel
    regularizer = tf.keras.regularizers.L2(self.paramsModel['regularization'])
    self.dense = [Dense(self.paramsArchitecture['width'], 
                            activation="elu",
                            kernel_regularizer=regularizer, 
                            use_bias= False) for i in range(self.paramsArchitecture['depth'])] 
    
    if self.paramsModel['embedding'] == True:
        self.embed = [Dense(1, activation="linear", use_bias= True) \
                      for i in range(self.paramsModel['number_c_features'])]
    # self.embed_group = Dense(2, activation="elu", use_bias= True)
    # self.embed_purpose = Dense(2, activation="elu", use_bias= True)
    # self.embed_first = Dense(2, activation="elu", use_bias= True)
    # self.embed_ticket = Dense(2, activation="elu", use_bias= True)
    # self.embed_who = Dense(2, activation="elu", use_bias= True)
    # self.embed_luggage = Dense(2, activation="elu", use_bias= True)
    # self.embed_age = Dense(2, activation="elu", use_bias= True)
    # self.embed_male = Dense(2, activation="elu", use_bias= True)
    # self.embed_income = Dense(2, activation="elu", use_bias= True)
    # self.embed_ga = Dense(2, activation="elu", use_bias= True)
    # self.embed_origin = Dense(2, activation="elu", use_bias= True)
    # self.embed_dest = Dense(2, activation="elu", use_bias= True)

    self.last = Dense(self.paramsModel['number_products'], 
                    activation="linear",
                    kernel_regularizer=regularizer, 
                    use_bias= False)

  def call(self,inputs):
    tol = self.paramsModel['tol']
    n = self.paramsModel['number_products']  
    
    X = inputs[:n]
    Z = inputs[n:]      
    # x1 = inputs[0]
    # x2 = inputs[1]
    # x3 = inputs[2]
    
    
    if self.paramsModel['embedding']==True:                
        Z = [self.embed[c](z) for (c,z) in enumerate(Z)]
        
    # z_group = inputs[3]
    # z_purpose = inputs[4]
    # z_first = inputs[5]
    # z_ticket = inputs[6]
    # z_who = inputs[7]
    # z_luggage = inputs[8]
    # z_age = inputs[9]
    # z_male = inputs[10]
    # z_income = inputs[11]
    # z_ga = inputs[12]
    # z_origin = inputs[13]
    # z_dest = inputs[14] 

    # z_group = self.embed_group(z_group)
    # z_purpose = self.embed_purpose(z_purpose)
    # z_first = self.embed_first(z_first)
    # z_ticket = self.embed_ticket(z_ticket)
    # z_who = self.embed_who(z_who)
    # z_luggage = self.embed_luggage(z_luggage)
    # z_age = self.embed_age(z_age)
    # z_male = self.embed_male(z_male)
    # z_income = self.embed_income(z_income)
    # z_ga = self.embed_ga(z_ga)
    # z_origin = self.embed_origin(z_origin)
    # z_dest = self.embed_dest(z_dest)

    z = tf.keras.layers.Concatenate()(Z)
    # z = tf.keras.layers.Concatenate()([z_group,z_purpose,z_first,z_ticket,z_who,z_luggage,z_age,z_male,z_income,z_ga,z_origin,z_dest])

    combined = tf.keras.layers.Concatenate()(X+(z,))    
    # combined = tf.keras.layers.Concatenate()([x1,x2,x3,z])
    for k in range(self.paramsArchitecture['depth']):
        combined = self.dense[k](combined)
    combined = self.last(combined)

    return (tol+tf.keras.layers.Activation(activation=tf.nn.softmax)(combined))/(1+ tol*n)




