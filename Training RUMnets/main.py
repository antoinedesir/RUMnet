#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.backend import count_params
tf.config.threading.set_inter_op_parallelism_threads(16)

from model_DeepMNL import *
from model_RUMnet import *


def create_data(number_products, number_products_features, number_customer_features, number_samples):
    '''
    Input data X,Y has the following structure.
        X = [X1, X2, .. , Xn, Z] is a list where Xi is a matrix of size (number_samples,number_product_features) and Z is a matrix of size (number_samples, number_customer_features)
        Y is a vector of size number_sample 

    Here we use a simple MNL model to generate some training data.
    '''

    coef_MNL =(np.random.rand(number_products_features)-0.5)*2
    
    X = [] 
    for i in range(number_products):
        X.append(np.random.rand(number_samples,number_products_features))

    Y = np.zeros(number_samples)
    for i in range(number_samples):
        weights = [np.exp(np.dot(coef_MNL,X[j][i])) for j in range(number_products)]
        weights /= np.sum(weights) 
        Y[i] = np.random.choice(np.arange(number_products), p=weights)

    Z = np.random.rand(number_samples,number_customer_features)
    X.append(Z)
    
    return X,Y


if __name__ == '__main__':
    '''
    '''    
    
    ## Choose model
    modelName = 'DeepMNL'
    modelName = 'RUMnet'
    
    ## Architecture parameters
    if modelName=='DeepMNL':
        ## MNL is a special case of DeepMNL where depth and width are set to 0
        paramsArchitecture = {
            'depth': 2,
            'width': 5  ,
        }
    elif modelName=='RUMnet': 
        paramsArchitecture = {
            'depth_u':2,
            'width_u':5,
            'depth_eps_x':2,
            'width_eps_x':5,
            'last_x':5, 
            'depth_eps_z':2,
            'width_eps_z':5,
            'last_z':5,
            'heterogeneity_x': 5,
            'heterogeneity_z': 5
        }
    
    ## Model parameters
    paramsModel = {
        'earlyStopping' : True,
        'numberEpochs' : 500,
        'regularization' : 0.,
        'learningRate' : 0.001,
        'batchSize' : 32,
        'embedding' : False,
        'tol' : 1e-2,
        'early_stopping_patience': 20,
        'assortmentSize': 3
    }

    ## Create data
    number_samples = 500
    X, Y = create_data(number_products=3, 
                        number_products_features=3, 
                        number_customer_features=2, 
                        number_samples = number_samples) 
   
    ## Split train and val
    sizeTrain = int(0.8*number_samples)
    sizeVal = number_samples - sizeTrain
    
    Xval = [X[j][0:sizeVal,:] for j in range(len(X))]
    Xtrain = [X[j][sizeVal:sizeVal+sizeTrain,:] for j in range(len(X))]
    Yval = Y[0:sizeVal]
    Ytrain = Y[sizeVal:sizeVal+sizeTrain]

    ## Creating model
    if modelName == 'DeepMNL':
        model = DeepMNL(paramsArchitecture, paramsModel)
    elif modelName == 'RUMnet':
        model = RUMnet(paramsArchitecture, paramsModel)
                
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=paramsModel['learningRate'])
    model.compile(optimizer=optimizer, 
                    loss=loss, 
                    metrics="accuracy")

    es = EarlyStopping(monitor = 'val_loss',
                        patience = paramsModel['early_stopping_patience'],
                        verbose = 0,
                        restore_best_weights =True)
    callbacks = []
    if paramsModel["earlyStopping"]:
        callbacks += [es]                
                
    ## Fitting model
    history = model.fit(Xtrain,Ytrain,
                    validation_data= (Xval,Yval),
                    batch_size=paramsModel['batchSize'],
                    epochs=paramsModel['numberEpochs'],
                    callbacks = callbacks,
                    verbose=2,
                    shuffle=True)
                
    ## Computing metrics on train and validation data
    trainLoss , trainAcc = model.evaluate(Xtrain,Ytrain, verbose=0) 
    valLoss , valAcc = model.evaluate(Xval,Yval, verbose=0) 
    
    print("train loss: {}, val_loss: {}".format(trainLoss,valLoss))




    









