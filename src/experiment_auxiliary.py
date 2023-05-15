#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 09:43:05 2023

@author: aaouad
"""

import os
import joblib
import tensorflow as tf

import pandas as pd
import numpy as np
import csv
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import count_params
    
from matplotlib import pyplot as plt
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from .synthetic.data_preparation import prediction, compute_loss
from .models import *


def split_data_X(X, splitNumber, paramsGeneral, paramsExperiment, RF_indicator):
    
    sizeObservation = paramsExperiment['assortmentSettings']['number_samples']
    sizeTest = int(paramsGeneral['crossVal']['fractionTest']*sizeObservation)
    sizeVal = int(paramsGeneral['crossVal']['fractionVal']*sizeObservation) 
    sizeTrain = sizeObservation - sizeTest - sizeVal
    
    if (RF_indicator == 1):
        Xval = X.take(range(splitNumber*sizeVal,(splitNumber+1)*sizeVal),mode='wrap',axis=0) 
        Xtrain = X.take(range((splitNumber+1)*sizeVal,(splitNumber+1)*sizeVal+sizeTrain),mode='wrap',axis=0) 
        Xtest = X.take(range((splitNumber+1)*sizeVal+sizeTrain,(splitNumber+1)*sizeVal+sizeTrain+sizeTest),mode='wrap',axis=0)
    else:
        Xval = [X[j].take(range(splitNumber*sizeVal,(splitNumber+1)*sizeVal),mode='wrap',axis=0) for j in range(len(X))]
        Xtrain = [X[j].take(range((splitNumber+1)*sizeVal,(splitNumber+1)*sizeVal+sizeTrain),mode='wrap',axis=0) for j in range(len(X))]
        Xtest = [X[j].take(range((splitNumber+1)*sizeVal+sizeTrain,(splitNumber+1)*sizeVal+sizeTrain+sizeTest),mode='wrap',axis=0) for j in range(len(X))]
    
    return Xval, Xtrain, Xtest

def split_data_Y(Y, splitNumber, paramsGeneral, paramsExperiment):
    
    sizeObservation = paramsExperiment['assortmentSettings']['number_samples']
    sizeTest = int(paramsGeneral['crossVal']['fractionTest']*sizeObservation)
    sizeVal = int(paramsGeneral['crossVal']['fractionVal']*sizeObservation) 
    sizeTrain = sizeObservation - sizeTest - sizeVal
    
    Yval = Y.take(range(splitNumber*sizeVal,(splitNumber+1)*sizeVal),mode='wrap',axis=0)
    Ytrain = Y.take(range((splitNumber+1)*sizeVal,(splitNumber+1)*sizeVal+sizeTrain),mode='wrap',axis=0)
    Ytest = Y.take(range((splitNumber+1)*sizeVal+sizeTrain,(splitNumber+1)*sizeVal+sizeTrain+sizeTest),mode='wrap',axis=0) 
    
    return Yval, Ytrain, Ytest

def cross_validate_synthetic_ground_truth(X, Y, paramsGeneral, coef_true, paramsExperiment):
    
    #tol = paramsGeneral['training']['tol']
    assortment_size = paramsExperiment['assortmentSettings']["number_products"]
    data_rows = []

    sizeObservation = paramsExperiment['assortmentSettings']['number_samples']
    sizeTest = int(paramsGeneral['crossVal']['fractionTest']*sizeObservation)
    sizeVal = int(paramsGeneral['crossVal']['fractionVal']*sizeObservation) 
    sizeTrain = sizeObservation - sizeTest - sizeVal
    
    assortment_size += 1
    for i in range(paramsGeneral['crossVal']['numberFold']):
        
        Xval, Xtrain, Xtest = split_data_X(X,i, paramsGeneral, paramsExperiment, 0)
        Yval, Ytrain, Ytest = split_data_Y(Y,i,paramsGeneral, paramsExperiment)

        #### Compute error on training set
        y_pred_train, proba_pred_train = prediction(Xtrain, sizeTrain, coef_true, paramsExperiment)
        true_trainAcc = accuracy_score(Ytrain, y_pred_train)
        true_trainLoss = compute_loss(proba_pred_train, Ytrain,paramsGeneral["training"]["tol"],paramsExperiment["assortmentSettings"]["number_products"])

        #### Compute error on validation set
        y_pred_val, proba_pred_val = prediction(Xval, sizeVal, coef_true, paramsExperiment)
        true_valAcc = accuracy_score(Yval, y_pred_val)
        true_valLoss = compute_loss(proba_pred_val, Yval,paramsGeneral["training"]["tol"],paramsExperiment["assortmentSettings"]["number_products"])

        #### Compute error on test set
        y_pred_test, proba_pred_test = prediction(Xtest, sizeTest, coef_true, paramsExperiment)
        true_testAcc = accuracy_score(Ytest, y_pred_test)
        true_testLoss = compute_loss(proba_pred_test, Ytest, paramsGeneral["training"]["tol"],paramsExperiment["assortmentSettings"]["assortment_size"])

        data_rows += [{
            'Fold': i,  
            'trainLoss': true_trainLoss,
            'trainAcc': true_trainAcc,
            'testLoss': true_testLoss,
            'testAcc': true_testAcc,
            'valLoss': true_valLoss,
            'valAcc': true_valAcc
        }]

    data = pd.DataFrame(data_rows)
    data = data.set_index('Fold').transpose()
    temp_mean = data.mean(numeric_only=True, axis=1)
    temp_std = data.std(numeric_only=True, axis=1)
    data['mean'] = temp_mean
    data['std'] = temp_std
    data.to_csv('output/'+paramsExperiment['testGroup']+'/'+paramsExperiment['testName']+'/'+'ground_truth_results.csv', index = True)   

def plot_history(history, iterNumber, paramsArchitecture, paramsExperiment):
    '''
    Plot history of training
    '''
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('output/'+paramsExperiment['testGroup']+'/'+paramsExperiment['testName']+'/'+paramsArchitecture['name']+'/loss_'+str(iterNumber)+'.png')
    plt.clf()
        

def model_fit(Xtrain, Xval, Ytrain, Yval, iterNumber, paramsGeneral, paramsArchitecture, paramsExperiment):
    '''
    Single run for cross validation
    '''
    
    ##### Create model
    model = create_model(paramsArchitecture, paramsGeneral, paramsExperiment)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=paramsGeneral['training']['learningRate'])
    model.compile(optimizer=optimizer, 
                  loss=loss, 
                  metrics = ["accuracy"]
                  #metrics=["accuracy",
                  #         lambda x,y:mse(x,y,paramsExperiment['assortmentSettings']["assortment_size"])
                  #          ]
                 )
    es = EarlyStopping(monitor = 'val_loss',
                       patience = paramsGeneral['training']['early_stopping_patience'],
                       verbose = 0,
                       restore_best_weights =True)
    callbacks = []
    if paramsGeneral['training']["earlyStopping"]:
        callbacks += [es]                
    
    ###### Fitting model
    start = time.time()                                     
    history = model.fit(Xtrain,Ytrain,
                    validation_data= (Xval,Yval),
                    batch_size=paramsGeneral['training']['batchSize'],
                    epochs=paramsGeneral['training']['numberEpochs'],
                    callbacks = callbacks,
                    verbose= 0,#2,
                    shuffle=True)
    model.save('output/'+paramsExperiment['testGroup']+'/'+paramsExperiment['testName']+'/'+paramsArchitecture['name']+'/model_'+str(iterNumber),save_format='tf')
    end = time.time()

    ###### Plot history
    plot_history(history, iterNumber, paramsArchitecture, paramsExperiment)

    ###### Save history
    hist_df = pd.DataFrame(history.history) 
    hist_csv_file = 'output/'+paramsExperiment['testGroup']+'/'+paramsExperiment['testName']+'/'+paramsArchitecture['name']+'/history_'+str(iterNumber)+'.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


    ###### Count number of parameters
    if (iterNumber==0):    
        with open('output/'+paramsExperiment['testGroup']+'/'+paramsExperiment['testName']+'/'+paramsArchitecture['name']+'/number_parameters.csv','a') as fd:
            writer = csv.writer(fd)
            count = np.sum([count_params(p) for p in model.trainable_weights])
            writer.writerow(['numberParameters', count])
    
    return model, end-start, es.stopped_epoch

def model_fit_tree(Xtrain, Xval, Ytrain, Yval, iterNumber, paramsGeneral, paramsArchitecture, paramsExperiment):

    ##### Create model
    model = RandomForestClassifier(n_estimators = paramsArchitecture['param']['n_estimators'],
                                                 max_depth=paramsArchitecture['param']['max_depth'], verbose = 2,
                                                 criterion = 'entropy', random_state = 42)
    
    ##### Fitting model
    start = time.time()
    model.fit(Xtrain,Ytrain)
    end = time.time()
    dirName = "output/"+paramsExperiment["testGroup"]+"/"+paramsExperiment["testName"]+'/'+paramsArchitecture['name']
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    joblib.dump(model,'output/'+paramsExperiment['testGroup']+'/'+paramsExperiment['testName']+'/'+paramsArchitecture['name']+'/model_'+str(iterNumber)+'.joblib')

    return model, end-start

def cross_validate(X, Y, paramsGeneral, paramsArchitecture, paramsExperiment):
    
    tol = paramsGeneral['training']['tol']
    assortment_size = paramsExperiment['assortmentSettings']["number_products"]
    data_rows = []
    
    assortment_size += 1
    for i in range(paramsGeneral['crossVal']['numberFold']):
        print("Starting cross fold number {}.".format(i+1))
        
        Xval, Xtrain, Xtest = split_data_X(X,i, paramsGeneral, paramsExperiment, paramsArchitecture['modelName']=='RandomForest')
        Yval, Ytrain, Ytest = split_data_Y(Y,i,paramsGeneral, paramsExperiment)

        data_cross_val = []
        
        if paramsArchitecture['modelName']=='RandomForest':

            Xtrain = np.array(Xtrain)
            Xval = np.array(Xval)
            Xtest = np.array(Xtest)
            Yval = np.array(Yval)
            Ytrain = np.array(Ytrain)
            Ytest = np.array(Ytest)

            #### model fit
            model, time_fit = model_fit_tree(Xtrain, Xval, Ytrain, Yval, i, paramsGeneral, paramsArchitecture, paramsExperiment)

            #### Evaluate model
            trainAcc = accuracy_score(Ytrain, model.predict(Xtrain))
            valAcc = accuracy_score(Yval, model.predict(Xval))
            testAcc = accuracy_score(Ytest, model.predict(Xtest))

            y_pred = (model.predict_proba(Xtrain)+tol)/(1+ tol*paramsExperiment['assortmentSettings']["number_products"])    
            trainLoss = -np.mean(np.log([y_pred[q,Ytrain[q]] for q in range(len(Ytrain))]))
            y_pred = (model.predict_proba(Xval)+tol)/(1+ tol*paramsExperiment['assortmentSettings']["number_products"])    
            valLoss = -np.mean(np.log([y_pred[q,Yval[q]] for q in range(len(Yval))]))
            y_pred = (model.predict_proba(Xtest)+tol)/(1+ tol*paramsExperiment['assortmentSettings']["number_products"])    
            testLoss = -np.mean(np.log([y_pred[q,Ytest[q]] for q in range(len(Ytest))]))

            stopped_epoch = 0

        else:
            
            #### model fit 
            model, time_fit, stopped_epoch = model_fit(Xtrain, Xval, Ytrain, Yval, i, paramsGeneral, paramsArchitecture, paramsExperiment)
            #### Evaluate model
            trainLoss , trainAcc = model.evaluate(Xtrain,Ytrain, verbose=0) 
            testLoss , testAcc = model.evaluate(Xtest,Ytest, verbose=0) 
            valLoss , valAcc = model.evaluate(Xval,Yval, verbose=0) 
        
            
        data_cross_val = [{
                    "Fold":"Fold "+str(i), 
                    "train_loss": trainLoss, "test_loss": testLoss,"val_loss": valLoss,  
                    "train_acc": trainAcc, "test_acc": testAcc, "val_acc": valAcc,
                    "time_fit": time_fit, 'number_epochs': stopped_epoch
                }]
            
        data_rows += data_cross_val


    data = pd.DataFrame(data_rows)
    data = data.set_index('Fold').transpose()
    temp_mean = data.mean(numeric_only=True, axis=1)
    temp_std = data.std(numeric_only=True, axis=1)
    data['mean'] = temp_mean
    data['std'] = temp_std
    data.to_csv('output/'+paramsExperiment['testGroup']+'/'+paramsExperiment['testName']+'/'+paramsArchitecture['name']+'/results.csv', index = True)              
        
