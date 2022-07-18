#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import joblib
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
import csv
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.backend import count_params
    
from matplotlib import pyplot as plt
import time
import random
from models import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from sklearn.externals import joblib

tf.config.threading.set_inter_op_parallelism_threads(16)
tf.config.threading.set_intra_op_parallelism_threads(16)    

def mse38(y_true, y_pred):
    depth = 38   
    y_true = tf.one_hot(tf.cast(y_true,tf.int32), depth)
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(tf.reduce_sum(squared_difference, axis=1),axis = 0)

def mse3(y_true, y_pred):
    depth = 3   
    y_true = tf.one_hot(tf.cast(y_true,tf.int32), depth)
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(tf.reduce_sum(squared_difference, axis=1),axis = 0)

def cross_validate(X, Y, modelName, paramsCross, paramsArchitecture, paramsModel, testName):

    sizeObservation = Y.size
    sizeTest = int(paramsCross['fractionTest']*sizeObservation)
    sizeVal = int(paramsCross['fractionVal']*sizeObservation) 
    sizeTrain = sizeObservation - sizeTest - sizeVal
    
    tol = paramsModel['tol']
    data_rows = []
    
    for i in range(paramsCross['numberFold']):
        print("Starting cross fold number {}.".format(i+1))
        data_cross_val = []
        if modelName == 'RandomForest':
            Xval = X.take(range(i*sizeVal,(i+1)*sizeVal),mode='wrap',axis=0)
            Xtrain = X.take(range((i+1)*sizeVal,(i+1)*sizeVal+sizeTrain),mode='wrap',axis=0) 
            Xtest = X.take(range((i+1)*sizeVal+sizeTrain,(i+1)*sizeVal+sizeTrain+sizeTest),mode='wrap',axis=0)
        else:
            Xval = [X[j].take(range(i*sizeVal,(i+1)*sizeVal),mode='wrap',axis=0) for j in range(len(X))]
            Xtrain = [X[j].take(range((i+1)*sizeVal,(i+1)*sizeVal+sizeTrain),mode='wrap',axis=0) for j in range(len(X))]
            Xtest = [X[j].take(range((i+1)*sizeVal+sizeTrain,(i+1)*sizeVal+sizeTrain+sizeTest),mode='wrap',axis=0) for j in range(len(X))]
                    
        Yval = Y.take(range(i*sizeVal,(i+1)*sizeVal),mode='wrap',axis=0)
        Ytrain = Y.take(range((i+1)*sizeVal,(i+1)*sizeVal+sizeTrain),mode='wrap',axis=0)
        Ytest = Y.take(range((i+1)*sizeVal+sizeTrain,(i+1)*sizeVal+sizeTrain+sizeTest),mode='wrap',axis=0)
        
        if modelName == 'RandomForest':
            valBest = 10
            for j in range(len(paramsArchitecture['n_estimators'])):
                for k in range(len(paramsArchitecture['max_depth'])):
                    model = RandomForestClassifier(n_estimators = paramsArchitecture['n_estimators'][j],
                                                 max_depth=paramsArchitecture['max_depth'][k], verbose = 2,
                                                 criterion = 'entropy', random_state = 42)
                    Xtrain = np.array(Xtrain)
                    Xval = np.array(Xval)
                    Xtest = np.array(Xtest)
                    Yval = np.array(Yval)
                    Ytrain = np.array(Ytrain)
                    Ytest = np.array(Ytest)
                    
                    start = time.time()
                    model.fit(Xtrain,Ytrain)
                    end = time.time()
                    trainAcc = accuracy_score(Ytrain, model.predict(Xtrain))
                    valAcc = accuracy_score(Yval, model.predict(Xval))
                    testAcc = accuracy_score(Ytest, model.predict(Xtest))

                    y_pred = (model.predict_proba(Xtrain)+tol)/(1+ tol*paramsModel["number_products"])    
                    trainLoss = -np.mean(np.log([y_pred[q,Ytrain[q]] for q in range(len(Ytrain))]))
                    y_pred = (model.predict_proba(Xval)+tol)/(1+ tol*paramsModel["number_products"])    
                    valLoss = -np.mean(np.log([y_pred[q,Yval[q]] for q in range(len(Yval))]))
                    y_pred = (model.predict_proba(Xtest)+tol)/(1+ tol*paramsModel["number_products"])    
                    testLoss = -np.mean(np.log([y_pred[q,Ytest[q]] for q in range(len(Ytest))]))
                    
                    data_cross_val += [
                           {"Fold":"Fold "+str(i), 
                            "n_estimators": paramsArchitecture['n_estimators'][j], "max_depth": paramsArchitecture['max_depth'][k],
                            "train_loss": trainLoss, "test_loss": testLoss,"val_loss": valLoss,  
                            "train_acc": trainAcc, "test_acc": testAcc, "val_acc": valAcc,
                            "time_fit": end-start
                             }]
                    if valLoss < valBest:
                        valBest = valLoss
                        data_best = [
                           {"Fold":"Fold "+str(i), 
                            "n_estimators": paramsArchitecture['n_estimators'][j], "max_depth": paramsArchitecture['max_depth'][k],
                            "train_loss": trainLoss, "test_loss": testLoss,"val_loss": valLoss,  
                            "train_acc": trainAcc, "test_acc": testAcc, "val_acc": valAcc,
                            "time_fit": end-start
                             }]
                        joblib.dump(model,'output/'+testName+'/random_forest_'+str(i)+'.joblib')

            data_cross_val = pd.DataFrame(data_cross_val)
            data_cross_val.to_csv('output/'+testName+'/results_'+str(i)+'.csv', index = True)
            data_rows += data_best


        else:
            valBest = 10                
            for j in range(len(paramsModel['learningRate'])):
                if modelName == 'MNL':
                    model = MNLModel(paramsArchitecture, paramsModel)
                elif modelName == 'RUMnet':
                    model = RUMModel(paramsArchitecture, paramsModel)
                elif modelName == 'TasteNet':
                    model = TasteNet(paramsArchitecture, paramsModel)
                elif modelName == 'NN':
                    model = VanillaNN(paramsArchitecture, paramsModel)
                
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
                optimizer = tf.keras.optimizers.Adam(learning_rate=paramsModel['learningRate'][j])
                model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy",mse3])
                es = EarlyStopping(monitor = 'val_loss',
                                    patience = paramsModel['early_stopping_patience'],
                                    verbose = 0,
                                    restore_best_weights =True)
                callbacks = []
                if paramsModel["earlyStopping"]:
                    callbacks += [es]                
                start = time.time()                                     
                history = model.fit(Xtrain,Ytrain,
                    validation_data= (Xval,Yval),
                    batch_size=paramsModel['batchSize'],
                    epochs=paramsModel['numberEpochs'],
                    callbacks = callbacks,
                    verbose=2,
                    shuffle=True)
                model.save('output/'+testName+'/model_'+str(i),save_format='tf')
                end = time.time()


                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'val'], loc='upper left')
                plt.savefig('output/'+testName+'/loss_'+str(i)+str(j)+'.png')
                plt.clf()
        

                trainLoss , trainAcc, trainMse = model.evaluate(Xtrain,Ytrain, verbose=0) 
                testLoss , testAcc, testMse = model.evaluate(Xtest,Ytest, verbose=0) 
                valLoss , valAcc, valMse = model.evaluate(Xval,Yval, verbose=0) 
            
                #print("train loss: {}, val_loss: {}".format(trainLoss,valLoss))
                data_cross_val += [
                        {"Fold":"Fold "+str(i), 
                            "learning_rate": paramsModel['learningRate'][j],
                            "train_loss": trainLoss, "test_loss": testLoss,"val_loss": valLoss,  
                            "train_acc": trainAcc, "test_acc": testAcc, "val_acc": valAcc,
                            "time_fit": end-start, 'number_epochs': es.stopped_epoch
                             }]
                if valLoss < valBest:
                    valBest = valLoss
                    data_best = [
                        {"Fold":"Fold "+str(i), 
                            "learning_rate": paramsModel['learningRate'][j],
                            "train_loss": trainLoss, "test_loss": testLoss,"val_loss": valLoss,  
                            "train_acc": trainAcc, "test_acc": testAcc, "val_acc": valAcc,
                            "time_fit": end-start, 'number_epochs': es.stopped_epoch
                             }]
                hist_df = pd.DataFrame(history.history) 
                hist_csv_file = 'output/'+testName+'/history_'+str(i)+str(j)+'.csv'
                with open(hist_csv_file, mode='w') as f:
                    hist_df.to_csv(f)

                if (i==0):    
                    with open('output/'+testName+'/main.csv','a') as fd:
                        writer = csv.writer(fd)
                        # count = count_params(model.trainable_weights)
                        count = np.sum([count_params(p) for p in model.trainable_weights])
                        writer.writerow(['numberParameters', count])
    
            data_rows += data_best

    data = pd.DataFrame(data_rows)
    data = data.set_index('Fold').transpose()
    temp_mean = data.mean(numeric_only=True, axis=1)
    temp_std = data.std(numeric_only=True, axis=1)
    data['mean'] = temp_mean
    data['std'] = temp_std
    data.to_csv('output/'+testName+'/results.csv', index = True)



def data_prep(modelName):
    '''
    Load and preprocess the data
    '''

    ## Loading the data
    raw_data = pd.read_csv('swissmetro.dat',sep='\t')
    raw_data["CAR_HE"] = 0
    
    c_features = ["GROUP", "PURPOSE", "FIRST", "TICKET", "WHO", "LUGGAGE", "AGE", "MALE", "INCOME", "GA", "ORIGIN", "DEST"]
    
    p_features = ["TRAIN_AV", "SM_AV", "CAR_AV", "TRAIN_TT", "SM_TT", "CAR_TT", "TRAIN_CO", "SM_CO", "CAR_CO",
              "TRAIN_HE", "SM_HE", "CAR_HE"]
    
    target = "CHOICE"
    
    number_features_per_product = 4
    num_p_features = len(p_features)
    print("RAW DATA | The number of observations is {:,.0f}.".format(raw_data.shape[0]))
    print("RAW DATA | The number of columns per observations is {:,.0f}.".format(raw_data.shape[1]))

    ### dropping no choice
    raw_data = raw_data[raw_data[target] > 0]
    raw_data.loc[:,target] = raw_data.loc[:,target]-1

    
    raw_data = raw_data.reset_index()
    raw_data = raw_data.sample(frac=1).reset_index(drop=True)
    print ("DROPPING NO CHOICE | The number of observations is {:,.0f}.".format(raw_data.shape[0]))

    ### Creating dummies
    long_data = pd.get_dummies(raw_data, columns=c_features, drop_first=False)
    
    features_list = long_data.columns.values.tolist()
    drop = ["index","ID","SURVEY","SP","SM_SEATS"]
    c_features = [feature for feature in features_list if feature not in p_features+[target]+drop]
    print(len(c_features), "customer features." , sep =" ")
    print(len(p_features), "product features." , sep =" ")

    ## Preparing input
    ## Normalization
    long_data[["TRAIN_TT", "SM_TT", "CAR_TT"]] /= 1000
    long_data[["TRAIN_CO", "SM_CO", "CAR_CO"]] /= 5000
    long_data[["TRAIN_HE", "SM_HE", "CAR_HE"]] /= 100
    
    
    X1 = long_data[["TRAIN_AV", "TRAIN_TT", "TRAIN_CO", "TRAIN_HE"]].values.astype(float)
    X2 = long_data[["SM_AV", "SM_TT", "SM_CO", "SM_HE"]].values.astype(float)
    X3 = long_data[["CAR_AV", "CAR_TT","CAR_CO", "CAR_HE"]].values.astype(float)
    
    Z = long_data[c_features].values.astype(float)

    col_group = [col for col in long_data if col.startswith('GROUP')]
    Z_group = long_data[col_group].values.astype(float)
    col_purpose = [col for col in long_data if col.startswith('PURPOSE')]
    Z_purpose = long_data[col_purpose].values.astype(float)
    col_first = [col for col in long_data if col.startswith('FIRST')]
    Z_first = long_data[col_first].values.astype(float)
    col_ticket = [col for col in long_data if col.startswith('TICKET')]
    Z_ticket = long_data[col_ticket].values.astype(float)
    col_who = [col for col in long_data if col.startswith('WHO')]
    Z_who = long_data[col_who].values.astype(float)
    col_luggage = [col for col in long_data if col.startswith('LUGGAGE')]
    Z_luggage = long_data[col_luggage].values.astype(float)
    col_age = [col for col in long_data if col.startswith('AGE')]
    Z_age = long_data[col_age].values.astype(float)
    col_male = [col for col in long_data if col.startswith('MALE')]
    Z_male = long_data[col_male].values.astype(float)
    col_income = [col for col in long_data if col.startswith('INCOME')]
    Z_income = long_data[col_income].values.astype(float)
    col_ga = [col for col in long_data if col.startswith('GA')]
    Z_ga = long_data[col_ga].values.astype(float)
    col_origin = [col for col in long_data if col.startswith('ORIGIN')]
    Z_origin = long_data[col_origin].values.astype(float)
    col_dest = [col for col in long_data if col.startswith('DEST')]
    Z_dest = long_data[col_dest].values.astype(float)
    
    Y = long_data[target].values

    X = [X1,X2,X3,Z_group,Z_purpose,Z_first,Z_ticket,Z_who,Z_luggage,Z_age,Z_male,Z_income,Z_ga,Z_origin,Z_dest]
    if modelName == 'RandomForest':
        X = np.concatenate(X,axis=1)
    if modelName == 'MNLcross':
        Z = np.concatenate([Z_group,Z_purpose,Z_first,Z_ticket,Z_who,Z_luggage,Z_age,Z_male,Z_income,Z_ga,Z_origin,Z_dest],axis=1)
        Z1 = np.concatenate([np.multiply(np.transpose([X1[:,0]]),Z) for i in range(number_features_per_product)],axis=1)
        Z2 = np.concatenate([np.multiply(np.transpose([X2[:,0]]),Z) for i in range(number_features_per_product)],axis=1)
        Z3 = np.concatenate([np.multiply(np.transpose([X3[:,0]]),Z) for i in range(number_features_per_product)],axis=1)
        X = [X1,X2,X3,Z1,Z2,Z3]

    return X,Y,3,12


if __name__ == '__main__':
        
    
    ## Choose model
    modelName = "NN"
    testName = 'NN_10d30w'   

    ## Creating model
    if modelName=="MNL":
        paramsArchitecture = {
            'depth': 2,
            'width': 5  ,
        }
    elif modelName=="MNLcross" :
         paramsArchitecture = {
            }       
    elif modelName=="RUMnet":
        paramsArchitecture = {
            'depth_u':2,
            'width_u':5,
            'depth_eps_x':2,
            'width_eps_x':5,
            'last_x':0, 
            'depth_eps_z':2,
            'width_eps_z':5,
            'last_z':0,
            'heterogeneity_x': 5,
            'heterogeneity_z': 5
            }
    elif modelName=="NN":
        paramsArchitecture = {
            'depth':10,
            'width':30
            }
    elif modelName=="TasteNet":
        paramsArchitecture = {
            'depth':3,
            'width':10
            }   
    elif modelName=="RandomForest":
        paramsArchitecture = {
            'n_estimators' : [200,300,400], 
            'max_depth' : [10,15,20]
        }
    
    ## Additional model parameters
    paramsModel = {
        'earlyStopping' : True,
        'numberEpochs' : 1000,
        'regularization' : 0.,
        'learningRate' : [0.001],
        'batchSize' : 32,
        'embedding' : False,
        'tol' : 1e-2,
        'early_stopping_patience': 300,
        'number_p_features':4
    }

    ## Creating folder to output results
    dirName = 'output/'+testName
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    
    ## Parameters of the cross validation
    paramsCross = {
        'seed':1234,
        'numberFold': 10,
        'fractionTest': 0.15,
        'fractionVal': 0.15
        }
    

    ## Set random seed to have same training data for all runs. Since Keras and scikit learn use different pseudo random generators, need to seed a bunch of things
    random.seed(paramsCross['seed'])
    np.random.seed(paramsCross['seed'])
    os.environ['PYTHONHASHSEED']=str(paramsCross['seed'])
    tf.random.set_seed(paramsCross['seed'])


    ## Write parameters to file
    with open('output/'+testName+'/main.csv','a') as fd:
        writer = csv.writer(fd)
        writer.writerow(['model',modelName])
        writer.writerow(['Cross fold parameters'])
        for key, value in paramsCross.items():
            writer.writerow([key, value])
        writer.writerow(['Model parameters'])
        for key, value in paramsModel.items():
            writer.writerow([key, value])
        writer.writerow(['Model architecture'])
        for key, value in paramsArchitecture.items():
            writer.writerow([key, value])
        
    ## Load data
    #X,Y,number_products,number_c_features = data_prepE(modelName,1)
    X,Y,number_products,number_c_features = data_prep(modelName)
    paramsModel["number_products"] = number_products
    paramsModel["number_c_features"] = number_c_features
    
    cross_validate(X, Y, modelName, paramsCross, paramsArchitecture, paramsModel, testName)
    




    









