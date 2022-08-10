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
                                                 criterion = 'gini', random_state = 42)
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
                        valbest = valLoss
                        data_best = [
                           {"Fold":"Fold "+str(i), 
                            "n_estimators": paramsArchitecture['n_estimators'][j], "max_depth": paramsArchitecture['max_depth'][k],
                            "train_loss": trainLoss, "test_loss": testLoss,"val_loss": valLoss,  
                            "train_acc": trainAcc, "test_acc": testAcc, "val_acc": valAcc,
                            "time_fit": end-start
                             }]

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
                elif modelName == 'NN':
                    model = VanillaNN(paramsArchitecture,paramsModel)
                elif modelName == 'TasteNet':
                    model = TasteNet(paramsArchitecture,paramsModel)                    
                
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
                optimizer = tf.keras.optimizers.Adam(learning_rate=paramsModel['learningRate'][j])
                model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy",mse38])
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



def scheduler(epoch, lr):
    if epoch < 1000:
        return 0.001
    elif epoch < 2000:
        return 0.0001
    else:
        return 0.00001


def data_prepE(modelName,perm = 1):
    '''
    Takes wide data (normalized) and returns list of product features, separate from customers'
    '''
    raw_data = np.load("X_matrix_vAli.npy")
    raw_target = np.load("Y_matrix_vAli.npy")
    
    print("RAW DATA | The number of observations is {:,.0f}.".format(raw_data.shape[0]))
    print("RAW DATA | The number of columns per observations is {:,.0f}.".format(raw_data.shape[1]))

    features_list = [ 'prop_starrating',
                     'prop_review_score',
                     'prop_brand_bool',
                     'prop_location_score1',
                     'prop_location_score2',
                     'prop_log_historical_price',
                     'position',
                     'promotion_flag',
                     'srch_length_of_stay',
                     'srch_adults_count',
                     'srch_children_count',
                     'srch_room_count',
                     'srch_saturday_night_bool',
                     'orig_destination_distance',
                     'random_bool',
                     'day_of_week',
                     'month',
                     'hour',
                     'log_price',
                     'booking_window',                  
                     'site_id_-1',
                     'site_id_5',
                     'site_id_7',
                     'site_id_9',
                     'site_id_11',
                     'site_id_12',
                     'site_id_14',
                     'site_id_15',
                     'site_id_16',
                     'site_id_18',
                     'site_id_19',
                     'site_id_22',
                     'site_id_24',
                     'site_id_29',
                     'site_id_31',
                     'site_id_32',
                     'visitor_location_country_id_-1',
                     'visitor_location_country_id_31',
                     'visitor_location_country_id_39',
                     'visitor_location_country_id_50',
                     'visitor_location_country_id_55',
                     'visitor_location_country_id_59',
                     'visitor_location_country_id_92',
                     'visitor_location_country_id_99',
                     'visitor_location_country_id_100',
                     'visitor_location_country_id_103',
                     'visitor_location_country_id_117',
                     'visitor_location_country_id_129',
                     'visitor_location_country_id_132',
                     'visitor_location_country_id_158',
                     'visitor_location_country_id_216',
                     'visitor_location_country_id_219',
                     'visitor_location_country_id_220',
                     'prop_country_id_-1',
                     'prop_country_id_4',
                     'prop_country_id_15',
                     'prop_country_id_31',
                     'prop_country_id_32',
                     'prop_country_id_39',
                     'prop_country_id_55',
                     'prop_country_id_59',
                     'prop_country_id_73',
                     'prop_country_id_81',
                     'prop_country_id_92',
                     'prop_country_id_98',
                     'prop_country_id_99',
                     'prop_country_id_100',
                     'prop_country_id_103',
                     'prop_country_id_117',
                     'prop_country_id_129',
                     'prop_country_id_132',
                     'prop_country_id_158',
                     'prop_country_id_202',
                     'prop_country_id_205',
                     'prop_country_id_215',
                     'prop_country_id_216',
                     'prop_country_id_219',
                     'prop_country_id_220',
                     'srch_destination_id_-1',
                     'srch_destination_id_3073',
                     'srch_destination_id_4562',
                     'srch_destination_id_8192',
                     'srch_destination_id_8347',
                     'srch_destination_id_9402',
                     'srch_destination_id_10979',
                     'srch_destination_id_13216',
                     'srch_destination_id_13292',
                     'srch_destination_id_13870',
                     'srch_destination_id_15307',
                     'srch_destination_id_18774',
                     'srch_destination_id_23904']
    p_features = ['prop_starrating',
                     'prop_review_score',
                     'prop_brand_bool',
                     'prop_location_score1',
                     'prop_location_score2',
                     'prop_log_historical_price',
                     'position',
                     'promotion_flag',
                     'orig_destination_distance',                     
                     'log_price',   
                     'prop_country_id_-1',
                     'prop_country_id_4',
                     'prop_country_id_15',
                     'prop_country_id_31',
                     'prop_country_id_32',
                     'prop_country_id_39',
                     'prop_country_id_55',
                     'prop_country_id_59',
                     'prop_country_id_73',
                     'prop_country_id_81',
                     'prop_country_id_92',
                     'prop_country_id_98',
                     'prop_country_id_99',
                     'prop_country_id_100',
                     'prop_country_id_103',
                     'prop_country_id_117',
                     'prop_country_id_129',
                     'prop_country_id_132',
                     'prop_country_id_158',
                     'prop_country_id_202',
                     'prop_country_id_205',
                     'prop_country_id_215',
                     'prop_country_id_216',
                     'prop_country_id_219',
                     'prop_country_id_220'                     
                     ]
    c_features = [
                     'srch_length_of_stay',
                     'srch_adults_count',
                     'srch_children_count',
                     'srch_room_count',
                     'srch_saturday_night_bool', 
                     'booking_window',                     
                     'random_bool',
                     'day_of_week',
                     'month',
                     'hour',                     
                    'site_id_-1',
                     'site_id_5',
                     'site_id_7',
                     'site_id_9',
                     'site_id_11',
                     'site_id_12',
                     'site_id_14',
                     'site_id_15',
                     'site_id_16',
                     'site_id_18',
                     'site_id_19',
                     'site_id_22',
                     'site_id_24',
                     'site_id_29',
                     'site_id_31',
                     'site_id_32',
                     'visitor_location_country_id_-1',
                     'visitor_location_country_id_31',
                     'visitor_location_country_id_39',
                     'visitor_location_country_id_50',
                     'visitor_location_country_id_55',
                     'visitor_location_country_id_59',
                     'visitor_location_country_id_92',
                     'visitor_location_country_id_99',
                     'visitor_location_country_id_100',
                     'visitor_location_country_id_103',
                     'visitor_location_country_id_117',
                     'visitor_location_country_id_129',
                     'visitor_location_country_id_132',
                     'visitor_location_country_id_158',
                     'visitor_location_country_id_216',
                     'visitor_location_country_id_219',
                     'visitor_location_country_id_220',
                     'srch_destination_id_-1',
                     'srch_destination_id_3073',
                     'srch_destination_id_4562',
                     'srch_destination_id_8192',
                     'srch_destination_id_8347',
                     'srch_destination_id_9402',
                     'srch_destination_id_10979',
                     'srch_destination_id_13216',
                     'srch_destination_id_13292',
                     'srch_destination_id_13870',
                     'srch_destination_id_15307',
                     'srch_destination_id_18774',
                     'srch_destination_id_23904'                                          
        ]
    number_p_features = len(p_features) 
    number_products = 38
    number_c_features = len(c_features)
    
    if perm >1:
        perms = [np.random.permutation(number_products)  for q in range(perm)]
    else:
        perms = [range(number_products)]
        
    raw_data = np.concatenate([raw_data[:,:,pi] for pi in perms],axis = 0)
    raw_target = np.concatenate([raw_target[:,pi] for pi in perms],axis = 0)    
    ## Normalization
    
    for p in range(raw_data.shape[1]):
        if(np.max(np.abs(raw_data[:,p,:])) != 1):
            raw_data[:,p,:] = raw_data[:,p,:]/np.mean(raw_data[:,p,:])
    
    ## creation of list of product features per alternative
    p_features_r = [features_list.index(k) for k in p_features]
    X = [raw_data[:,p_features_r,i] for i in range(number_products)]
    
    ## creation of list of customer features 
    c_features_r = [features_list.index(k) for k in c_features]
    X += [raw_data[:,c,0].reshape((-1,1)) for c in c_features_r]    
    
    ## target TODO check indexing
    Y = np.argmax(raw_target,axis = 1)
    

    if modelName == 'RandomForest':
        X = np.concatenate(X,axis=1)

    return X,Y,number_products,number_c_features,number_p_features 
    

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
    long_data["one"] = 1
    long_data["zero"] = 0

    X1 = long_data[["TRAIN_AV", "TRAIN_TT", "TRAIN_CO", "TRAIN_HE","one","zero","zero"]].values.astype(float)
    X2 = long_data[["SM_AV", "SM_TT", "SM_CO", "SM_HE","zero","one","zero"]].values.astype(float)
    X3 = long_data[["CAR_AV", "CAR_TT","CAR_CO", "CAR_HE","zero","zero","one"]].values.astype(float)
    
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

    return X,Y


if __name__ == '__main__':
        
    
    ## Choose model
    modelName = "RandomForest"
    testName = 'Expedia/'+modelName+'vcheck' 

    ## Creating model
    if modelName=="MNL":
        paramsArchitecture = {
            'depth': 10,
            'width': 30,
        }
    elif modelName=="RUMnet":
        paramsArchitecture = {
            'depth_u':5,
            'width_u':20,
            'depth_eps_x':5,
            'width_eps_x':20,
            'last_x':20,
            'depth_eps_z':5,
            'width_eps_z':20,
            'last_z':20,
            'heterogeneity_x': 10,
            'heterogeneity_z': 10
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
            'n_estimators' : [300,500], 
            'max_depth' : [5,10,20]
        }
    
    ## Additional model parameters
    paramsModel = {
        'earlyStopping' : True,
        'numberEpochs' : 50,
        'regularization' : 0.,
        'learningRate' : [0.001],
        'batchSize' : 32,
        'embedding' : False,
        'tol' : 1e-4,
        'early_stopping_patience':10
    }

    ## Creating folder to output results
    dirName = 'output/'+testName
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    
    ## Parameters of the cross validation
    paramsCross = {
        'seed':1234,
        'numberFold': 10,
        'fractionTest': 0.1,
        'fractionVal': 0.9*0.1
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
    X,Y,number_products,number_c_features,number_p_features = data_prepE(modelName,1)
    paramsModel["number_products"] = number_products
    paramsModel["number_c_features"] = number_c_features
    paramsModel["number_p_features"] = number_p_features
    print(Y.shape)
    
    cross_validate(X, Y, modelName, paramsCross, paramsArchitecture, paramsModel, testName)
    



    

    









