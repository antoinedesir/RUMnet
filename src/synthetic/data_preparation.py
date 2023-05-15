#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
from sklearn.metrics import accuracy_score
import tensorflow as tf

def create_data(paramsExperiment):
    
    if paramsExperiment['testName'] == 'mnl':
        return data_synthetic_MNL(paramsExperiment)

    if paramsExperiment['testName'] == 'nonlinear':
        return data_synthetic_nonlinear(paramsExperiment)

    if paramsExperiment['testName'] == 'unobserved':
        return data_synthetic_unobserved(paramsExperiment)


def mse(y_true, y_pred, depth):
    y_true = tf.one_hot(tf.cast(y_true,tf.int32), depth)
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(tf.reduce_sum(squared_difference, axis=1),axis = 0)


def data_synthetic_MNL(paramsExperiment):
    '''
    Create syntethic data from MNL model
    '''

    number_products = paramsExperiment['assortmentSettings']['number_products']
    assortment_size = paramsExperiment['assortmentSettings']['assortment_size']
    number_samples = paramsExperiment['assortmentSettings']['number_samples']
    number_p_features = paramsExperiment['assortmentSettings']['number_p_features']
    assort_repeat = paramsExperiment['assortmentSettings']['assort_repeat']
                                  

    coef_MNL =(np.random.rand(number_p_features+number_products)-0.5)*2
    

    offered_assortment = np.zeros((number_samples*assort_repeat,assortment_size))
    for i in range(number_samples):
        temp = random.sample(range(number_products), assortment_size) 
        for j in range(assort_repeat):
            offered_assortment[i*assort_repeat+j,:] = temp
    
    X = [] 
    for i in range(assortment_size):
        temp = np.zeros((number_samples*assort_repeat,number_p_features+number_products))
        temp[:,0:number_p_features] = np.random.rand(number_samples*assort_repeat,number_p_features)
        for j in range(number_samples*assort_repeat):
            temp[j,number_p_features+ int(offered_assortment[j,i])] = 1 
        X.append(temp)
    
    Y = np.zeros(number_samples*assort_repeat)
    for i in range(number_samples*assort_repeat):
        weights = [np.exp(np.dot(coef_MNL,X[j][i])) for j in range(assortment_size)]
        weights /= np.sum(weights) 
        Y[i] = np.random.choice(np.arange(assortment_size), p=weights)
    
    return X,Y, coef_MNL


def data_synthetic_nonlinear(paramsExperiment):
    '''
    Create syntethic data from MNL model
    '''

    number_products = paramsExperiment['assortmentSettings']['number_products']
    assortment_size = paramsExperiment['assortmentSettings']['assortment_size']
    number_samples = paramsExperiment['assortmentSettings']['number_samples']
    number_p_features = paramsExperiment['assortmentSettings']['number_p_features']
    assort_repeat = paramsExperiment['assortmentSettings']['assort_repeat']

    coef_MNL =(np.random.rand(number_p_features+number_products+3)-0.5)*2
    
    offered_assortment = np.zeros((number_samples*assort_repeat,assortment_size))
    for i in range(number_samples):
        temp = random.sample(range(number_products), assortment_size) 
        for j in range(assort_repeat):
            offered_assortment[i*assort_repeat+j,:] = temp
    
    X = [] 
    X_true = []
    for i in range(assortment_size):
        temp = np.zeros((number_samples*assort_repeat,number_p_features+number_products))
        temp_true = np.zeros((number_samples*assort_repeat,number_p_features+number_products+3))
        temp[:,0:number_p_features] = 10*np.random.rand(number_samples*assort_repeat,number_p_features) #1+ 10*np.random.rand(number_samples,number_p_features)
        temp_true[:,0:number_p_features] = temp[:,0:number_p_features]
        for j in range(number_samples*assort_repeat):
            temp[j,number_p_features+ int(offered_assortment[j,i])] = 1
            temp_true[j,number_p_features+ int(offered_assortment[j,i])]= 1

        X.append(temp)
        
        temp_true[:,number_p_features+number_products] = np.square(temp_true[:,0])
        temp_true[:,number_p_features+number_products+1] = np.square(temp_true[:,1])
        temp_true[:,number_p_features+number_products+2] = np.multiply(temp_true[:,1],temp_true[:,0])
        X_true.append(temp_true)
        

    
    Y = np.zeros(number_samples*assort_repeat)
    for i in range(number_samples*assort_repeat):
        weights = [np.exp(np.dot(coef_MNL,X_true[j][i])) for j in range(assortment_size)]
        weights /= np.sum(weights) 
        Y[i] = np.random.choice(np.arange(assortment_size), p=weights)
    
    return X,Y, coef_MNL


def data_synthetic_unobserved(paramsExperiment):
    '''
    Create syntethic data from MNL model with unobserved features
    '''

    number_products = paramsExperiment['assortmentSettings']['number_products']
    assortment_size = paramsExperiment['assortmentSettings']['assortment_size']
    number_samples = paramsExperiment['assortmentSettings']['number_samples']
    number_p_features = paramsExperiment['assortmentSettings']['number_p_features']
    assort_repeat = paramsExperiment['assortmentSettings']['assort_repeat']

    coef_MNL_1 = np.multiply(np.random.rand(number_p_features+number_products)-0.5,np.random.randint(1,100,number_p_features+number_products))
    coef_MNL_2 = np.multiply(np.random.rand(number_p_features+number_products)-0.5,np.random.randint(1,100,number_p_features+number_products))
    
    offered_assortment = np.zeros((number_samples*assort_repeat,assortment_size))
    for i in range(number_samples):
        temp = random.sample(range(number_products), assortment_size) 
        for j in range(assort_repeat):
            offered_assortment[i*assort_repeat+j,:] = temp


    X_true = [] 
    for i in range(assortment_size):
        temp = np.zeros((number_samples,number_p_features+number_products))
        temp[:,0:number_p_features] = np.random.rand(number_samples,number_p_features)
        for j in range(number_samples):
            temp[j,number_p_features+ int(offered_assortment[j,i])] = 1 
        X_true.append(temp)
        
    
    Y = np.zeros(number_samples)
    for i in range(number_samples):
        weights1 = [np.exp(np.dot(coef_MNL_1,X_true[j][i])) for j in range(assortment_size)]
        weights1 /= np.sum(weights1) 
        weights2 = [np.exp(np.dot(coef_MNL_2,X_true[j][i])) for j in range(assortment_size)]
        weights2 /= np.sum(weights2) 
        if (np.random.rand()<0.3):
            Y[i] = np.random.choice(np.arange(assortment_size), p=weights1)
        else:
            Y[i] = np.random.choice(np.arange(assortment_size), p=weights2)
        
    
    return X_true,Y, [coef_MNL_1, coef_MNL_2]


def prediction(X, sizeX, coef_true, paramsExperiment):
    '''
    '''

    if (paramsExperiment["testName"] == 'mnl'):
        y_pred_train = np.zeros(sizeX)
        proba_pred_train = np.zeros((sizeX, paramsExperiment['assortmentSettings']["assortment_size"]))
        for k in range(sizeX):
            weights = [np.exp(np.dot(coef_true,X[j][k])) for j in range(paramsExperiment['assortmentSettings']["assortment_size"])]
            weights /= np.sum(weights) 
            proba_pred_train[k,:] = weights
            y_pred_train[k] = np.argmax(weights)
    
    if (paramsExperiment["testName"] == 'nonlinear'):
        number_p_features = paramsExperiment["assortmentSettings"]["number_p_features"]
        number_products = paramsExperiment["assortmentSettings"]["number_products"]
        X_temp = [] 
        for k in range(paramsExperiment['assortmentSettings']["assortment_size"]):
            temp1 = np.zeros((sizeX,number_p_features+number_products+3))
            temp1[:,0:number_p_features+number_products] = X[k]
            temp1[:,number_p_features+number_products] = np.square(temp1[:,0])
            temp1[:,number_p_features+number_products+1] = np.square(temp1[:,1])
            temp1[:,number_p_features+number_products+2] = np.multiply(temp1[:,1],temp1[:,0])
            X_temp.append(temp1)
        
        y_pred_train = np.zeros(sizeX)
        proba_pred_train = np.zeros((sizeX,paramsExperiment['assortmentSettings']["assortment_size"]))
        for k in range(sizeX):
            weights = [np.exp(np.dot(coef_true,X_temp[j][k])) for j in range(paramsExperiment['assortmentSettings']["assortment_size"])]
            weights /= np.sum(weights) 
            proba_pred_train[k,:] = weights
            y_pred_train[k] = np.argmax(weights)
    
    if (paramsExperiment["testName"] == 'unobserved'):
        y_pred_train = np.zeros(sizeX)
        proba_pred_train = np.zeros((sizeX,paramsExperiment['assortmentSettings']["assortment_size"]))
        for k in range(sizeX):
            weights1 = [np.exp(np.dot(coef_true[0],X[j][k])) for j in range(paramsExperiment['assortmentSettings']["assortment_size"])]
            weights1 /= np.sum(weights1) 
            weights2 = [np.exp(np.dot(coef_true[1],X[j][k])) for j in range(paramsExperiment['assortmentSettings']["assortment_size"])]
            weights2 /= np.sum(weights2) 
            proba_pred_train[k,:] = 0.3*weights1 + 0.7*weights2
            y_pred_train[k] = np.argmax(0.3*weights1 + 0.7*weights2)

    return y_pred_train, proba_pred_train

def compute_loss(proba_pred,Y, tol, number_products):
    '''
    '''
    y_pred = (proba_pred+tol)/(1+ tol*number_products)    
    loss = -np.mean(np.log([y_pred[q,int(Y[q])] for q in range(len(Y))]))
    
    return loss