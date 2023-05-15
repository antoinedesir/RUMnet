#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf

import numpy as np
import csv
import random

from src.synthetic.data_preparation import create_data
from src.experiment_auxiliary import cross_validate, cross_validate_synthetic_ground_truth

tf.config.threading.set_inter_op_parallelism_threads(16)
tf.config.threading.set_intra_op_parallelism_threads(16) 
 
def set_seed(seed):
    '''
    Set random seed to have same training data for all runs. Since Keras and scikit learn use different pseudo random generators, need to seed a bunch of things
    '''
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)

if __name__ == '__main__':
        
    ##### Parameters common to all experiments
    with open("src/synthetic/config_general.json") as json_data_file:
        paramsGeneral = json.load(json_data_file)
    set_seed(paramsGeneral['seed'])

    ##### Parameters of models tested
    with open("src/synthetic/config_modelsArchitecture.json") as json_data_file:
        paramsArchitecture = json.load(json_data_file)
        
    numberModels = len(paramsArchitecture["models"])

    ##### Parameters specific to this experiment
    with open("src/synthetic/config_experiment_3.json") as json_data_file:
        paramsExperiment = json.load(json_data_file)
    
    ##### Creating folder to output results
    dirName = 'output/Synthetic/'+paramsExperiment['testName']
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    
    ##### Create dataset
    X, Y, coef_true = create_data(paramsExperiment)

    ##### Write parameters to file
    with open('output/Synthetic/'+paramsExperiment['testName']+'/ground_truth_param.csv','a') as fd:
        writer = csv.writer(fd)
        writer.writerow(['coefs_ground_truth', coef_true])

    ##### Compute metrics under ground truth model
    metrics = cross_validate_synthetic_ground_truth(X,Y,paramsGeneral,coef_true, paramsExperiment)
    
    ##### Compute metrics for each model 
    for i in range(numberModels):
        print('------', paramsArchitecture["models"][i]["name"],'------')
        cross_validate(X, Y, paramsGeneral, paramsArchitecture["models"][i], paramsExperiment)
        
            


    









