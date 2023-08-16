#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
import numpy as np
import csv
import random


from src.expedia.data_preparation_expedia import data_expedia_opt
from src.swiss_metro.data_preparation_swissmetro import data_swissmetro
from src.experiment_auxiliary import cross_validate
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

    #### Choose data set in {swiss_metro, expedia}
    testname = "swiss_metro"
    #### Choose model in {MNL, Tastenet, DeepMNL, mixDeepMNL, RUMnet, NN, RF}
    modelFamily = "RUMnet"
    
    ##### Parameters common to all experiments
    with open("src/"+testname+"/config_general.json") as json_data_file:
        paramsGeneral = json.load(json_data_file)
    set_seed(paramsGeneral['seed'])

    ##### Parameters of models tested
    with open("src/"+testname+"/modelsArchitecture/"+modelFamily+"/config_modelsArchitecture.json") as json_data_file:
        paramsArchitecture = json.load(json_data_file)
        
    numberModels = len(paramsArchitecture["models"])

    ##### Parameters specific to this experiment
    with open("src/"+testname+"/config_experiment.json") as json_data_file:
        paramsExperiment = json.load(json_data_file)
    
    ##### Creating folder to output results
    dirName = "output/"+paramsExperiment["testGroup"]+"/"+paramsExperiment["testName"]
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    
    
    ##### Create dataset
    
    #X,Y,number_products,number_c_features,number_p_features = data_expedia_opt()
    X,Y,number_products,number_c_features,number_p_features = data_swissmetro()
    if modelFamily == "RF":
        X = np.concatenate(X, axis=1)
    
    paramsExperiment["assortmentSettings"]["number_products"] = number_products
    paramsExperiment["assortmentSettings"]["assortment_size"] = number_products
    paramsExperiment["assortmentSettings"]["number_samples"] = Y.size
    paramsExperiment["universe_number_products"] = number_products    
    paramsExperiment["number_c_features"] = number_c_features
    paramsExperiment["number_p_features"] = number_p_features
    
    ##### Compute metrics for each model 
    for i in range(numberModels):
        print('------', paramsArchitecture["models"][i]["name"],'------')
        cross_validate(X, Y, paramsGeneral, paramsArchitecture["models"][i], paramsExperiment)
    








