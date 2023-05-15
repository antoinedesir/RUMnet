#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 09:44:24 2023

@author: aaouad
"""

import pandas as pd 
import numpy as np



def data_swissmetro():
    '''
    Load and preprocess the data
    '''

    ## Loading the data"
    raw_data = pd.read_csv('./data/Swissmetro/swissmetro.dat',sep='\t')
    raw_data["CAR_HE"] = 0
    
    c_features = ["GROUP", "PURPOSE", "FIRST", "TICKET", "WHO", "LUGGAGE", "AGE", "MALE", "INCOME", "GA", "ORIGIN", "DEST"]
    
    p_features = ["TRAIN_AV", "SM_AV", "CAR_AV", "TRAIN_TT", "SM_TT", "CAR_TT", "TRAIN_CO", "SM_CO", "CAR_CO",
              "TRAIN_HE", "SM_HE", "CAR_HE"]
    
    target = "CHOICE"
    
    # number_features_per_product = 4
    # num_p_features = len(p_features)
    print("RAW DATA | The number of observations is {:,.0f}.".format(raw_data.shape[0]))
    print("RAW DATA | The number of columns per observations is {:,.0f}.".format(raw_data.shape[1]))

    ### dropping no choice
    raw_data = raw_data[raw_data[target] > 0]
    raw_data.loc[:,target] = raw_data.loc[:,target]-1

    ### dropping unknown age
    #raw_data = raw_data[raw_data["AGE"] != 6]
    #raw_data = raw_data[raw_data["PURPOSE"] != 9]
    
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
    number_features_per_product = 7
    
    # X1 = long_data[["TRAIN_AV", "TRAIN_TT", "TRAIN_CO", "TRAIN_HE"]].values.astype(float)
    # X2 = long_data[["SM_AV", "SM_TT", "SM_CO", "SM_HE"]].values.astype(float)
    # X3 = long_data[["CAR_AV", "CAR_TT","CAR_CO", "CAR_HE"]].values.astype(float)
    # number_features_per_product = 4
    
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
    
    return X,Y,3,12,number_features_per_product


