#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 09:44:24 2023

@author: aaouad
"""


import tensorflow as tf
import pandas as pd
import numpy as np

def mse38(y_true, y_pred):
    depth = 38
    y_true = tf.one_hot(tf.cast(y_true,tf.int32), depth)
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(tf.reduce_sum(squared_difference, axis=1),axis = 0)

def mse39(y_true, y_pred):
    depth = 39
    y_true = tf.one_hot(tf.cast(y_true,tf.int32), depth)
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(tf.reduce_sum(squared_difference, axis=1),axis = 0)

def mse3(y_true, y_pred):
    depth = 3
    y_true = tf.one_hot(tf.cast(y_true,tf.int32), depth)
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(tf.reduce_sum(squared_difference, axis=1),axis = 0)


def data_expedia_woopt(perm = 1):
    '''
    Takes wide data (normalized) and returns list of product features, separate from customers'
    '''
    raw_data = np.load("./data/expedia/data_without_outside_opt/X_matrix.npy")
    raw_target = np.load("./data/expedia/data_without_outside_opt/Y_matrix.npy")

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
    number_features_per_product = len(p_features)
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


    #if modelName == 'RandomForest':
    #    X = np.concatenate(X,axis=1)

    return X,Y,number_products,number_c_features,number_features_per_product


def data_expedia_opt(perm = 1):
    '''
    Takes wide data (normalized) and returns list of product features, separate from customers'
    '''
    raw_data = np.load("./data/expedia/data_with_outside_opt/X_matrix.npy")
    raw_target = np.load("./data/expedia/data_with_outside_opt/Y_matrix.npy")

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
     'srch_destination_id_23904',
     'is_no_purchase']
    p_features = [
                    'prop_starrating',
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
                     'prop_country_id_220',
                     'is_no_purchase'
                     ]
    c_features = [
                     'srch_length_of_stay',
                     'srch_adults_count',
                     'srch_children_count',
                     'srch_room_count',
                     'srch_saturday_night_bool',
                     'random_bool',
                     'day_of_week',
                     'month',
                     'hour',
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
    number_features_per_product = len(p_features)
    number_products = 39
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


    #if modelName == 'RandomForest':
    #    X = np.concatenate(X,axis=1)

    return X,Y,number_products,number_c_features,number_features_per_product
