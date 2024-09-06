# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""
import os
import warnings
import tensorflow as tf
import numpy as np

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import time
from BF.inference_utils import InputReader

from BF.network.my_model import Model
from BF.network.pre_trained_embedding import Settings

def run_BF(preparation_config):

    #==================================Model===================================
    start_time = time.time()
    print ("Run OPUS-BF...")
    
    start_time = time.time()
    test_reader = InputReader(data_list=preparation_config["filenames"], 
                              preparation_config=preparation_config)

    #============================Parameters====================================
    params = {}
    params["n_1d_feat"] = 256
    params["n_2d_feat"] = 128
    params["max_relative_distance"] = 32
    params["evofomer_config"] = Settings.CONFIG
    
    #============================Models====================================
    model_bf = Model(model_params=params)
    model_bf(feat_esm=np.zeros((1,10,1280)), 
             feat_1d=np.zeros((1,10,43)), 
             feat_2d=np.zeros((1,10,10, 130)), 
             residue_index=np.array([range(10)]), 
             L=10,
             preparation_config=preparation_config)    
    
    if preparation_config["type"] == "seq":
        model_bf.load_model(name="./models/opus_bf_seq.h5")
    elif preparation_config["type"] == "struct":
        model_bf.load_model(name="./models/opus_bf_struct.h5")

    for step, filenames_batch in enumerate(test_reader.dataset):

        filenames, x, x_trr, x_esm, L = \
            test_reader.read_file_from_disk(filenames_batch)
        
        residue_index = np.array([range(L)])
        
        if L > 512:
            n = L // 512 + 1
            bf_predictions = []
            for i in range(n):
                residue_index_ = residue_index[:,i*512:(i+1)*512]
                L_ = residue_index_.shape[1]
                bf_prediction = model_bf(x[:,i*512:(i+1)*512,:], x_trr[:,i*512:(i+1)*512,i*512:(i+1)*512,:], x_esm[:,i*512:(i+1)*512,:], 
                                         residue_index_, L_, preparation_config, training=False)  
                bf_predictions.append(bf_prediction.numpy())
            bf_predictions = np.concatenate(bf_predictions, 0)
        else:
            bf_predictions = model_bf(x, x_trr, x_esm,
                                      residue_index, L, preparation_config, training=False)   
            bf_predictions = bf_predictions.numpy()
        bf_predictions = np.array(bf_predictions)
        
        assert bf_predictions.shape[0] == L
        np.save(os.path.join(preparation_config["output_path"],
                             filenames[0]+".bfs"), bf_predictions)
        
    run_time = time.time() - start_time
    print('OPUS-BF done..., time: %3.3f' % (run_time)) 
    #==================================Model===================================
    
    
    