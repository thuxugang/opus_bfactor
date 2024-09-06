# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.stats import pearsonr

from BF.network.my_layer import TrackableLayer
from BF.network.my_rnn import MyBilstm

import BF.network.pre_trained_embedding.model.EmbeddingModel as Pre_MSA_emb
import BF.network.pre_trained_embedding.model.EvoFormer as EvoFormer

DTYPE = tf.float32
DTYPE_NP = np.float32

class AAEmbedding(TrackableLayer):
    def __init__(self, config):
        super(AAEmbedding, self).__init__()
        self.config = config
        self.emb = Pre_MSA_emb.Embedding(self.config)

    def call(self, inp_1d, residue_index):
        return self.emb(inp_1d, residue_index)

class AFEvoformer(TrackableLayer):
    def __init__(self, config, name_layer, name='evoformer_iteration', global_config=None):
        super(AFEvoformer, self).__init__(name=name_layer+"_"+str(global_config['iter']))
        self.config = config
        self.global_config = global_config
        self.evo_iteration = EvoFormer.Evoformer(config, name=name, global_config=global_config)
    def call(self, msa, pair, training=True):
        return self.evo_iteration(msa, pair, training=training)

class AFEvoformerEnsemble(TrackableLayer):
    def __init__(self, config, name_layer, iter_layer, name='evoformer_iteration', iters=None):
        super(AFEvoformerEnsemble, self).__init__(name=name_layer+"_"+str(iter_layer))
        self.config = config
        self.evo_iterations = []
        for i in range(len(iters)):
            global_config = {'iter': iters[i]}
            self.evo_iteration = EvoFormer.Evoformer(config, name=name, global_config=global_config)
            self.evo_iterations.append(self.evo_iteration)

    def call(self, msa, pair, training=True):
        for i in range(len(self.evo_iterations)):
            msa, pair = self.evo_iterations[i](msa, pair, training=training)
        return msa, pair

class ESMEmbedding(keras.layers.Layer):
    def __init__(self, n_feat):
        super(ESMEmbedding, self).__init__()
        self.esm_msa_proj = keras.layers.Dense(n_feat, name='esm_msa_linear')
        self.esm_msa_act_norm = keras.layers.LayerNormalization(name='esm_msa_act_norm')
        
    def call(self, feat):
        return self.esm_msa_act_norm(self.esm_msa_proj(feat))

class TRREmbedding(keras.layers.Layer):
    def __init__(self, name, n_feat):
        super(TRREmbedding, self).__init__()
        self.trr_emb = keras.layers.Conv2D(name=name, filters=n_feat, kernel_size=1, padding='SAME')
        self.norm = keras.layers.LayerNormalization()
        
    def call(self, feat):
        return self.norm(self.trr_emb(feat))

class Model(keras.Model):

    def __init__(self, model_params):
        super(Model, self).__init__()
        
        self.optimizer = None 
        
        self.model_params = model_params
        #=========================== Embedding ===========================
        self.pre_msa_emb = AAEmbedding(self.model_params)

        self.evoformers = []
        for i in range(6):
            self.evoformers.append(AFEvoformerEnsemble(self.model_params["evofomer_config"]["evoformer"],
                                                       name_layer='evoformer_ensemble',
                                                       iter_layer=i,
                                                       iters=[4*i, 4*i+1, 4*i+2, 4*i+3]))

        self.esm_msa_proj = ESMEmbedding(n_feat=256)
        self.trr_emb = TRREmbedding(name="trr_feat", n_feat=self.model_params["n_2d_feat"])
        #=========================== Output ===========================
        self.bilstm = MyBilstm(num_layers=4,
                               units=512,
                               rate=0.25,
                               output=1)
        
    def call(self, feat_1d, feat_2d, feat_esm, residue_index, L, preparation_config, training=False):
        
        assert feat_esm.shape == (1, L, 1280)
        assert feat_1d.shape == (1, L, 43)
        assert feat_2d.shape == (1, L, L, 130)

        if preparation_config["type"] == "seq":
            feat_1d = np.zeros((1, L, 43))
            feat_2d = np.zeros((1, L, L, 130))            
        elif preparation_config["type"] == "struct":
            feat_esm = np.zeros((1, L, 1280))
        
        f_esm = self.esm_msa_proj(feat_esm)
        assert f_esm.shape == (1, L, 256)

        f_2d_trr = self.trr_emb(feat_2d)[0]
        assert f_2d_trr.shape == (L, L, 128)
        
        f_1d, f_2d = self.pre_msa_emb(feat_1d, residue_index)
        
        f_2d += f_2d_trr
        f_1d += f_esm

        for i in range(len(self.evoformers)):
            f_1d, f_2d = self.evoformers[i](f_1d, f_2d, training=training)
            
        bf_out = self.bilstm(f_1d, training=training)
        
        bf_out = tf.reshape(bf_out, (L,))
        return bf_out
 
    def load_model(self, name):
        print ("load model:", name)
        self.load_weights(name)
        
