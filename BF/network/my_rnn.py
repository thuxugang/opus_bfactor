# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Bidirectional

class MyBilstm(keras.layers.Layer):
    def __init__(self, num_layers, units, rate, output):
        super(MyBilstm, self).__init__()
        
        self.num_layers = num_layers
        
        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]
        
        self.output_layer = keras.layers.Dense(output)
        
    def call(self, x, training):
        # x.shape (1, L, 128)
        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, training=training)
        
        bf_predictions = self.output_layer(x)
        return bf_predictions
    
    
