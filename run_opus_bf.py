# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""
import sys
sys.path.append("/work/home/xugang/projects/esm/esm-main")

import os
import warnings
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
import numpy as np
import time
import multiprocessing

import torch
import esm

from BF.inference_utils import make_input, toPDB
from BF.inference import run_BF

def preparation(multi_iter):
    preparation_config, filename = multi_iter
    
    if (not os.path.exists(os.path.join(preparation_config["tmp_files_path"], filename + ".input1d.npz")) or
        not os.path.exists(os.path.join(preparation_config["tmp_files_path"], filename + ".input2d.npz"))):    
        make_input(preparation_config, filename)
        
if __name__ == '__main__':

    #============================Parameters====================================
    list_path = r"./list_casp15"
    files_path = []
    f = open(list_path)
    for i in f.readlines():
        files_path.append(i.strip())
    f.close()
    list_path = r"./list_cameo65"
    f = open(list_path)
    for i in f.readlines():
        files_path.append(i.strip())
    f.close()
    list_path = r"./list_cameo82"
    f = open(list_path)
    for i in f.readlines():
        files_path.append(i.strip())
    f.close()
    print (len(files_path))    
    
    preparation_config = {}
    preparation_config["batch_size"] = 1
    preparation_config["type"] = "seq"
    # preparation_config["type"] = "struct"
    preparation_config["tmp_files_path"] = os.path.join(os.path.abspath('.'), "tmp_files")
    preparation_config["pdb_path"] = "testsets"
    preparation_config["output_path"] = os.path.join(os.path.abspath('.'), "predictions_seq")
    # preparation_config["output_path"] = os.path.join(os.path.abspath('.'), "predictions_struct")
    preparation_config["mkdssp_path"] = os.path.join(os.path.abspath('.'), "BF/mkdssp/mkdssp")
    
    num_cpu = 56
    
    #============================Parameters====================================
    
    
    #============================Preparation===================================
    print('Preparation start...')
    start_time = time.time()
    
    multi_iters = []
    filenames = []
    for file_path in files_path:
        filename = file_path.split('/')[-1].split('.')[0]
        multi_iters.append([preparation_config, filename])
        filenames.append(filename)
        
    pool = multiprocessing.Pool(num_cpu)
    pool.map(preparation, multi_iters)
    pool.close()
    pool.join()

    preparation_config["filenames"] = filenames

    run_time = time.time() - start_time
    print('Preparation done..., time: %3.3f' % (run_time))  
    #============================Preparation===================================

    #============================ESM2===============================
    print('Cal ESM2 start...')
    start_time = time.time()
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    
    names = []
    seqs = []
    multi_iters = []
    for file_path in files_path:
        filename = file_path.split('/')[-1].split('.')[0]
        names.append(filename)
        f = open(os.path.join(preparation_config["tmp_files_path"], filename+".fasta"))
        seqs.append(f.readlines()[1].strip())
        f.close()
    
    assert len(names) == len(seqs)
    for name, seq in zip(names, seqs):
        if os.path.exists(os.path.join(preparation_config["tmp_files_path"], name+".esm.npz")): continue
        # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
        data = [
            (name, seq),
        ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        # batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        
        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33][0]
        
        token_representations = np.array(token_representations, dtype=np.float16)
        np.savez_compressed(os.path.join(preparation_config["tmp_files_path"], name+".esm"), l=token_representations)  
    run_time = time.time() - start_time
    print('Cal ESM2 done..., time: %3.3f' % (run_time))          
    #============================ESM2===============================
        
    #============================OPUS-BF===============================
    run_BF(preparation_config)
    #============================OPUS-BF===============================
    
    #============================mkpdb===============================
    toPDB(preparation_config)
    #============================mkpdb===============================
    print('All done...')
    