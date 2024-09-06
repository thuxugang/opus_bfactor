# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import tensorflow as tf
import numpy as np
from BF.mkinputs import PDBreader, structure, vector, Geometry, getPhiPsiOmega
    
ss8 = "CSTHGIEB"
ss8_dict = {}
for k,v in enumerate(ss8):
    ss8_dict[v] = k
    
def get_psp_dict():
    resname_to_psp_dict = {}
    resname_to_psp_dict['G'] = [1,4,7]
    resname_to_psp_dict['A'] = [1,3,7]
    resname_to_psp_dict['V'] = [1,7,12]
    resname_to_psp_dict['I'] = [1,3,7,12]
    resname_to_psp_dict['L'] = [1,5,7,12]
    resname_to_psp_dict['S'] = [1,2,5,7]
    resname_to_psp_dict['T'] = [1,7,15]
    resname_to_psp_dict['D'] = [1,5,7,11]
    resname_to_psp_dict['N'] = [1,5,7,14]
    resname_to_psp_dict['E'] = [1,6,7,11]
    resname_to_psp_dict['Q'] = [1,6,7,14]
    resname_to_psp_dict['K'] = [1,5,6,7,10]
    resname_to_psp_dict['R'] = [1,5,6,7,13]
    resname_to_psp_dict['C'] = [1,7,8]
    resname_to_psp_dict['M'] = [1,6,7,9]
    resname_to_psp_dict['F'] = [1,5,7,16]
    resname_to_psp_dict['Y'] = [1,2,5,7,16]
    resname_to_psp_dict['W'] = [1,5,7,18]
    resname_to_psp_dict['H'] = [1,5,7,17]
    resname_to_psp_dict['P'] = [7,19]
    return resname_to_psp_dict

def get_pc7_dict():
    resname_to_pc7_dict = {'A': [-0.350, -0.680, -0.677, -0.171, -0.170, 0.900, -0.476],
                'C': [-0.140, -0.329, -0.359, 0.508, -0.114, -0.652, 0.476],
                'D': [-0.213, -0.417, -0.281, -0.767, -0.900, -0.155, -0.635],
                'E': [-0.230, -0.241, -0.058, -0.696, -0.868, 0.900, -0.582],
                'F': [ 0.363, 0.373, 0.412, 0.646, -0.272, 0.155, 0.318],
                'G': [-0.900, -0.900, -0.900, -0.342, -0.179, -0.900, -0.900],
                'H': [ 0.384, 0.110, 0.138, -0.271, 0.195, -0.031, -0.106],
                'I': [ 0.900, -0.066, -0.009, 0.652, -0.186, 0.155, 0.688],
                'K': [-0.088, 0.066, 0.163, -0.889, 0.727, 0.279, -0.265],
                'L': [ 0.213, -0.066, -0.009, 0.596, -0.186, 0.714, -0.053],
                'M': [ 0.110, 0.066, 0.087, 0.337, -0.262, 0.652, -0.001],
                'N': [-0.213, -0.329, -0.243, -0.674, -0.075, -0.403, -0.529],
                'P': [ 0.247, -0.900, -0.294, 0.055, -0.010, -0.900, 0.106],
                'Q': [-0.230, -0.110, -0.020, -0.464, -0.276, 0.528, -0.371],
                'R': [ 0.105, 0.373, 0.466, -0.900, 0.900, 0.528, -0.371],
                'S': [-0.337, -0.637, -0.544, -0.364, -0.265, -0.466, -0.212],
                'T': [ 0.402, -0.417, -0.321, -0.199, -0.288, -0.403, 0.212],
                'V': [ 0.677, -0.285, -0.232, 0.331, -0.191, -0.031, 0.900],
                'W': [ 0.479, 0.900, 0.900, 0.900, -0.209, 0.279, 0.529],
                'Y': [ 0.363, 0.417, 0.541, 0.188, -0.274, -0.155, 0.476]}
    return resname_to_pc7_dict

resname_to_psp_dict = get_psp_dict()
resname_to_pc7_dict = get_pc7_dict()

def mtx2bins(x_ref, start, end, nbins, mask):
    bins = np.linspace(start, end, nbins)
    x_true = np.digitize(x_ref, bins).astype(np.uint8)
    x_true[mask] = 0
    return np.eye(nbins+1)[x_true][...,:-1]

def read_ss(ss_result):
    ss_len = len(ss_result)
    
    ss8 = np.zeros(ss_len*8).reshape(ss_len, 8)
    for i in range(ss_len):
        ss8[i][ss8_dict[ss_result[i]]] = 1
        
    ss3 = np.zeros((ss_len, 3))
    ss3[:,0] = np.sum(ss8[:,:3],-1)
    ss3[:,1] = np.sum(ss8[:,3:6],-1)
    ss3[:,2] = np.sum(ss8[:,6:8],-1)
    
    return ss8, ss3

def make_input(preparation_config, filename):
    
    pdb_path = os.path.join(preparation_config["pdb_path"], filename + ".pdb")
    
    atomsData = PDBreader.readPDB(pdb_path) 
    residuesData = structure.getResidueData(atomsData) 
    
    fasta = "".join([i.resname for i in residuesData])
    fw = open(os.path.join(preparation_config["tmp_files_path"], filename + ".fasta"), 'w')
    fw.writelines(">"+filename+"\n")
    fw.writelines(fasta)
    fw.close()
    seq_len = len(fasta)
    
    dihedralsData = getPhiPsiOmega.getDihedrals(residuesData)
    assert seq_len == len(dihedralsData)

    pps = []
    for i in dihedralsData:
        pps.append([np.sin(np.deg2rad(i.pp[0])), np.cos(np.deg2rad(i.pp[0])), 
                    np.sin(np.deg2rad(i.pp[1])), np.cos(np.deg2rad(i.pp[1])),
                    np.sin(np.deg2rad(i.pp[2])), np.cos(np.deg2rad(i.pp[2]))])
    assert len(pps) == seq_len
    
    cmd = preparation_config["mkdssp_path"] + ' ' + pdb_path
    print (cmd) 
    
    output = os.popen(cmd).read()
    ss = []
    for i in output.split("\n"):
        if i != "" and i[0] != '#':
            ss.append(i.strip().split()[2].strip())
    assert seq_len == len(ss)
    ss8, ss3 = read_ss(ss)

    pc7 = np.zeros((seq_len, 7))
    for i in range(seq_len):
        pc7[i] = resname_to_pc7_dict[fasta[i]]
    
    psp = np.zeros((seq_len, 19))
    for i in range(seq_len):
        psp19 = resname_to_psp_dict[fasta[i]]
        for j in psp19:
            psp[i][j-1] = 1

    inputs_1d = np.concatenate((pc7, psp, ss8, ss3, pps),axis=1)
    assert inputs_1d.shape == (seq_len, 43)

    inputs_1d = np.array(inputs_1d, dtype=np.float16)
    np.savez_compressed(os.path.join(preparation_config["tmp_files_path"], filename+".input1d"), l=inputs_1d)  
    
    length = seq_len
    dist_ref, omega_ref, theta_ref, phi_ref = \
        np.zeros((length, length)), np.zeros((length, length)), np.zeros((length, length)), np.zeros((length, length))
        
    for i in range(length):
        residue_a = residuesData[i]
        a_ca = residue_a.atoms["CA"].position
        a_n = residue_a.atoms["N"].position
        if "CB" in residue_a.atoms:
            a_cb = residue_a.atoms["CB"].position
        else:
            if residue_a.resname == 'G':
                res_name = 'A'
            else:
                res_name = residue_a.resname
            geo = Geometry.geometry(res_name)
            a_cb = vector.calculateCoordinates(
                residue_a.atoms["C"], residue_a.atoms["N"], residue_a.atoms["CA"], geo.CA_CB_length, geo.C_CA_CB_angle, geo.N_C_CA_CB_diangle)
            
        for j in range(length):
            if i == j:
                continue
            residue_b = residuesData[j]
            b_ca = residue_b.atoms["CA"].position
            if "CB" in residue_b.atoms:
                b_cb = residue_b.atoms["CB"].position
            else:
                if residue_b.resname == 'G':
                    res_name = 'A'
                else:
                    res_name = residue_b.resname
                geo = Geometry.geometry(res_name)
                b_cb = vector.calculateCoordinates(
                    residue_b.atoms["C"], residue_b.atoms["N"], residue_b.atoms["CA"], geo.CA_CB_length, geo.C_CA_CB_angle, geo.N_C_CA_CB_diangle)

            dist_ref[i][j] = np.linalg.norm(a_cb - b_cb)
            omega_ref[i][j] = np.deg2rad(vector.calc_dihedral(a_ca, a_cb, b_cb, b_ca))
            theta_ref[i][j] = np.deg2rad(vector.calc_dihedral(a_n, a_ca, a_cb, b_cb))
            phi_ref[i][j] = np.deg2rad(vector.calc_angle(a_ca, a_cb, b_cb))

    p_dist  = mtx2bins(dist_ref,     2.0,  20.0, 37, mask=(dist_ref > 20))
    p_omega = mtx2bins(omega_ref, -np.pi, np.pi, 37, mask=(p_dist[...,0]==1))
    p_theta = mtx2bins(theta_ref, -np.pi, np.pi, 37, mask=(p_dist[...,0]==1))
    p_phi   = mtx2bins(phi_ref,      0.0, np.pi, 19, mask=(p_dist[...,0]==1))
    feat    = np.concatenate([p_theta, p_phi, p_dist, p_omega],-1)
    
    assert feat.shape == (length, length, 130)

    feat = np.array(feat, dtype=np.int8)
    np.savez_compressed(os.path.join(preparation_config["tmp_files_path"], filename+".input2d"), l=feat)  

#=============================================================================    
def read_inputs(filenames, preparation_config):
    """
    7pc + 19psp + 8ss + 3ss + 6pp + 1prob / trr130
    """
    inputs_1ds = []
    inputs_2ds = []
    inputs_esms = []
    inputs_total_len = 0

    assert len(filenames) == 1
    for filename in filenames:

        fasta_path = os.path.join(preparation_config["tmp_files_path"], filename+'.fasta')
    
        with open(fasta_path, 'r') as r:
            fasta_content = [i.strip() for i in r.readlines()]
        fasta = fasta_content[1]
        seq_len = len(fasta)

        feat_esm = np.load(os.path.join(preparation_config["tmp_files_path"], filename+".esm.npz"))['l']
        feat_esm = feat_esm[1:-1,:]
        assert feat_esm.shape == (seq_len, 1280)
        
        feat_1d = np.load(os.path.join(preparation_config["tmp_files_path"], filename+".input1d.npz"))['l']
        assert feat_1d.shape == (seq_len, 43)
        
        feat_2d = np.load(os.path.join(preparation_config["tmp_files_path"], filename+".input2d.npz"))['l']
        assert feat_2d.shape == (seq_len, seq_len, 130)

        inputs_total_len += seq_len
        
        inputs_1ds.append(feat_1d)
        inputs_2ds.append(feat_2d)
        inputs_esms.append(feat_esm)
        
    inputs_1ds = np.array(inputs_1ds)
    inputs_2ds = np.array(inputs_2ds)
    inputs_esms = np.array(inputs_esms)
            
    return inputs_1ds, inputs_2ds, inputs_esms, inputs_total_len

class InputReader(object):

    def __init__(self, data_list, preparation_config):

        self.data_list = data_list
        self.preparation_config = preparation_config
        self.dataset = tf.data.Dataset.from_tensor_slices(self.data_list).batch(1)          
        
        print ("Data Size:", len(self.data_list)) 
    
    def read_file_from_disk(self, filenames_batch):
        
        filenames_batch = [bytes.decode(i) for i in filenames_batch.numpy()]
        inputs_1ds_batch, inputs_2ds_batch, inputs_esms_batch, inputs_total_len = \
            read_inputs(filenames_batch, self.preparation_config)

        inputs_esms_batch = tf.cast(inputs_esms_batch, dtype=tf.float32)
        inputs_1ds_batch = tf.cast(inputs_1ds_batch, dtype=tf.float32)
        inputs_2ds_batch = tf.cast(inputs_2ds_batch, dtype=tf.float32)

        return filenames_batch, inputs_1ds_batch, inputs_2ds_batch, inputs_esms_batch, inputs_total_len
            
def toPDB(preparation_config):
    
    for filename in preparation_config["filenames"]:
        bfs = np.load(os.path.join(preparation_config["output_path"], filename + ".bfs.npy"),'r')
        f = open(os.path.join(preparation_config["pdb_path"], filename + ".pdb"),'r')
        atomsDatas = []
        idx = 0
        for line in f.readlines():   
            if (line.strip() == ""):
                break
            else:
                if (line[:4] == 'ATOM'):
                    name1 = line[11:16].strip()
                    # bf = line[60:66].strip()
                    if name1 == 'CA':
                        bf_pred = bfs[idx]
                        idx += 1
                        bf_pred = format(bf_pred,".2f")
                        bf_pred_len = len(list(bf_pred))
                        string = " "*(6-bf_pred_len) + bf_pred    
                        line = line[:60] + string + line[66:]
                        atomsDatas.append(line.strip())
                    else:
                        bf_pred = format(0,".2f")
                        bf_pred_len = len(list(bf_pred))
                        string = " "*(6-bf_pred_len) + bf_pred     
                        line = line[:60] + string + line[66:]                        
                        atomsDatas.append(line.strip())
        f.close()
        assert idx == bfs.shape[0]
        
        fw = open(os.path.join(preparation_config["output_path"], filename + "_bf.pdb"), 'w')
        for i in atomsDatas:
            fw.writelines(i + "\n")
        fw.close()
    