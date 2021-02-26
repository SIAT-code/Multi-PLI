import keras
from keras.models import Model
from keras.layers import Input,Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import pandas as pd
import matplotlib.pyplot as plt
import string
import numpy as np
from keras.utils import multi_gpu_model
import csv
import json
import time
import multiprocessing as mp
from multiprocessing import Process,Queue,Pool,Manager,Lock
import re
from random import shuffle
import data_process
import json
seed_value= 2020

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

def read_class(filename):
    x_prot=[]
    x_comp=[]
    y_=[]
    pro_comp_df = pd.read_csv(filename)
    for i in range(len(pro_comp_df)):
        pro=str(pro_comp_df['seqs'][i])
        comp=str(pro_comp_df['rdkit_smiles'][i])
        label = pro_comp_df['label'][i]
        pro_result = data_process_v1.one_hot_protein(pro)
        pro_result = np.expand_dims(pro_result,axis=-1)
        comp_result = data_process_v1.one_hot_smiles(comp)
        comp_result = np.expand_dims(comp_result,axis=-1)
        x_prot.append(pro_result)
        x_comp.append(comp_result)
        y_.append(label)
    print(filename,"has read successed!")
    print('shape:', len(x_prot))
    x_prot=np.array(x_prot)
    x_prot = x_prot.reshape([-1, 1200, 20, 1])
    x_comp=np.array(x_comp)
    x_comp = x_comp.reshape([-1, 200, 67, 1])
    y_=np.array(y_)
    y_=y_.reshape([-1,1])
    return x_prot,x_comp,y_

def read_reg(filename):
    x_prot=[]
    x_comp=[]
    y_=[]
    pro_comp_df = pd.read_csv(filename)
    for i in range(len(pro_comp_df)):
        pro=str(pro_comp_df['seqs'][i])
        comp=str(pro_comp_df['rdkit_smiles'][i])
        label = pro_comp_df['Affinity-Value'][i]
        pro_result = data_process_v1.one_hot_protein(pro)
        pro_result = np.expand_dims(pro_result,axis=-1)
        comp_result = data_process_v1.one_hot_smiles(comp)
        comp_result = np.expand_dims(comp_result,axis=-1)
        x_prot.append(pro_result)
        x_comp.append(comp_result)
        y_.append(label)
    print(filename,"has read successed!")
    print('shape:', len(x_prot))
    x_prot=np.array(x_prot)
    x_prot = x_prot.reshape([-1, 1200, 20, 1])
    x_comp=np.array(x_comp)
    x_comp = x_comp.reshape([-1, 200, 67, 1])
    y_=np.array(y_)
    y_=y_.reshape([-1,1])
    return x_prot,x_comp,y_

def read_class_nolabel(filename):
    x_prot=[]
    x_comp=[]
    y_=[]
    pro_comp_df = pd.read_csv(filename)
    for i in range(len(pro_comp_df)):
        pro=str(pro_comp_df['seqs'][i])
        comp=str(pro_comp_df['rdkit_smiles'][i])
        pro_result = data_process_v1.one_hot_protein(pro)
        pro_result = np.expand_dims(pro_result,axis=-1)
        comp_result = data_process_v1.one_hot_smiles(comp)
        comp_result = np.expand_dims(comp_result,axis=-1)
        x_prot.append(pro_result)
        x_comp.append(comp_result)
    print(filename,"has read successed!")
    print('shape:', len(x_prot))
    x_prot=np.array(x_prot)
    x_prot = x_prot.reshape([-1, 1200, 20, 1])
    x_comp=np.array(x_comp)
    x_comp = x_comp.reshape([-1, 200, 67, 1])
    return x_prot,x_comp

def read_class_generator(filename, batch_size, ft_flag = False):
    pro_comp_df = pd.read_csv(filename)
    x_prot=[]
    x_comp=[]
    y_=[]
    while 1:
        pro_comp_df = pro_comp_df.sample(frac=1).reset_index(drop=True)
        for i in range(len(pro_comp_df)):
            pro=str(pro_comp_df['seqs'][i])
            comp=str(pro_comp_df['rdkit_smiles'][i])
            label = pro_comp_df['label'][i]
            pro_result = data_process_v1.one_hot_protein(pro)
            pro_result = np.expand_dims(pro_result,axis=-1)
            comp_result = data_process_v1.one_hot_smiles(comp)
            comp_result = np.expand_dims(comp_result,axis=-1)
            x_prot.append(pro_result)
            x_comp.append(comp_result)
            y_.append(label)
            if len(x_prot)==batch_size:
                x_prot=np.array(x_prot)
                x_prot = x_prot.reshape([-1, 1200, 20, 1])
                x_comp=np.array(x_comp)
                x_comp = x_comp.reshape([-1, 200, 67, 1])
                y_=np.array(y_)
                y_=y_.reshape([-1,1])
                if ft_flag:
                    yield ({'protein_input':x_prot, 'comp_input':x_comp}, {'average_1':y_})
                else:
                    yield (x_prot, x_comp, y_)  
                x_prot=[]
                x_comp=[]
                y_=[]


def read_reg_generator(filename, batch_size, ft_flag = False):
    pro_comp_df = pd.read_csv(filename)
    x_prot=[]
    x_comp=[]
    y_=[]
    while 1:
        pro_comp_df = pro_comp_df.sample(frac=1).reset_index(drop=True)
        for i in range(len(pro_comp_df)):
            pro=str(pro_comp_df['seqs'][i])
            comp=str(pro_comp_df['rdkit_smiles'][i])
            label = pro_comp_df['Affinity-Value'][i]
            pro_result = data_process_v1.one_hot_protein(pro)
            pro_result = np.expand_dims(pro_result,axis=-1)
            comp_result = data_process_v1.one_hot_smiles(comp)
            comp_result = np.expand_dims(comp_result,axis=-1)
            x_prot.append(pro_result)
            x_comp.append(comp_result)
            y_.append(label)
            if len(x_prot)==batch_size:
                x_prot=np.array(x_prot)
                x_prot = x_prot.reshape([-1, 1200, 20, 1])
                x_comp=np.array(x_comp)
                x_comp = x_comp.reshape([-1, 200, 67, 1])
                y_=np.array(y_)
                y_=y_.reshape([-1,1])
                if ft_flag:
                    yield ({'protein_input':x_prot, 'comp_input':x_comp}, {'average_1':y_}) 
                else:
                    yield (x_prot, x_comp, y_)
                x_prot=[]
                x_comp=[]
                y_=[]



