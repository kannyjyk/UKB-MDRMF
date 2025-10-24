import os
import argparse
import warnings
from datetime import datetime

# Environment configurations
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
gpu = 6
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display as IPD

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Project-specific modules
from survutil import (
    Survivaldata,
    ModelSaving,
    Cox,
    DeepSurv,
    EmbeddingModel,
    NegativeLogLikelihood,
    c_index,  # Moved this into the grouped import
)
from fileloader import load, loadindex
from config import *
from util_surv import *


save_path = '../../results/ComparativeAnalysis/'
net_loc = '../../results/Disease_diagnosis/surv/610_0model'
df = pd.read_csv('../../data/phecode_icd10.csv')
input_file = pd.read_csv('../../data/phecode.csv')

d_dict = {
    'infection': '^A|^B[0-8]',
    'Binfection': '^A[0-7]',
    'Vinfection': '^A[8-9]|^B[0-2]|^B3[0-4]',
    'cancer': '^C[0-8]|C9[0-7]',
    'lung_cancer': '^C34',
    'breast_cancer': '^C50',
    'prostate_cancer': '^C61',
    'leukaemia': '^C8[1-9]|^C9[0-6]',
    'BI_disorders': '^D[5-8]',
    'Anemia': '^D5|^D6[0-4]',
    'endocrine_disorders': '^E[0-8]|^E90',
    'diabetes': '^E1[0-4]',
    'obesity': '^E66',
    'MB_disorders': '^F',
    'dementia': '^F0[0-3]|^G30|^G31',
    'mood_disorders': '^F3[0-9]',
    'neurotic_disorders': '^F4[0-8]',
    'nervous_system_disorders': '^G',
    'parkinson_disease': '^G2[0-2]',
    'sleep_disorders': '^G47',
    'eye_disorders': '^H[0-5]',
    'ear_disorders': '^H[6-9]',
    'circulatory_system_disorders': '^I',
    'hypertension': '^I1[0-5]',
    'ischemic_heart_diseases': '^I2[0-5]',
    'arrhythmias': '^I4[6-9]',
    'heart_failure': '^I50',
    'stroke': '^I6[0-1]|^I6[3-4]',
    'peripheral_artery_diseases': '^I7[0-9]',
    'Respiratory_system_disorders': '^J',
    'COPD': '^J4[0-4]|^J47',
    'Astma': '^J4[5-6]',
    'Digestive_system_disorders': '^K',
    'inflammatory_bowel_diseases': '^K5[2-5]',
    'Liver_diseases': '^K7[0-7]',
    'skin_disorders': '^L',
    'musculoskeletal_system_disorders': '^M',
    'osteoarthritis': '^M1[5-9]',
    'genitourinary_system_disorders': '^N',
    'renal_failure': '^N1[7-9]'
}
d_dict_a = {
    'auc1':'^(I050|I051|I052|I060|I061|I062|I21(\d{0,3})?|I22(\d{0,3})?|I23(\d{0,3})?|I24[1-9]|I25(\d{0,3})?|I34[02]|I35[012]|I50(\d{0,3})?|I63|I64|G45)',
    'auc2':'^(I2[1-3]|^I241|^I252)',
    'auc3':'^N80',
    'auc4':'^I48',
    'auc5':'^I48|^I6[3-4]'
}
d_dict_c = {
    'cindex2chd':'^(I20|I21|I22|I23|I24|I252|I25|I46|R96|R98|Z951|T822)',
    'cindex2af':'^I48',
    'cindex2t2d':'E1[0-1]',
    'cindex2bc':'C50',
    'cindex2pc':'C61',
    'cindex3':'^(G45|I11|I13|I2[0-5]|I42|I46|I50|I6[0-7]|I71|I72|I74)',
    'cindex4CVD':'^(I2[1-5]|I6)',
    'cindex4CHD':'^I2[1-5]',
    'cindex4stroke':'I6[0-9]'
}

def icd_to_phelogit(dname,type='pattern'):
    if type == 'pattern':
        PHECODE = df[df['ICD10'].str.contains(dname)]['PheCode'].unique()
    else:
        PHECODE = df[df['ICD10'].str.contains('|'.join(dname))]['PheCode'].unique()
    PHECODE = PHECODE[~np.isnan(PHECODE)]
    # find the index of each PHECODE in the input_file 1st column
    index = []
    for i in PHECODE:
        try:
            index.append(input_file[input_file['0'] == i].index.values[0])
        except:
            pass
    logits = np.zeros(1560)
    logits[index] = 1
    logits = logits.astype(bool)
    return logits

np.random.seed(0)
torch.manual_seed(0)

for key in d_dict_c.keys():
    raw={}
    print(key)
    device = torch.device("cuda")
    net=torch.load(net_loc,map_location=device)
    category = 6
    model = 1
    hyperp = 0
    image_X = 0
    imageX = image_X
    only = False
    waiting = 15
    epoch = 1000
    learning_rate = 0.005
    input_logit = icd_to_phelogit(d_dict_c[key])
    net.outlayer=nn.Linear(in_features=300, out_features=1, bias=True)
    learning_rate = 0.005
    X, Y, E = load(image_X=image_X, category=category, only=False,inputcolumn=input_logit)
    index = np.isnan(E)
    E = np.where(index, torch.tensor(0), E)
    numbers = list(range(X.shape[0]))
    *_, trainindex, valindex, testindex = loadindex(image_X)
    trainset = Survivaldata(X, Y, E, trainindex)
    valset = Survivaldata(X, Y, E, valindex)
    testset = Survivaldata(X, Y, E, valindex)
    train_loader = DataLoader(trainset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(valset, batch_size=1024, shuffle=True)
    shape_data = int(next(iter(train_loader))[0].shape[1])
    shape_label = int(next(iter(train_loader))[1].shape[1])
   
    if model == 0:  #'cox':
        nnet = Cox(shape_data, shape_label)
    elif model == 1:  #'deepsurv':
        nnet = DeepSurv(shape_data, shape_label, 5, 300)
    elif model == 2:  #'popdx':
        label_emb = np.load(
            "../../data/Embedding/phe.npy", allow_pickle=True
        )
        hidden_size = 400
        nnet = EmbeddingModel(shape_data, shape_label, hidden_size, label_emb, device)
    elif model == 3:  #'mith':
        label_emb = np.load(
             "../../data/Embedding/conv.npy", allow_pickle=True
        )
        hidden_size = 400
        nnet = EmbeddingModel(shape_data, shape_label, hidden_size, label_emb, device)
    nnet = train(nnet, epoch, waiting, train_loader, val_loader, device, learning_rate)
    nnet.to("cuda")
    whole_loader = DataLoader(testset, batch_size=len(testset))
    xtest, ytest, etest = next(iter(whole_loader))
    ytest.cpu()
    etest.cpu()
    out = nnet(xtest.to("cuda")).cpu().detach().numpy()
    cindex = []
    for j in range(out.shape[1]):
        try:
            na_indices = np.where(np.isnan(ytest[:, j]))[0]
            o = np.delete(-out[:, j], na_indices)
            y = np.delete(ytest[:, j], na_indices)
            e = np.delete(etest[:, j], na_indices)
            cindex.append(c_index(o, y, e))
        except Exception as e:
            cindex.append(np.nan)
    loc = key
    try:
        os.mkdir(f"./{loc}surv/{str(key)}/")
    except:
        pass
    np.save(f"./{loc}surv/{str(key)}/{category}{modelchar(model)}{imageX}_{hyperp}", cindex)
    torch.save(nnet, f"./{loc}surv/{str(key)}/{category}{modelchar(model)}{imageX}_{hyperp}model")
    print("complete")
