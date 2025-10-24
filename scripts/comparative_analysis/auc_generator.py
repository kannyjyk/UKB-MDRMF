import sys
import os
import warnings
import argparse

# Set up the system path
sys.path.append("../../disease_diagnosis")

# Environment configurations
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
gpu = 6
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

# Import project-specific modules
from config import *
from utils import *
from fileloader import *

# Standard libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt  # Renamed for clarity

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn import metrics


save_path = '../../results/ComparativeAnalysis/'
net_loc = '../../results/Disease_diagnosis/surv/610_0model'
df = pd.read_csv('../../data/phecode_icd10.csv')
input_file = pd.read_csv('../../data/phecode.csv')

d_dict_a = {
    'auc1':'^(I050|I051|I052|I060|I061|I062|I21(\d{0,3})?|I22(\d{0,3})?|I23(\d{0,3})?|I24[1-9]|I25(\d{0,3})?|I34[02]|I35[012]|I50(\d{0,3})?|I63|I64|G45)',
    'auc2':'^(I2[1-3]|^I241|^I252)',
    'auc3':'^N80',
    'auc4':'^I48',
    'auc5':'^I48|^I6[3-4]'
}
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



def icd_to_phelogit(dname,type='pattern'):
    if type == 'pattern':
        PHECODE = df[df['ICD10'].str.contains(dname)]['PheCode'].unique()
    else:
        PHECODE = df[df['ICD10'].str.contains('|'.join(dname))]['PheCode'].unique()
    PHECODE = PHECODE[~np.isnan(PHECODE)]
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

for key in d_dict_a.keys():
    raw={}
    print(key)
    device = torch.device("cuda")
    net=torch.load(net_loc,map_location=device)
    category = 5
    model = 1
    hyperp = 0
    image_X = 0
    input_logit = icd_to_phelogit(d_dict_a[key])
   
    # Load the dataset
    Xdata, _,lab = load(image_X, category,inputcolumn=input_logit)
    # lab turn to (1,), when there's no 1 in the row, it will turn to 0, else 1
    net.outlayer=nn.Linear(in_features=300, out_features=1, bias=True)
    numbers = list(range(lab.shape[0]))
    *_, trainindex, valindex, testindex = loadindex(image_X)
    learning_rate = 0.0001
    weight_decay = 0
    trainset = ukbdata(Xdata[trainindex], lab[trainindex])
    valset = ukbdata(Xdata[valindex], lab[valindex])
    testset = ukbdata(Xdata[testindex], lab[testindex])
    train_loader = DataLoader(trainset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(valset, batch_size=1024, shuffle=True)
    shape_data = int(next(iter(train_loader))[0].shape[1])
    shape_label = int(next(iter(train_loader))[1].shape[1])
    
    # Train the selected model
    nnnet = train(net, 10, 15)
    whole_loader = DataLoader(testset, batch_size=len(testset))
    inputs, labels = next(iter(whole_loader))
    out = nnnet(inputs.to(device)).cpu().detach().numpy()
    out = torch.sigmoid(torch.from_numpy(out)).numpy()
    aucresult = []
    
    auc = renderresult(labels.cpu().detach().numpy()[:], out[:])
    aucresult.append(auc)
    loc=key
    try:
        os.mkdir(f"./{loc}")
    except:
        pass
    print(np.nanmean(aucresult))

    # save results
    np.save(f"./{loc}/{category}{modelchar(model)}{image_X}_{hyperp}", aucresult)
    np.save(f"./{loc}/{image_X}lab", labels)
    torch.save(nnnet, f"./{loc}/{category}{modelchar(model)}{image_X}_{hyperp}model")
    print("complete")


