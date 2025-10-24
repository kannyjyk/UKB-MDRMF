from lifelines.utils import concordance_index
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from IPython import display as IPD
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

class DeepSurv(nn.Module):
    def __init__(self, input_size, output_size, depth, width):
        super(DeepSurv, self).__init__()
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(width, width))

        normalization = []
        for i in range(depth):
            normalization.append(nn.BatchNorm1d(width,affine=False))
        self.inlayer = nn.Linear(input_size, width)
        self.layers = nn.ModuleList(layers)
        self.normalization = nn.ModuleList(normalization)
        self.outlayer = nn.Linear(width, output_size)
        self.initialize()

    def forward(self, x):
        x = self.inlayer(x)
        for (layer, normal) in zip(self.layers, self.normalization):
            x = layer(x)
            x = normal(x)
            x = nn.SELU()(x)
        x = self.outlayer(x)
        return x
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

class Cox(nn.Module):
    def __init__(self, input_size, output_size):
        super(Cox, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
        self.initialize()

    def forward(self, x):
        x = self.layer(x)
        return x
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    

""" class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()

    def forward(self, risk_pred, time, event):
        num = time.shape[1]
        row_indices = time.sort(axis = 0, descending=True)[1]
        col_indices = torch.tensor(list(range(num)),dtype=torch.int64)
        risk = risk_pred[row_indices, col_indices]
        # t = time[row_indices, col_indices]
        e = event[row_indices, col_indices]
        gamma = risk.max(axis=0)[0]
        risk_log = risk.sub(gamma).exp().cumsum(axis = 0).log().add(gamma)
        neg_log_loss = -((risk - risk_log) * e).mean(axis=0).mean()
        return neg_log_loss """
    
class Survivaldata(Dataset):
    def __init__(self, dataframe,date,event,index):
        self.df = torch.from_numpy(dataframe)[index]
        self.date = torch.from_numpy(date)[index]
        self.event = torch.from_numpy(event)[index]
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        data = self.df[idx].float()
        date = self.date[idx].float()
        label = self.event[idx].float()
        return data, date, label
    
class ModelSaving:
    def __init__(self, waiting=3, printing=True):
        self.patience = waiting
        self.printing = printing
        self.count = 0
        self.best = None
        self.save = False

    def __call__(self, validation_loss, model):
        if not self.best:
            self.best = -validation_loss
        elif self.best <= -validation_loss:
            self.best = -validation_loss
            self.count = 0
        elif self.best > -validation_loss:
            self.count += 1
            print(
                f'Validation loss has increased: {self.count} / {self.patience}.')
            if self.count >= self.patience:
                self.save = True

class EmbeddingModel(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb, device):
        super(EmbeddingModel, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = torch.from_numpy(y_emb).to(device).float()
        self.linears = nn.ModuleList([nn.Linear(feature_num, hidden_size, bias=True),
                                      nn.Linear(hidden_size, hidden_size, bias=True),
                                    nn.Linear(hidden_size, y_emb.shape[1], bias=True)])
        self.normalization = nn.ModuleList([nn.BatchNorm1d(hidden_size,affine=False),
                                            nn.BatchNorm1d(hidden_size,affine=False),
                                            nn.BatchNorm1d(y_emb.shape[1],affine=False)])
        self.initialize()

    def forward(self, x):
        i=0
        for (linear,normal) in zip(self.linears,self.normalization):
            x = linear(x)
            x = normal(x)
            if i!=len(self.linears)-1:
                x = nn.SELU()(x)
            i+=1
        x = torch.matmul(x, torch.transpose(self.y_emb, 0, 1))
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()

    def forward(self, risk_pred, time, event):
        num = time.shape[1]
        row_indices = time.sort(axis = 0, descending=True)[1]
        col_indices = torch.tensor(list(range(num)),dtype=torch.int64)
        risk = risk_pred[row_indices, col_indices]
        # t = time[row_indices, col_indices]
        e = event[row_indices, col_indices]
        gamma = risk.max(axis=0)[0]
        risk_log = (risk.sub(gamma).exp().cumsum(axis = 0)+0.0000000001).log().add(gamma)
        neg_log_loss = -((risk - risk_log) * e).mean(axis=0).mean()
        return neg_log_loss
    
def c_index(risk_pred, y, e):
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)
