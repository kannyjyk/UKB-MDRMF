import os
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
import numpy as np
import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import warnings
import torch
import torch.nn as nn
import os
from fileloader import load,loadindex
import torch.nn.functional as F
from config import *
warnings.filterwarnings("ignore")


def renderresult(label, predict, supress=True):
    na_indices = np.where(np.isnan(label) | np.isnan(predict))[0]
    predict = np.delete(predict, na_indices)
    label = np.delete(label, na_indices)
    fpr, tpr, thresholds = metrics.roc_curve(label, predict, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    if supress:
        return roc_auc
    pyplot.figure()
    lw = 2
    pyplot.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.3f)" % roc_auc,
    )
    pyplot.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.title("Receiver operating characteristic")
    pyplot.legend(loc="lower right")
    try:
        pyplot.show()
    except:
        pass
    return roc_auc


class BCEWithLogitsLossIgnoreNaN(nn.BCEWithLogitsLoss):
    def forward(self, input, target):
        mask = ~torch.isnan(target)
        masked_input = torch.masked_select(input, mask)
        masked_target = torch.masked_select(target, mask)
        return F.binary_cross_entropy_with_logits(
            masked_input,
            masked_target,
        )


def custom_loss(pred, target):
    nans = torch.isnan(target)
    pred = torch.where(nans, torch.tensor(1), pred)
    target = torch.where(nans, torch.tensor(1), target)
    bceloss = torch.nn.BCEWithLogitsLoss()(pred, target)
    return bceloss


class ukbdata(Dataset):
    def __init__(self, dataframe, labels):
        self.df = dataframe
        self.label = labels

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data = torch.from_numpy(self.df[idx]).float()
        label = torch.from_numpy(self.label[idx]).float()
        return data, label


class ModelSaving:
    def __init__(self, waiting=3, printing=True):
        self.patience = waiting
        self.printing = printing
        self.count = 0
        self.best = None
        self.save = False

    def __call__(self, validation_loss):
        if not self.best:
            self.best = -validation_loss
        elif self.best <= -validation_loss:
            self.best = -validation_loss
            self.count = 0
        elif self.best > -validation_loss:
            self.count += 1
            print(f"Validation loss has increased: {self.count} / {self.patience}.")
            if self.count >= self.patience:
                self.save = True



def modelchar(x):
    if x >= 0 and x <= 9:
        return str(x)
    elif x >= 10:
        return chr(65 + x - 10)


class POPDxModel(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(POPDxModel, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb
        self.linears = nn.ModuleList(
            [
                nn.Linear(feature_num, hidden_size, bias=True),
                nn.Linear(hidden_size, y_emb.shape[1], bias=True),
            ]
        )

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
        x = torch.relu(x)
        x = torch.matmul(x, torch.transpose(self.y_emb, 0, 1))
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


class POPDxModelC(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(POPDxModelC, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb
        self.linears = nn.ModuleList(
            [
                nn.Linear(feature_num, hidden_size, bias=True),
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.Linear(hidden_size, y_emb.shape[1], bias=True),
            ]
        )

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i <= 2:
                x = torch.relu(x)
        x = torch.matmul(x, torch.transpose(self.y_emb, 0, 1))
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


class POPDxModelC1(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(POPDxModelC1, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb
        self.linears = nn.ModuleList(
            [
                nn.Linear(feature_num, hidden_size, bias=True),
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.Linear(hidden_size, y_emb.shape[1], bias=True),
            ]
        )

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
            x = torch.relu(x)
        x = torch.matmul(x, torch.transpose(self.y_emb, 0, 1))
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


class pheNN(nn.Module):
    def __init__(self, input_size, output_size, depth, width):
        super(pheNN, self).__init__()
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(width, width))
        self.inlayer = nn.Linear(input_size, width)
        self.layers = nn.ModuleList(layers)
        self.outlayer = nn.Linear(width, output_size)

    def forward(self, x):
        x = self.inlayer(x)
        for layer in self.layers:
            x = layer(x)
            x = nn.ReLU()(x)
        x = self.outlayer(x)
        return x

    def initialize(self):
        pass


class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        return out

    def initialize(self):
        pass
np.random.seed(0)
torch.manual_seed(0)
for category in range(5,0,-1):
    for model in [0,1,2,3]:
        for hyperp in range(4):
            for image_X in [0]:
                Xdata, _,lab = load(image_X, category)
                print(
                    "model", model, "cat", category, len(Xdata), "params", hyperp, "Xtype", image_X
                )
                print(Xdata.shape, lab.shape)
                print("loaded")
                numbers = list(range(lab.shape[0]))
                *_, trainindex, valindex, testindex = loadindex(image_X)
                learning_rate = 0.0001
                weight_decay = 0
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                trainset = ukbdata(Xdata[trainindex], lab[trainindex])
                valset = ukbdata(Xdata[valindex], lab[valindex])
                testset = ukbdata(Xdata[testindex], lab[testindex])
                train_loader = DataLoader(trainset, batch_size=1024, shuffle=True)
                val_loader = DataLoader(valset, batch_size=256, shuffle=True)
                fn=f'{path_prefix}/pred/{category}{modelchar(model)}{image_X}_{hyperp}'
                                
                def trainauc(fn):
                    try:
                        nnnet=torch.load(fn+'model')
                    except:
                        return
                    whole_loader = DataLoader(trainset, batch_size=int(len(trainset)/10))
                    init=True
                    for i in whole_loader:
                        inputs, labels = i
                        labels=labels.cpu().detach().numpy()
                        out = nnnet(inputs.to(device)).cpu().detach().numpy()
                        out = torch.sigmoid(torch.from_numpy(out)).numpy()
                        if init:
                            outall=out
                            labelsall=labels
                            init=False
                        else:
                            outall=np.concatenate([outall,out])
                            labelsall=np.concatenate([labelsall,labels])
                    trainaucresult=[]
                    for i in range(labelsall.shape[1]):
                        auc = renderresult(labelsall[:, i], outall[:, i])
                        trainaucresult.append(auc)
                    np.save(fn+'trainauc',trainaucresult)
                print(fn)
                trainauc(fn)