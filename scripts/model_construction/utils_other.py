# Import necessary libraries for deep learning, data manipulation, and evaluation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
import xgboost as xgb
from xgboost import XGBClassifier
import pickle
import lightgbm as lgb

# Import custom modules for data loading and configuration
from fileloader import load, loadindex
from config import *

# Set the current date and time
current_date = datetime.now()

# Set the GPU device to use (device index 2)
os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

def renderresult(label, predict, filename="", supress=True):
    """
    Function to compute and optionally plot the ROC AUC score.

    Parameters:
    - label: True binary labels.
    - predict: Predicted scores or probabilities.
    - filename: Optional filename to save the plot.
    - supress: If True, only returns the AUC without plotting.

    Returns:
    - roc_auc: Area Under the Receiver Operating Characteristic Curve.
    """
    # Identify indices where either label or prediction is NaN
    na_indices = np.where(np.isnan(label) | np.isnan(predict))[0]
    # Remove NaN entries from predictions and labels
    predict = np.delete(predict, na_indices)
    label = np.delete(label, na_indices)
    
    # Compute False Positive Rate, True Positive Rate, and thresholds for ROC
    fpr, tpr, thresholds = metrics.roc_curve(label, predict, pos_label=1)
    # Calculate the Area Under the Curve (AUC)
    roc_auc = metrics.auc(fpr, tpr)
    
    if supress:
        # If suppression is enabled, return only the AUC
        return roc_auc
    
    # Plot the ROC curve
    plt.figure(dpi=500)
    lw = 2  # Line width
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.3f)" % roc_auc,
    )
    # Plot the diagonal line representing random chance
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # Set plot limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # Label axes
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    # Add legend
    plt.legend(loc="lower right")
    try:
        # Attempt to display the plot
        plt.show()
    except:
        pass  # If unable to display, skip plotting
    return roc_auc  # Return the AUC

def extract_importance(model, count=30):
    """
    Function to extract feature importance from various models.

    Parameters:
    - model: Trained machine learning model.
    - count: Number of top features to return.

    Returns:
    - top: Indices of the top features.
    - importance: Array of feature importance scores.
    """
    inv = 1  # Inversion factor (used for sorting)
    absolute = True  # Whether to take absolute value of importance

    try:
        # Attempt to get feature importance using common methods
        importance = model.feature_importance()
    except:
        try:
            importance = model.get_feature_importance()
        except:
            try:
                importance = model.feature_importances_
            except:
                try:
                    # For models like XGBoost that return a dict
                    importance = np.array(
                        list(model.get_score(importance_type="gain").values())
                    )
                except:
                    try:
                        # For statistical models with p-values
                        importance = np.log(model.pvalues)
                        coef = model.params
                        absolute = False
                        inv = -1
                    except:
                        try:
                            importance = model.coef_
                        except:
                            pass  # If all methods fail, importance remains undefined

    if absolute:
        # Take absolute value if specified
        importance = abs(importance)
    
    # Sort features based on importance
    indices = np.argsort(inv * importance)
    # Select top 'count' features
    top = indices[-1 * count:]
    # Return top feature indices and their importance scores
    return top, importance

# Reset random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

def single_disease_data(lab, Xdata, i, index_number):
    """
    Function to prepare training, validation, and test datasets for a single disease.

    Parameters:
    - lab: Label matrix where each column corresponds to a disease.
    - Xdata: Feature data.
    - i: Index of the disease column to process.
    - index_number: Image data identifier used in loadindex.

    Returns:
    - y_train: Training labels.
    - X_train: Training features.
    - y_val: Validation labels.
    - X_val: Validation features.
    - y_test: Test labels.
    - X_test: Test features.
    """
    # Load train, validation, and test indices based on index_number
    *_, trainindex, valindex, testindex = loadindex(index_number)
    
    # Extract the labels for the i-th disease
    y_col = lab[:, i]
    
    # Create masks to filter out NaN labels in validation and test sets
    val_mask = ~np.isnan(y_col[valindex])
    test_mask = ~np.isnan(y_col[testindex])
    train_mask = ~np.isnan(y_col[trainindex])
    
    # Apply masks to obtain filtered indices without NaN labels
    filtered_trainindex = trainindex[train_mask]
    filtered_valindex = valindex[val_mask]
    filtered_testindex = testindex[test_mask]
    
    # Extract training labels and features
    y_train = y_col[filtered_trainindex]
    X_train = Xdata[filtered_trainindex]
    
    # Extract validation labels and features
    y_val = y_col[filtered_valindex]
    X_val = Xdata[filtered_valindex]
    
    # Extract test labels and features
    y_test = y_col[filtered_testindex]
    X_test = Xdata[filtered_testindex]
    
    return y_train, X_train, y_val, X_val, y_test, X_test
