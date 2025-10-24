import pandas as pd
import numpy as np
import os
from config import *
from randomdatagen import *
import warnings
warnings.filterwarnings("ignore")

def loadindex(index_number, loc="../../results/Preprocess/"):
    """
    Loads training, validation, and testing indices from CSV files based on the `index_number` flag.
    
    Parameters:
        index_number (int): A flag indicating which set of indices to load.
                       - If 0: Load standard indices.
                       - If 1: Load image-specific indices.
                       - Otherwise: Generate indices based on `datacount`.
        loc (str): The directory location where the CSV files are stored. Defaults to a specific path.
    
    Returns:
        tuple: Depending on `index_number`, returns different sets of indices:
               - If `index_number` == 0:
                   (trainindex, valindex, testindex)
               - If `index_number` == 1:
                   (imageindex, trainindex, valindex, testindex)
               - Else:
                   (trainindex, valindex, testindex) generated from `datacount`
    """