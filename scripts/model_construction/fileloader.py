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
    if index_number == 0:
        # Load standard training indices from 'train_index.csv'
        trainindex = pd.read_csv(f"{loc}train_index.csv")["train"].to_numpy()
        
        # Load standard validation indices from 'val_index.csv'
        valindex = pd.read_csv(f"{loc}val_index.csv")["val"].to_numpy()
        
        # Load standard testing indices from 'test_index.csv'
        testindex = pd.read_csv(f"{loc}test_index.csv")["test"].to_numpy()
        
        return trainindex, valindex, testindex
    
    elif index_number == 1:
        # Load image-specific training indices from 'image_train_index.csv'
        trainindex = pd.read_csv(f"{loc}image_train_index.csv")["train"].to_numpy()
        
        # Load image-specific validation indices from 'image_val_index.csv'
        valindex = pd.read_csv(f"{loc}image_val_index.csv")["val"].to_numpy()
        
        # Load image-specific testing indices from 'image_test_index.csv'
        testindex = pd.read_csv(f"{loc}image_test_index.csv")["test"].to_numpy()
        
        # Load image indices from 'image_index.csv'
        imageindex = pd.read_csv(f"{loc}image_index.csv")["image"].to_numpy()
        
        return imageindex, trainindex, valindex, testindex
    
    else:
        # Assuming `datacount` is defined globally or elsewhere in the code
        # Generate training indices as the first 80% of the data
        trainindex = range(int(datacount * 0.8))
        
        # Generate validation indices as the next 10% of the data
        valindex = range(int(datacount * 0.8), int(datacount * 0.9))
        
        # Generate testing indices as the last 10% of the data
        testindex = range(int(datacount * 0.9), datacount)
        
        return trainindex, valindex, testindex



def load(
    index_number,
    category,
    only=False,
    xloc=Xblocklocation,
    yloc="../../results/cache",  # Current path for yloc
    MRIloc="",
    fullimgspec=0,
):
    """
    Loads and processes data based on the provided parameters.

    Parameters:
        index_number (int): Flag indicating the type of image data to load.
                       - 0: Standard image data.
                       - 1: Image data, Deprecated (raises ValueError).
        category (int): Category of the data to load.
                        - 1-6: Different data categories.
                        - 7: Generates random test data.
        only (bool, optional): If True, processes only a subset of Xdata. Defaults to False.
        xloc (str, optional): Location path for X data blocks. Defaults to Xblocklocation.
        yloc (str, optional): Location path for Y data. Defaults to "../../results/cache".
        MRIloc (str, optional): Location path for MRI data. Defaults to a specific Preprocess path.
        fullimgspec (int, optional): Specifies the imputation method to use. Defaults to 0.

    Returns:
        tuple: Depending on the inputs, returns different combinations of Xdata, y, and e.
               - If category == 7:
                   (X, y, e) from generate()
               - Else:
                   (Xdata, y, e) after processing
                   
    Raises:
        ValueError: If index_number == 1, as it's not implemented.
    """
    
    # If category is 7, generate random test data
    if category == 7:
        X, y, e = generate()  # Assumes generate() is defined elsewhere
        return X, y, e

    # List of imputation method names
    impnamel = [
        "hybrid",
        "mim",
        "zero_mode",
        "mean_mode",
        "missforest",
        "median_mode",
        "hybrid_II",
    ]
    
    # If a specific imputation method is selected, update the xloc path accordingly
    if fullimgspec != 0:
        raise ValueError("Update mri imputation locations")
        xloc = f"{Xblocklocation}../imputed_{impnamel[fullimgspec]}/"
    
    # List of MRI file names corresponding to different imputation methods
    fullmrilist = [
        "mripc_imputed_hybrid.csv",
        "mripc_imputed_mim.csv",
        "mripc_imputed_zero_mode.csv",
        "mripc_imputed_mean_mode.csv",
        "mripc_imputed_missforest.csv",
        "mripc_imputed_median_mode.csv",
        "mripc_imputed_hybrid_II.csv",
    ]
    
    # Load Y data based on the category
    if category == 6:
        # For category 6, load Y and e from specific image paths
        y = np.load(f"{yloc}/coximg.npy")
        e = np.load(f"{yloc}/0-1img.npy")
    else:
        # For other categories, load Y and e from standard paths
        y = np.load(f"{yloc}/cox.npy")
        e = np.load(f"{yloc}/0-1.npy")
    
    # Load X data based on the category
    if category < 6:
        # For categories 1-5, load the corresponding block
        Xdata = np.load(f"{xloc}blk{category}.npy")
    else:
        # For category 6, load block5.npy
        Xdata = np.load(f"{xloc}blk5.npy")
    
    # Handle different index_number flags
    if index_number == 1:
        # If index_number is 1, raise an error as it's not implemented
        raise ValueError("Not implemented")
    elif index_number == 0:
        if category == 6:
            print("cat6")  # Debug statement indicating category 6 processing
            # Load MRI data based on the selected imputation method
            # mri = pd.read_csv(
            #     "/home/tangbr/UKB/compare_missing/pipeline/scripts/Process_missingness/code_compare/code_no_adjust_v1/data_output/comparison_0518/"
            #     + fullmrilist[fullimgspec]
            # ).to_numpy()
        
            mri=pd.read_csv('../../results/Process_missingness/mri_imputed.csv').to_numpy()
            # Horizontally stack Xdata with MRI data
            Xdata = np.hstack((Xdata, mri))
            print(Xdata.shape)  # Debug statement showing the shape of Xdata after stacking
        
        if only:
            if category != 1:
                # If category is not 1, load the previous category's block
                Xdata0 = np.load(f"{xloc}blk{category-1}.npy")
                n0 = Xdata0.shape[1]  # Number of columns in the previous block
            else:
                n0 = 0  # If category is 1, start from column 0
            n = Xdata.shape[1]  # Total number of columns in current Xdata
            # Slice Xdata to include only columns from n0 to n
            Xdata = Xdata[:, n0:n]
    
    # Return the processed Xdata, y, and e
    return Xdata, y, e
