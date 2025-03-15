import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from captum.attr import (
    GradientShap,
    DeepLift,
    IntegratedGradients,
    KernelShap,
    DeepLiftShap,
)
import os
import sys
from config import *
from fileloader import load,loadindex
import pickle

from utils_surv import *
if __name__ == "__main__":
    # Retrieve command-line arguments, excluding the script name
    args = sys.argv[1:]
    
    # Parse and assign command-line arguments
    st = int(args[0])      # Start index
    end = int(args[1])     # End index
    gpu = int(args[2])     # GPU identifier
    folder = str(args[3])  # Model folder path
    
    # Set the CUDA_VISIBLE_DEVICES environment variable to specify which GPU to use
    if gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define category and image_X parameters
    category = 6  # Example category identifier (modify as needed)
    image_X = 0   # Example image identifier or parameter (modify as needed)
    
    # Load data using the custom 'load' function
    # Xdata: Feature data
    # _: Placeholder for a returned value that's not used
    # lab: Labels or additional data
    X, Y, E = load(image_X=image_X, category=category, only=False)
    
    # Generate datasets and DataLoaders using the 'dataset_generation' function.
    # This function splits the data into training, validation, and testing sets and creates corresponding DataLoaders.
    (
        numbers,
        trainset,
        valset,
        testset,
        train_loader,
        val_loader,
        shape_data,
        shape_label,
    ) = dataset_generation(X, Y, E, image_X)
    
    whole_loader = DataLoader(testset, batch_size=3000, shuffle=True)
    # Load a pre-trained model from the specified path and move it to the designated device
    # Ensure that the model path and device are correctly specified
    model = torch.load(f"./{folder}surv/{category}10_0model", map_location=device).to(device)
    
    # Initialize interpretability methods with the loaded model
    gs = GradientShap(model)         # Gradient SHAP
    dl = DeepLift(model)             # DeepLIFT
    ig = IntegratedGradients(model)  # Integrated Gradients
    ks = KernelShap(model)           # Kernel SHAP
    dls = DeepLiftShap(model)        # DeepLIFT SHAP
    
    # Retrieve a single batch of training data from the train_loader
    # This is used as a reference input for some interpretability methods
    train, labels = next(iter(train_loader))
    X_train = train.to(device)  # Move training data to the designated device
    
    # Define the output directory for storing importance scores
    putdir = f"./survimp/"
    
    # Create the output directory if it doesn't already exist
    try:
        os.mkdir(putdir)
    except FileExistsError:
        # Directory already exists; no action needed
        pass
    
    # Iterate over the target indices from 'st' to 'end'
    for tg in range(st, end):
        # Check if the target index has already been processed and saved
        if str(tg) in os.listdir(putdir):
            continue  # Skip to the next target index if already processed
        
        print(tg)  # Print the current target index being processed
        
        # Initialize a dictionary to store importance scores for different methods
        dumpdict = {"dl": [],"ig": [],"gs": [],"ks": [],"dls": []}
        
        # Iterate over all batches in the 'whole_loader'
        for t in range(len(whole_loader)):
            # Retrieve the next batch of inputs and labels
            inputs, labels = next(iter(whole_loader))
            
            # Move the input data to the designated device (CPU or GPU)
            X_test = inputs.to(device)
            
            # Compute Gradient SHAP attributions for the current target index
            # - X_test: Input data for which to compute attributions
            # - X_train: Reference data used by Gradient SHAP
            # - target=tg: The specific target index for which to compute attributions
            gsout = gs.attribute(X_test, X_train, target=tg).detach().cpu().numpy()
            
            # Uncomment the following lines to compute other types of attributions
            # igout = ig.attribute(X_test, target=tg).detach().cpu().numpy()
            # dlout = dl.attribute(X_test, target=tg).detach().cpu().numpy()
            # ksout = ks.attribute(X_test, target=tg).detach().cpu().numpy()
            # dlsout = dls.attribute(X_test, X_train, target=tg).detach().cpu().numpy()
            
            # Append the computed Gradient SHAP attributions to the dumpdict
            dumpdict["gs"].append(gsout)
            
            # Uncomment the following lines to store other attributions
            # dumpdict["ig"].append(igout)
            # dumpdict["dl"].append(dlout)
            # dumpdict['ks'].append(ksout)
            # dumpdict['dls'].append(dlsout)
        
        # Define the output file path for storing the summed Gradient SHAP attributions
        output_file_path = os.path.join(putdir, f"{tg}sumed")
        
        # Save the concatenated and summed Gradient SHAP attributions to the output file
        with open(output_file_path, "wb+") as file:
            # Concatenate all Gradient SHAP outputs along the first axis and compute the sum
            summed_gs = np.concatenate(dumpdict["gs"]).sum(0)
            pickle.dump(summed_gs, file)
        
        # Uncomment the following lines to save other attributions
        
        # with open(os.path.join(putdir, f"{tg}_ig.sumed"), "wb+") as file:
        #     summed_ig = np.concatenate(dumpdict["ig"]).sum(0)
        #     pickle.dump(summed_ig, file)
        
        # with open(os.path.join(putdir, f"{tg}_dl.sumed"), "wb+") as file:
        #     summed_dl = np.concatenate(dumpdict["dl"]).sum(0)
        #     pickle.dump(summed_dl, file)
        
        # with open(os.path.join(putdir, f"{tg}_ks.sumed"), "wb+") as file:
        #     summed_ks = np.concatenate(dumpdict["ks"]).sum(0)
        #     pickle.dump(summed_ks, file)
        
        # with open(os.path.join(putdir, f"{tg}_dls.sumed"), "wb+") as file:
        #     summed_dls = np.concatenate(dumpdict["dls"]).sum(0)
        #     pickle.dump(summed_dls, file)
        
