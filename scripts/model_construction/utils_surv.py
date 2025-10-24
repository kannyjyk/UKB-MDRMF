from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from IPython import display as IPD
import torch
import time
import os
import argparse
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
from fileloader import load, loadindex

# DeepSurv model for survival analysis using a deep neural network
class DeepSurv(nn.Module):
    def __init__(self, input_size, output_size, depth, width):
        """
        Initialize the DeepSurv model.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            depth (int): Number of hidden layers.
            width (int): Number of neurons in each hidden layer.
        """
        super(DeepSurv, self).__init__()
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(width, width))  # Add a linear layer with specified width

        normalization = []
        for i in range(depth):
            normalization.append(nn.BatchNorm1d(width, affine=False))  # Add batch normalization layers

        self.inlayer = nn.Linear(input_size, width)  # Input layer
        self.layers = nn.ModuleList(layers)  # Hidden layers
        self.normalization = nn.ModuleList(normalization)  # Batch normalization layers
        self.outlayer = nn.Linear(width, output_size)  # Output layer
        self.initialize()  # Initialize weights

    def forward(self, x):
        """
        Define the forward pass of the DeepSurv model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.inlayer(x)  # Pass through input layer
        for (layer, normal) in zip(self.layers, self.normalization):
            x = layer(x)  # Apply linear transformation
            x = normal(x)  # Apply batch normalization
            x = nn.SELU()(x)  # Apply SELU activation function
        x = self.outlayer(x)  # Pass through output layer
        return x

    def initialize(self):
        """
        Initialize the weights of the network using Kaiming Normal initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)  # Initialize weights
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize biases to zero


# Cox proportional hazards model
class Cox(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Initialize the Cox model.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features (typically 1 for Cox model).
        """
        super(Cox, self).__init__()
        self.layer = nn.Linear(input_size, output_size)  # Single linear layer
        self.initialize()  # Initialize weights

    def forward(self, x):
        """
        Define the forward pass of the Cox model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after linear transformation.
        """
        x = self.layer(x)  # Apply linear transformation
        return x

    def initialize(self):
        """
        Initialize the weights of the Cox model using Kaiming Normal initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)  # Initialize weights
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize biases to zero


# Dataset class for handling survival analysis data
class Survivaldata(Dataset):
    def __init__(self, dataframe, date, event, index):
        """
        Initialize the Survivaldata dataset.

        Args:
            dataframe (numpy.ndarray): Feature data.
            date (numpy.ndarray): Survival times.
            event (numpy.ndarray): Event indicators.
            index (numpy.ndarray or list): Indices to select data subsets.
        """
        self.df = torch.from_numpy(dataframe)[index]  # Convert features to tensor and select subset
        self.date = torch.from_numpy(date)[index]  # Convert survival times to tensor and select subset
        self.event = torch.from_numpy(event)[index]  # Convert event indicators to tensor and select subset

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.df.shape[0]  # Number of samples based on feature data

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (features, survival time, event indicator)
        """
        data = self.df[idx].float()  # Feature data as float tensor
        date = self.date[idx].float()  # Survival time as float tensor
        label = self.event[idx].float()  # Event indicator as float tensor
        return data, date, label


# Early stopping mechanism for model training
class ModelSaving:
    def __init__(self, waiting=3, printing=True):
        """
        Initialize the ModelSaving mechanism.

        Args:
            waiting (int): Number of epochs to wait before stopping if no improvement.
            printing (bool): Whether to print messages about validation loss.
        """
        self.patience = waiting  # Patience for early stopping
        self.printing = printing  # Whether to print status messages
        self.count = 0  # Counter for epochs without improvement
        self.best = None  # Best validation loss observed
        self.save = False  # Flag to indicate whether to stop training

    def __call__(self, validation_loss, model):
        """
        Call method to update the early stopping mechanism based on validation loss.

        Args:
            validation_loss (float): Current epoch's validation loss.
            model (nn.Module): Current state of the model.
        """
        if self.best is None:
            self.best = -validation_loss  # Initialize best as negative validation loss
        elif self.best <= -validation_loss:
            self.best = -validation_loss  # Update best if current loss is better
            self.count = 0  # Reset counter
        else:
            self.count += 1  # Increment counter if no improvement
            if self.printing:
                print(f'Validation loss has increased: {self.count} / {self.patience}.')
            # Stop training if patience threshold is reached
            if self.count >= self.patience:
                self.save = True  # Set flag to stop training


# Embedding-based model for survival prediction
class EmbeddingModel(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb, device):
        """
        Initialize the EmbeddingModel.

        Args:
            feature_num (int): Number of input features.
            label_num (int): Number of output labels.
            hidden_size (int): Number of neurons in hidden layers.
            y_emb (numpy.ndarray): Embedding matrix for labels.
            device (torch.device): Device to run the model on (CPU or GPU).
        """
        super(EmbeddingModel, self).__init__()

        # Initialize parameters and load embedding matrix
        self.feature_num = feature_num  # Number of input features
        self.label_num = label_num  # Number of output labels
        self.hidden_size = hidden_size  # Size of hidden layers
        self.y_emb = torch.from_numpy(y_emb).to(device).float()  # Load label embeddings as tensor on device

        # Define sequential layers for the model
        self.linears = nn.ModuleList([
            nn.Linear(feature_num, hidden_size, bias=True),  # First linear layer
            nn.Linear(hidden_size, hidden_size, bias=True),  # Second linear layer
            nn.Linear(hidden_size, y_emb.shape[1], bias=True)  # Output linear layer
        ])

        # Batch normalization layers corresponding to linear layers
        self.normalization = nn.ModuleList([
            nn.BatchNorm1d(hidden_size, affine=False),  # First batch norm
            nn.BatchNorm1d(hidden_size, affine=False),  # Second batch norm
            nn.BatchNorm1d(y_emb.shape[1], affine=False)  # Output batch norm
        ])

        self.initialize()  # Initialize weights

    def forward(self, x):
        """
        Define the forward pass of the EmbeddingModel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after embedding.
        """
        # Pass input through each linear and normalization layer, using SELU activation
        i = 0  # Layer counter
        for (linear, normal) in zip(self.linears, self.normalization):
            x = linear(x)  # Apply linear transformation
            x = normal(x)  # Apply batch normalization
            if i != len(self.linears) - 1:
                x = nn.SELU()(x)  # Apply SELU activation except for the last layer
            i += 1
        # Compute final output by matrix multiplication with embeddings
        x = torch.matmul(x, torch.transpose(self.y_emb, 0, 1))  # Matrix multiplication with embeddings
        return x

    def initialize(self):
        """
        Initialize the weights of the network using Kaiming Normal initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)  # Initialize weights
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize biases to zero


# Custom loss function for survival analysis, based on negative log likelihood
class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        """
        Initialize the NegativeLogLikelihood loss function.
        """
        super(NegativeLogLikelihood, self).__init__()

    def forward(self, risk_pred, time, event):
        """
        Compute the negative log likelihood loss for survival analysis.

        Args:
            risk_pred (torch.Tensor): Predicted risk scores.
            time (torch.Tensor): Survival times.
            event (torch.Tensor): Event indicators.

        Returns:
            torch.Tensor: Computed negative log likelihood loss.
        """
        num = time.shape[1]  # Number of samples

        # Sort observations by time in descending order
        row_indices = time.sort(dim=0, descending=True)[1]  # Get sorted indices
        col_indices = torch.tensor(list(range(num)), dtype=torch.int64, device=time.device)  # Column indices

        # Gather risk scores and events in sorted order
        risk = risk_pred[row_indices, col_indices]  # Sorted risk predictions
        e = event[row_indices, col_indices]  # Sorted event indicators

        # Apply gamma trick for numerical stability
        gamma = risk.max(dim=0)[0]  # Maximum risk score for each sample
        risk_log = (risk.sub(gamma).exp().cumsum(dim=0) + 1e-10).log().add(gamma)  # Stable computation

        # Compute negative log likelihood for survival
        neg_log_loss = -((risk - risk_log) * e).mean(dim=0).mean()  # Compute loss
        return neg_log_loss

# Concordance index calculation for evaluating survival models
def c_index(risk_pred, y, e):
    """
    Calculate the Concordance Index (C-Index) for survival models.

    The C-Index measures the predictive accuracy of a survival model by evaluating
    the concordance between predicted risks and actual survival times.

    Args:
        risk_pred (array-like or tensor): Predicted risk scores for each individual.
        y (array-like or tensor): Actual survival times.
        e (array-like or tensor): Event indicators (1 if the event occurred, 0 for censored).

    Returns:
        float: The computed Concordance Index.
    """
    # Convert y to a NumPy array if it's not already one (e.g., if it's a PyTorch tensor)
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
        # Explanation:
        # - `detach()` is used to separate the tensor from the computation graph.
        # - `cpu()` ensures the tensor is on the CPU.
        # - `numpy()` converts the tensor to a NumPy array.

    # Convert risk_pred to a NumPy array if it's not already one
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()

    # Convert e to a NumPy array if it's not already one
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    
    # Compute and return the Concordance Index using lifelines' concordance_index function
    return concordance_index(y, risk_pred, e)

def modelchar(x):
    """
    Generate a parameter code for the model based on the input integer.

    This function converts an integer input into a string representation.
    - For integers between 0 and 9 (inclusive), it returns the string of the integer.
    - For integers 10 and above, it returns uppercase alphabet letters starting from 'A'.

    Args:
        x (int): The input integer to convert.

    Returns:
        str: The corresponding parameter code as a string.

    Raises:
        ValueError: If x is negative.
    """
    if x < 0:
        raise ValueError("Input must be a non-negative integer.")

    # For x between 0 and 9, return the string representation of the number
    if 0 <= x <= 9:
        return str(x)
    # For x >= 10, convert to uppercase alphabet letters starting from 'A'
    elif x >= 10:
        # chr(65) is 'A', so 65 + x - 10 shifts the ASCII value accordingly
        return chr(65 + x - 10)
    # Optional: Handle cases where x is between 10 and 35 to map to 'A' to 'Z'
    # If x >= 36, it would go beyond 'Z' and may need additional handling



def train(net, n_epochs, waiting, train_loader, val_loader, device, lr):
    # Move model to the specified device (e.g., GPU or CPU)
    net.to(device)
    
    # Define learning rate and weight decay for the optimizer
    learning_rate = lr
    weight_decay = 0
    
    # Initialize optimizer (Adam) with model parameters, learning rate, and weight decay
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Define the loss function, move it to the specified device
    criterion = NegativeLogLikelihood().to(device)
    
    # Initialize early stopping mechanism with specified patience (waiting) and enable printing
    early_break = ModelSaving(waiting=waiting, printing=True)
    
    # Lists to store training and validation losses for each epoch
    train = []
    val = []
    val_lowest = np.inf  # Initialize the lowest validation loss for model saving

    # Training loop across the specified number of epochs
    for epoch in range(n_epochs):
        # Set model to training mode
        net.train()
        losses = []  # Track training losses for each batch in the epoch
        
        # Iterate through batches of training data
        for batch_idx, (train_inputs, train_dates, train_labels) in enumerate(train_loader):
            # Move batch data to the device and enable gradients for inputs
            train_inputs, train_dates, train_labels = (
                train_inputs.to(device),
                train_dates.to(device),
                train_labels.to(device),
            )
            train_inputs = train_inputs.requires_grad_()
            
            # Forward pass: compute model outputs and loss
            train_outputs = net(train_inputs)
            loss = criterion(train_outputs, train_dates, train_labels)
            
            # Backpropagation and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Append batch loss to list for this epoch
            losses.append(loss.data.mean().item())
        
        # Validation phase (without gradient computation)
        net.eval()
        val_losses = []  # Track validation losses for each batch
        for batch_idx, (val_inputs, val_dates, val_labels) in enumerate(val_loader):
            # Move validation data to the device
            val_inputs, val_dates, val_labels = (
                val_inputs.to(device),
                val_dates.to(device),
                val_labels.to(device),
            )
            val_outputs = net(val_inputs)
            val_loss = criterion(val_outputs, val_dates, val_labels)
            val_losses.append(val_loss.data.mean().item())
        
        # Append average training and validation losses for this epoch
        train.append(losses)
        val.append(val_losses)
        
        # Save the best model based on lowest validation loss
        if np.mean(val_losses) < val_lowest:
            val_lowest = np.mean(val_losses)
            bestmodel = net
        
        # Update early stopping with the current validation loss
        early_break(np.mean(val_losses), net)

        # Compute average losses for progress logging
        train_L = [np.mean(x) for x in train]
        val_L = [np.mean(x) for x in val]

        # Check if early stopping criteria are met
        if early_break.save:
            print(
                "Maximum waiting reached. Break the training.\t BestVal:{:.8f}\r".format(
                    min(val_L)
                )
            )
            break
        
        # Print progress for the current epoch
        print(
            "Epoch: {}\tLoss: {:.8f}({:.8f})\t BestVal:{:.8f}\r".format(
                epoch + 1, train_L[-1], val_L[-1], min(val_L)
            ),
            end="",
            flush=True,
        )
    
    # Print final epoch count on exit and return the best model
    print(f"Exit at {epoch}")
    return bestmodel


def dataset_generation(X, Y, E, index_number):
    """
    Generates training, validation, and testing datasets along with their corresponding DataLoaders.

    Parameters:
        X (np.ndarray or torch.Tensor): Feature data array.
        Y (np.ndarray or torch.Tensor): Label data array.
        E (np.ndarray or torch.Tensor): Event indicator or additional feature array.
        index_number (int or str): Flag indicating the type or source of image data.

    Returns:
        tuple: Contains the following elements:
            - numbers (list): List of sample indices.
            - trainset (Survivaldata): Training dataset.
            - valset (Survivaldata): Validation dataset.
            - testset (Survivaldata): Testing dataset.
            - train_loader (DataLoader): DataLoader for the training dataset.
            - val_loader (DataLoader): DataLoader for the validation dataset.
            - shape_data (int): Number of features in the data.
            - shape_label (int): Number of features in the labels.
    """
    # Identify the indices where 'E' contains NaN values.
    index = np.isnan(E)

    # Replace NaN values in 'E' with 0. This ensures that there are no missing values in 'E'.
    E = np.where(index, 0, E)
    # Generate a list of sample indices based on the number of samples in 'X'.
    numbers = list(range(X.shape[0]))

    # Load indices for training, validation, and testing datasets.
    # The '*' operator captures any preceding elements, while the last three are assigned to train, val, and test indices.
    # Ensure that 'loadindex' returns at least three index arrays.
    *_, trainindex, valindex, testindex = loadindex(index_number)

    # Create instances of the Survivaldata dataset for training, validation, and testing.
    # 'Survivaldata' is assumed to be a custom Dataset class defined elsewhere.
    trainset = Survivaldata(X, Y, E, trainindex)
    valset = Survivaldata(X, Y, E, valindex)
    testset = Survivaldata(X, Y, E, testindex)

    # Initialize DataLoaders for training and validation datasets.
    # 'batch_size' is set to 1024, and 'shuffle=True' ensures that the data is shuffled at every epoch.
    # Consider parameterizing 'batch_size' for flexibility.
    train_loader = DataLoader(trainset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(valset, batch_size=1024, shuffle=True)

    # Determine the number of features in the data and labels by inspecting the first batch.
    # 'next(iter(train_loader))' retrieves the first batch from the training DataLoader.
    # '[0]' accesses the features, and '[1]' accesses the labels.
    # 'shape_data' corresponds to the number of features in 'X', and 'shape_label' corresponds to the number of features in 'Y'.
    first_batch = next(iter(train_loader))
    shape_data = int(first_batch[0].shape[1])
    shape_label = int(first_batch[1].shape[1])

    # Return all relevant objects for further processing.
    return (
        numbers,
        trainset,
        valset,
        testset,
        train_loader,
        val_loader,
        shape_data,
        shape_label,
    )