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
from fileloader import load, loadindex
import torch.nn.functional as F
from config import *
import torch
from torch.autograd import Variable
import numpy as np
import torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
from utils_surv import (
    Survivaldata,
    ModelSaving,
    Cox,
    DeepSurv,
    EmbeddingModel,
    NegativeLogLikelihood,
)


def renderresult(label, predict, no_image=True):
    """
    Computes the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) for the given labels and predictions.
    Optionally plots the ROC curve.

    Parameters:
        label (np.ndarray): True binary labels (0 or 1).
        predict (np.ndarray): Predicted scores or probabilities.
        no_image (bool, optional): If False, plots the ROC curve. Defaults to True.

    Returns:
        float: Computed AUC value.
    """
    # Identify indices where either label or predict is NaN
    na_indices = np.where(np.isnan(label) | np.isnan(predict))[0]

    # Remove NaN values from predictions and labels
    predict = np.delete(predict, na_indices)
    label = np.delete(label, na_indices)

    # Compute False Positive Rate (fpr), True Positive Rate (tpr), and thresholds for ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(label, predict, pos_label=1)

    # Compute Area Under the Curve (AUC) using the trapezoidal rule
    roc_auc = metrics.auc(fpr, tpr)

    if not no_image:
        # Create a new figure for the ROC curve
        pyplot.figure()

        # Define line width for the plot
        lw = 2

        # Plot the ROC curve
        pyplot.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.3f)" % roc_auc,
        )

        # Plot the diagonal line representing a random classifier
        pyplot.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")

        # Set the limits for the x and y axes
        pyplot.xlim([0.0, 1.0])
        pyplot.ylim([0.0, 1.05])

        # Label the axes
        pyplot.xlabel("False Positive Rate")
        pyplot.ylabel("True Positive Rate")

        # Set the title of the plot
        pyplot.title("Receiver Operating Characteristic")

        # Display the legend in the lower right corner
        pyplot.legend(loc="lower right")

        try:
            # Attempt to display the plot
            pyplot.show()
        except Exception as e:
            # If an exception occurs (e.g., no display environment), silently pass
            pass

    # Return the computed AUC value
    return roc_auc


class BCEWithLogitsLossIgnoreNaN(nn.BCEWithLogitsLoss):
    """
    A custom Binary Cross Entropy loss function that ignores NaN values in the target.

    This loss function extends PyTorch's BCEWithLogitsLoss by adding functionality to mask out
    NaN values in the target tensor. This is useful in scenarios where some target values are
    undefined or should not contribute to the loss computation.

    Attributes:
        pos_weight (Tensor, optional): A manual rescaling weight given to the positive examples.
            Must be a Tensor of size `C` where `C` is the number of classes.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'mean': the sum of the output will be divided by the number of
            elements in the output. 'sum': the output will be summed. 'none': no reduction will be applied.
    """

    def forward(self, input, target):
        """
        Computes the Binary Cross Entropy loss while ignoring NaN values in the target.

        Parameters:
            input (Tensor): Tensor of arbitrary shape as the input logits.
            target (Tensor): Tensor of the same shape as `input` containing the binary labels.
                             Positions with NaN values will be ignored in the loss computation.

        Returns:
            Tensor: The computed loss value.

        Raises:
            ValueError: If there are no valid (non-NaN) targets to compute the loss.
        """
        # Create a mask tensor where True indicates valid (non-NaN) target values
        mask = ~torch.isnan(target)

        # Check if there are any valid targets to compute the loss
        if not mask.any():
            raise ValueError(
                "All target values are NaN. No valid targets to compute the loss."
            )

        # Apply the mask to both input and target tensors to filter out NaN positions
        # torch.masked_select flattens the tensors, selecting elements where mask is True
        masked_input = torch.masked_select(input, mask)
        masked_target = torch.masked_select(target, mask)

        # Compute the Binary Cross Entropy loss with logits on the masked tensors
        # This function combines a Sigmoid layer and the BCELoss in one single class
        loss = F.binary_cross_entropy_with_logits(
            masked_input,
            masked_target,
            weight=self.weight,  # Optional rescaling weight
            pos_weight=self.pos_weight,  # Optional weight of positive examples
            reduction=self.reduction,  # Specifies the reduction to apply
        )

        return loss


def custom_loss(pred, target):
    """
    Computes the Binary Cross Entropy loss with logits, ignoring positions where the target is NaN.

    This function replaces NaN values in both the predictions and targets with 1s, ensuring that the loss
    for these positions is zero. This approach effectively ignores these positions during loss computation.

    Parameters:
        pred (torch.Tensor): Predicted logits from the model. Shape: (N, *) where * means any number of additional dimensions.
        target (torch.Tensor): Ground truth binary labels. Should be the same shape as `pred`.

    Returns:
        torch.Tensor: The computed Binary Cross Entropy loss.
    """
    # Identify positions where the target is NaN
    nans = torch.isnan(target)

    # Replace predictions with 1 where the target is NaN
    # This ensures that the loss at these positions is zero since BCEWithLogitsLoss(1, 1) = 0
    pred = torch.where(nans, torch.tensor(1.0, device=pred.device), pred)

    # Replace targets with 1 where the target is NaN
    # This aligns with replacing predictions to ensure zero loss at these positions
    target = torch.where(nans, torch.tensor(1.0, device=target.device), target)

    # Initialize the BCEWithLogitsLoss function
    bceloss = nn.BCEWithLogitsLoss()

    # Compute the loss between the modified predictions and targets
    loss = bceloss(pred, target)

    return loss


class ukbdata(Dataset):
    def __init__(self, dataframe, labels):
        """
        Initializes the UKB Dataset.

        Parameters:
            dataframe (np.ndarray or torch.Tensor): Feature data. Expected shape: (N, D), where
                N is the number of samples and D is the number of features.
            labels (np.ndarray or torch.Tensor): Label data. Expected shape: (N, L), where
                N is the number of samples and L is the number of labels.
        """
        # Store the feature data
        self.df = dataframe

        # Store the label data
        self.label = labels

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves the feature and label for a given index.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (data, label) where
                data (torch.Tensor): Feature tensor of shape (D,).
                label (torch.Tensor): Label tensor of shape (L,).
        """
        # Retrieve the feature data for the given index
        # Assumes that self.df is a NumPy array or similar indexable structure
        data = self.df[idx]

        # Convert the feature data to a PyTorch tensor and ensure it is of type float
        data = torch.from_numpy(data).float()

        # Retrieve the label data for the given index
        label = self.label[idx]

        # Convert the label data to a PyTorch tensor and ensure it is of type float
        label = torch.from_numpy(label).float()

        return data, label


class ModelSaving:
    def __init__(self, waiting=3, printing=True):
        """
        Initializes the ModelSaving instance to monitor validation loss and determine when to save the model.

        Parameters:
            waiting (int, optional): The number of consecutive epochs to wait for an improvement in validation loss
                                     before deciding to save the model. Defaults to 3.
            printing (bool, optional): Flag to control whether to print messages about validation loss increases.
                                       Defaults to True.
        """
        self.patience = waiting  # Maximum number of epochs to wait without improvement
        self.printing = printing  # Flag to enable or disable printing messages
        self.count = 0  # Counter for epochs without improvement
        self.best = None  # Stores the best (lowest) validation loss observed
        self.save = False  # Flag indicating whether to save the model

    def __call__(self, validation_loss):
        """
        Updates the ModelSaving instance with the latest validation loss and determines whether to save the model.

        Parameters:
            validation_loss (float): The validation loss from the current epoch.

        Sets:
            self.save (bool): Becomes True if validation loss has not improved for 'patience' epochs.
        """
        # Initialize 'best' with the first validation loss if it's not set yet
        if self.best is None:
            self.best = validation_loss
            if self.printing:
                print(f"Initial validation loss set to: {self.best:.4f}")
            return  # No need to check for improvement on the first call

        # Check if the current validation loss is better (lower) than the best observed
        if validation_loss < self.best:
            self.best = validation_loss  # Update the best validation loss
            self.count = 0  # Reset the counter as we have an improvement
            if self.printing:
                print(f"Validation loss improved to: {self.best:.4f}")
        else:
            self.count += 1  # Increment the counter as there is no improvement
            if self.printing:
                print(
                    f"Validation loss has not improved for {self.count} out of {self.patience} epochs."
                )
            # If the counter exceeds or equals patience, set save flag to True
            if self.count >= self.patience:
                self.save = True
                if self.printing:
                    print(
                        f"No improvement for {self.patience} consecutive epochs. Triggering model save."
                    )


def train(net, loaders):
    """
    Trains the neural network model using the provided data loaders.

    Parameters:
        net (torch.nn.Module): The neural network model to train.
        loaders (tuple): A tuple containing the training and validation DataLoaders.

    Returns:
        torch.nn.Module: The trained model with the lowest validation loss observed during training.
    """
    # Define hyperparameters
    n_epochs = 1000  # Maximum number of training epochs
    waiting = 5  # Number of epochs to wait for improvement before early stopping
    learning_rate = 0.0001  # Learning rate for the optimizer
    weight_decay = 0  # Weight decay (L2 regularization) parameter

    # Unpack the training and validation DataLoaders
    train_loader, val_loader = loaders

    # Initialize the network (e.g., reset weights if necessary)
    net.initialize()

    # Move the network to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)


    # Initialize the optimizer with model parameters, learning rate, and weight decay
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Define the loss criterion using the custom loss function
    criterion = custom_loss

    # Initialize the early stopping mechanism
    early_break = ModelSaving(waiting=waiting, printing=True)

    # Lists to store training and validation losses for each epoch
    train_losses = []
    val_losses = []

    # Variable to track the lowest validation loss observed
    val_lowest = np.inf

    # Variable to store the best-performing model
    best_model = None

    # Start the training loop over epochs
    for epoch in range(n_epochs):
        # Set the network to training mode
        net.train()

        # List to store losses for the current epoch
        epoch_train_losses = []

        # Iterate over batches in the training DataLoader
        for batch_idx, (train_inputs, train_labels) in enumerate(train_loader):
            # Move inputs and labels to the computation device and wrap them in Variables
            train_inputs, train_labels = Variable(train_inputs.to(device)), Variable(
                train_labels.to(device)
            )

            # Enable gradient computation for inputs (if necessary)
            train_inputs.requires_grad_()

            # Forward pass: compute model outputs
            train_outputs = net(train_inputs)

            # Compute the loss between outputs and true labels
            loss = criterion(train_outputs, train_labels)

            # Zero the gradients of the optimizer to prevent accumulation
            optimizer.zero_grad()

            # Backward pass: compute gradients
            loss.backward()

            # Update model parameters based on gradients
            optimizer.step()

            # Append the current batch loss to the epoch's training loss list
            epoch_train_losses.append(loss.data.mean().item())

        # Set the network to evaluation mode (disables dropout, batchnorm, etc.)
        net.eval()

        # List to store validation losses for the current epoch
        epoch_val_losses = []

        # Disable gradient computation for validation to speed up computations and reduce memory usage
        with torch.no_grad():
            # Iterate over batches in the validation DataLoader
            for batch_idx, (val_inputs, val_labels) in enumerate(val_loader):
                # Move inputs and labels to the computation device and wrap them in Variables
                val_inputs, val_labels = Variable(val_inputs.to(device)), Variable(
                    val_labels.to(device)
                )

                # Forward pass: compute model outputs
                val_outputs = net(val_inputs)

                # Compute the loss between outputs and true labels
                val_loss = criterion(val_outputs, val_labels)

                # Append the current batch validation loss to the epoch's validation loss list
                epoch_val_losses.append(val_loss.data.mean().item())

        # Append the epoch's training and validation losses to their respective lists
        train_losses.append(epoch_train_losses)
        val_losses.append(epoch_val_losses)

        # Compute the average validation loss for the current epoch
        avg_val_loss = np.mean(epoch_val_losses)

        # Check if the current epoch's validation loss is the lowest observed so far
        if avg_val_loss < val_lowest:
            val_lowest = avg_val_loss  # Update the lowest validation loss
            best_model = net  # Save the current model as the best model

        # Update the early stopping mechanism with the current average validation loss
        early_break(avg_val_loss)

        # If early stopping criteria are met, terminate training
        if early_break.save:
            print("Maximum waiting reached. Breaking the training.")
            break

    # Return the best-performing model observed during training
    return best_model


def modelchar(x):
    """
    Converts an integer index to its corresponding character representation.

    For indices 0 through 9:
        - Returns the string representation of the digit.
          Example: 0 -> "0", 1 -> "1", ..., 9 -> "9"

    For indices 10 and above:
        - Returns uppercase alphabet characters starting from 'A'.
          Example: 10 -> 'A', 11 -> 'B', ..., 35 -> 'Z'

    Parameters:
        x (int): The integer index to convert.

    Returns:
        str: The corresponding character representation.

    Raises:
        ValueError: If 'x' is negative or exceeds the supported range (0-35).
        TypeError: If 'x' is not an integer.
    """
    # Ensure that 'x' is an integer
    if not isinstance(x, int):
        raise TypeError(f"Input must be an integer, got {type(x).__name__} instead.")

    # Handle indices from 0 to 9 by converting them to their string representations
    if 0 <= x <= 9:
        return str(x)

    # Handle indices from 10 upwards by mapping them to uppercase letters starting from 'A'
    elif 10 <= x <= 35:
        # ASCII value of 'A' is 65
        return chr(65 + x - 10)

    # If 'x' is outside the supported range, raise an error
    else:
        raise ValueError("Input 'x' must be in the range 0 to 35 inclusive.")



class POPDxModel(nn.Module):
    """
    POPDxModel is a neural network model designed for predictive diagnostics.
    It consists of a sequence of linear layers followed by a ReLU activation and a
    matrix multiplication with an embedding matrix for labels.
    """

    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        """
        Initializes the POPDxModel.

        Args:
            feature_num (int): The number of input features.
            label_num (int): The number of output labels.
            hidden_size (int): The size of the hidden layer.
            y_emb (torch.Tensor): Embedding matrix for the labels with shape (label_num, embedding_dim).
        """
        super(POPDxModel, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb

        # Define a list of linear layers
        self.linears = nn.ModuleList(
            [
                nn.Linear(
                    feature_num, hidden_size, bias=True
                ),  # First linear layer: input to hidden
                nn.Linear(
                    hidden_size, y_emb.shape[1], bias=True
                ),  # Second linear layer: hidden to embedding dimension
            ]
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feature_num).

        Returns:
            torch.Tensor: Output tensor after processing, typically of shape (batch_size, label_num).
        """
        # Pass the input through each linear layer sequentially
        for i, linear in enumerate(self.linears):
            x = linear(x)
            # Optional: Add intermediate activations or other operations here if needed

        # Apply ReLU activation function
        x = torch.relu(x)

        # Perform matrix multiplication with the transposed label embedding matrix
        x = torch.matmul(x, torch.transpose(self.y_emb, 0, 1))

        return x

    def initialize(self):
        """
        Initializes the weights of all linear layers using Kaiming normal initialization.
        This helps in maintaining the variance of activations through the network, promoting better training.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight
                )  # Initialize weights with Kaiming normal
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize biases to zero


class POPDxModelC(nn.Module):
    """
    POPDxModelC is an enhanced neural network model for predictive diagnostics.
    It consists of multiple linear layers with ReLU activations and a final
    matrix multiplication with a label embedding matrix.
    """

    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        """
        Initializes the POPDxModelC.

        Args:
            feature_num (int): Number of input features.
            label_num (int): Number of output labels.
            hidden_size (int): Size of the hidden layers.
            y_emb (torch.Tensor): Embedding matrix for labels with shape (label_num, embedding_dim).
        """
        super(POPDxModelC, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb

        # Define a list of linear layers
        self.linears = nn.ModuleList(
            [
                nn.Linear(
                    feature_num, hidden_size, bias=True
                ),  # Input to first hidden layer
                nn.Linear(
                    hidden_size, hidden_size, bias=True
                ),  # First hidden layer to second hidden layer
                nn.Linear(
                    hidden_size, hidden_size, bias=True
                ),  # Second hidden layer to third hidden layer
                nn.Linear(
                    hidden_size, y_emb.shape[1], bias=True
                ),  # Third hidden layer to embedding dimension
            ]
        )

    def forward(self, x):
        """
        Defines the forward pass of the POPDxModelC.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feature_num).

        Returns:
            torch.Tensor: Output tensor after processing, typically of shape (batch_size, label_num).
        """
        # Sequentially apply each linear layer
        for i, linear in enumerate(self.linears):
            x = linear(x)
            # Apply ReLU activation after the first three linear layers
            if i <= 2:
                x = torch.relu(x)

        # Perform matrix multiplication with the transposed label embedding matrix
        x = torch.matmul(x, torch.transpose(torch.tensor(self.y_emb), 0, 1))
        return x

    def initialize(self):
        """
        Initializes the weights of all linear layers using Kaiming normal initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)  # Initialize weights
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize biases to zero


class POPDxModelC1(nn.Module):
    """
    POPDxModelC1 is a variant of the POPDxModelC with ReLU activations applied after every linear layer.
    """

    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        """
        Initializes the POPDxModelC1.

        Args:
            feature_num (int): Number of input features.
            label_num (int): Number of output labels.
            hidden_size (int): Size of the hidden layers.
            y_emb (torch.Tensor): Embedding matrix for labels with shape (label_num, embedding_dim).
        """
        super(POPDxModelC1, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb

        # Define a list of linear layers
        self.linears = nn.ModuleList(
            [
                nn.Linear(
                    feature_num, hidden_size, bias=True
                ),  # Input to first hidden layer
                nn.Linear(
                    hidden_size, hidden_size, bias=True
                ),  # First hidden layer to second hidden layer
                nn.Linear(
                    hidden_size, hidden_size, bias=True
                ),  # Second hidden layer to third hidden layer
                nn.Linear(
                    hidden_size, y_emb.shape[1], bias=True
                ),  # Third hidden layer to embedding dimension
            ]
        )

    def forward(self, x):
        """
        Defines the forward pass of the POPDxModelC1.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feature_num).

        Returns:
            torch.Tensor: Output tensor after processing, typically of shape (batch_size, label_num).
        """
        # Sequentially apply each linear layer followed by ReLU activation
        for linear in self.linears:
            x = linear(x)
            x = torch.relu(x)

        # Perform matrix multiplication with the transposed label embedding matrix
        x = torch.matmul(x, torch.transpose(self.y_emb, 0, 1))
        return x

    def initialize(self):
        """
        Initializes the weights of all linear layers using Kaiming normal initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)  # Initialize weights
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize biases to zero


class pheNN(nn.Module):
    """
    pheNN is a customizable neural network with configurable depth and width.
    It consists of an input layer, multiple hidden layers with ReLU activations, and an output layer.
    """

    def __init__(self, input_size, output_size, depth, width):
        """
        Initializes the pheNN.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output units.
            depth (int): Number of hidden layers.
            width (int): Number of units in each hidden layer.
        """
        super(pheNN, self).__init__()
        layers = []
        for i in range(depth):
            layers.append(
                nn.Linear(width, width)
            )  # Add a linear layer for each hidden layer
        self.inlayer = nn.Linear(input_size, width)  # Input layer
        self.layers = nn.ModuleList(layers)  # Hidden layers
        self.outlayer = nn.Linear(width, output_size)  # Output layer

    def forward(self, x):
        """
        Defines the forward pass of the pheNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor after processing, typically of shape (batch_size, output_size).
        """
        x = self.inlayer(x)  # Apply input layer
        for layer in self.layers:
            x = layer(x)  # Apply hidden layer
            x = nn.ReLU()(x)  # Apply ReLU activation
        x = self.outlayer(x)  # Apply output layer
        return x

    def initialize(self):
        """
        Initializes the weights of the network.
        Currently, this method does nothing and can be customized as needed.
        """
        pass


class LogisticRegression(nn.Module):
    """
    LogisticRegression is a simple logistic regression model using a single linear layer.
    """

    def __init__(self, input_size, output_size):
        """
        Initializes the LogisticRegression model.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output units (typically 1 for binary classification).
        """
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(
            input_size, output_size
        )  # Linear layer for logistic regression
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for output probabilities

    def forward(self, x):
        """
        Defines the forward pass of the LogisticRegression model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor after applying the linear layer, typically of shape (batch_size, output_size).
        """
        out = self.linear(x)  # Apply linear transformation
        return out  # Return raw scores (logits)

    def initialize(self):
        """
        Initializes the weights of the linear layer.
        Currently, this method does nothing and can be customized as needed.
        """
        pass


def dataset_generation(Xdata: np.ndarray, lab: np.ndarray, index_number: int) -> tuple:
    """
    Generates training, validation, and testing datasets along with their DataLoaders.

    Parameters:
        Xdata (numpy.ndarray): The feature data matrix.
        lab (numpy.ndarray): The label data matrix.
        index_number (int): A flag indicating the type of image data to load, used in `loadindex`.

    Returns:
        tuple: Contains sample indices, datasets, DataLoaders, and feature dimensions.
            - numbers (list): List of sample indices.
            - trainset (ukbdata): Training dataset.
            - valset (ukbdata): Validation dataset.
            - testset (ukbdata): Testing dataset.
            - train_loader (DataLoader): DataLoader for the training dataset.
            - val_loader (DataLoader): DataLoader for the validation dataset.
            - shape_data (int): Number of features in the data.
            - shape_label (int): Number of features in the labels.

    """

    # Create a list of sample indices
    numbers = list(range(lab.shape[0]))

    # Load train, validation, and test indices
    indices = loadindex(index_number)
    if len(indices) < 3:
        raise ValueError(
            "loadindex must return at least three values: trainindex, valindex, testindex."
        )
    *_, trainindex, valindex, testindex = indices

    # Create datasets using the custom `ukbdata` class
    trainset = ukbdata(Xdata[trainindex], lab[trainindex])
    valset = ukbdata(Xdata[valindex], lab[valindex])
    testset = ukbdata(Xdata[testindex], lab[testindex])

    # Create DataLoaders for training and validation datasets
    train_loader = DataLoader(trainset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(valset, batch_size=1024, shuffle=True)

    # Retrieve the first batch from train_loader to determine feature shapes
    first_batch = next(iter(train_loader))

    shape_data = first_batch[0].shape[1]
    shape_label = first_batch[1].shape[1]

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
