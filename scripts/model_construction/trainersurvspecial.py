from trainersurv import *
import os
import torch

if __name__ == "__main__":
    # Split the 'loc' variable into 'loc' and 'position' based on the '$' delimiter.
    # This assumes that 'loc' is a string containing exactly one '$' character.
    # Example: If loc = "directory$path/to/data", then after split:
    # loc = "directory"
    # position = "path/to/data"
    loc, position = loc.split("$")

    # Load the dataset using the 'load' function with specified parameters.
    # Parameters:
    # - index_number: Identifier or flag related to image data.
    # - category: Category identifier for data selection or processing.
    # - only: Additional flag or parameter for data loading (purpose depends on implementation).
    # - xloc: Path to the directory containing priority data for training.
    # Note: Ensure that 'index_number', 'category', and 'only' are defined before this block.
    X, Y, E = load(index_number, category=category, only=only, xloc=position)

    # Print a confirmation message indicating that the data has been loaded.
    print("loaded")

    # Construct and print a specific directory path for saving results or other purposes.
    # - loc: Base directory.
    # - "special": A subdirectory or category for organizing results.
    # - position.split('/')[-2]: Extracts the second last component of the 'position' path.
    #   For example, if position = "path/to/data", then position.split('/')[-2] = "to".
    #   Adjust this based on your directory structure to ensure correctness.
    print(f"./{loc}special/{position.split('/')[-2]}")

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
    ) = dataset_generation(X, Y, E, index_number)

    # Select and initialize the appropriate model based on the 'model_selection' function.
    # Parameters:
    # - model: Integer flag indicating which model to select (e.g., 0 for Cox, 1 for DeepSurv).
    # - shape_data: Number of input features in the data.
    # - shape_label: Number of output features in the labels.
    nnet = model_selection(model, shape_data, shape_label)

    # Train the selected model using the 'train' function.
    # Parameters:
    # - nnet: The initialized neural network model to train.
    # - epoch: Number of training epochs.
    # - waiting: Additional parameter or flag related to training (context needed).
    # - train_loader: DataLoader for the training dataset.
    # - val_loader: DataLoader for the validation dataset.
    # - device: Computation device (CPU or GPU).
    # - learning_rate: Learning rate for the optimizer.
    nnet = train(nnet, epoch, waiting, train_loader, val_loader, device, learning_rate)

    nnet.to(device)

    # Evaluate the trained model using the 'evaluate' function.
    # Parameters:
    # - nnet: The trained neural network model.
    # - testset: The testing dataset.
    # - 'special/surv': A keyword to categorize the saved results
    evaluate(nnet, testset, "special/surv")
