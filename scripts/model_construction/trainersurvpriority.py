from trainersurv import *
if __name__ == "__main__":
    # Load the dataset using the 'load' function with specified parameters.
    # Parameters:
    # - index_number: Identifier or flag related to image data.
    # - category: Category identifier for data selection or processing.
    # - xloc: Path to the directory containing priority data for training.
    # Note: Ensure that 'index_number', 'category', and 'hyperparameter' are defined before this block.
    X, Y, E = load(index_number, category, xloc=f'../../results/xblock/priority/{hyperparameter}/')
    
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
    # - 'survpri': A keyword to categorize the saved results 
    evaluate(nnet, testset, 'survpri')
