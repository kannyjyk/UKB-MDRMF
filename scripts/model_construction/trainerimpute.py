from trainer import *  # Import all functions, classes, and variables from the 'trainer' module

if __name__ == "__main__":
    # Load the dataset using the `load` function.
    # Parameters:
    # - `index_number`: A flag indicating the type of image data to load.
    # - `category`: Specifies the data category.
    # - `fullimgspec=hyperparameter`: Selects the imputation method based on the `hyperparameter` index.
    # Returns:
    # - `Xdata`: Feature data array.
    # - `_`: An unused value returned by `load`.
    # - `lab`: Label data array.
    Xdata, _, lab = load(index_number, category, fullimgspec=hyperparameter)

    # Generate datasets and DataLoaders using the `dataset_generation` function.
    # Parameters:
    # - `Xdata`: The feature data array loaded previously.
    # - `lab`: The label data array loaded previously.
    # - `index_number`: The same flag used in the `load` function.
    # Returns:
    # - `numbers`: List of sample indices.
    # - `trainset`: Training dataset.
    # - `valset`: Validation dataset.
    # - `testset`: Testing dataset.
    # - `train_loader`: DataLoader for the training dataset.
    # - `val_loader`: DataLoader for the validation dataset.
    # - `shape_data`: Number of features in the data.
    # - `shape_label`: Number of features in the labels.
    (
        numbers,
        trainset,
        valset,
        testset,
        train_loader,
        val_loader,
        shape_data,
        shape_label,
    ) = dataset_generation(Xdata, lab, index_number)

    # Select and initialize the appropriate model using the `model_selection` function.
    # Parameters:
    # - `model`: An integer flag to select the model type.
    # - `shape_data`: Number of features in the input data.
    # - `shape_label`: Number of features in the label data.
    # - `3`: Specifies the hyperparameter index directly (overriding the `hyperparameter` variable).
    # Returns:
    # - `net`: The initialized model instance.
    net = model_selection(model, shape_data, shape_label, 3)

    # Train the selected model using the `train` function.
    # Parameters:
    # - `net`: The model instance to be trained.
    # - `[train_loader, val_loader]`: A list containing the DataLoaders for training and validation datasets.
    # Returns:
    # - `nnnet`: The trained model instance.
    nnnet = train(net, [train_loader, val_loader])

    # Evaluate the trained model on the test dataset using the `evaluate` function.
    # Parameters:
    # - `nnnet`: The trained neural network model to evaluate.
    # - `testset`: The testing dataset.
    # - `'impute'`: An optional keyword to include in the file paths for saving results.
    # Returns:
    # - None
    evaluate(nnnet, testset, "impute")
