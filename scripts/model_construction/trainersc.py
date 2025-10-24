from trainer import *  # Import all functions, classes, and variables from the 'trainer' module

if __name__ == "__main__":
    # Assign the hyperparameter index from command-line arguments or other configurations
    # `args` should be defined and parsed earlier in the script using argparse or a similar library
    imgimp = args.hyperparameter

    # Load the dataset using the `load` function.
    # Parameters:
    # - `index_number`: A flag indicating the type of image data to load.
    # - `category`: Specifies the data category.
    # - `fullimgspec=imgimp`: Selects the imputation method based on the `imgimp` index.
    # Returns:
    # - `Xdata`: Feature data array.
    # - `_`: An unused value returned by `load`.
    # - `lab`: Label data array.
    Xdata, _, lab = load(index_number, category, fullimgspec=imgimp)

    # Load the category-to-label mapping from a NumPy file.
    # `phecat.npy` is expected to contain a dictionary mapping category indices to label indices.
    # `allow_pickle=True` allows loading objects saved in the NumPy file.
    # `[0]` accesses the first element if the NumPy file contains a list or similar iterable.
    cats = np.load("../../data/phecat.npy", allow_pickle=True)[0]

    # Select the appropriate key from the `cats` dictionary based on the `model` index.
    # `model` should be defined and represent the current model selection.
    key = list(cats.keys())[model]

    # Update the `lab` (labels) array to include only the columns corresponding to the selected category.
    # `cats[key]` provides the list of label indices associated with the selected category.
    lab = lab[:, cats[key]]

    # Generate datasets and DataLoaders using the `dataset_generation` function.
    # Parameters:
    # - `Xdata`: The feature data array loaded previously.
    # - `lab`: The label data array loaded and filtered based on the category.
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
    # - `3`: Specifies the hyperparameter index directly (overriding the `imgimp` variable).
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
    # - `'pri'`: An optional keyword to include in the file paths for saving results (e.g., "pri" for priority).
    # Returns:
    # - None
    evaluate(nnnet, testset, "pri")
