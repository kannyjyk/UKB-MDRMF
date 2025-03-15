from trainersurv import *

if __name__ == "__main__":
    # Load the dataset using the 'load' function with specified parameters.
    # Parameters:
    # - index_number: Identifier or flag related to image data.
    # - category: Category identifier for data selection or processing.
    # - only: Additional flag or parameter for data loading.
    # Note: Ensure that 'index_number', 'category', and 'only' are defined before this block.
    X, Y, E = load(index_number, category=category, only=only)
    
    # Load category mappings from a NumPy file.
    # 'allow_pickle=True' allows loading Python objects saved in the file.
    # The '[0]' index suggests that the file contains a list or array with the dictionary as the first element.
    cats = np.load('../../data/phecat.npy', allow_pickle=True)[0]
    
    # Retrieve the key corresponding to the current model.
    # 'model' is an integer flag that selects which category to use.
    # 'list(cats.keys())' converts the dictionary keys to a list, and '[model]' selects the key at the index 'model'.
    key = list(cats.keys())[model]
    
    # Subset the labels 'Y' and event indicators 'E' based on the selected category.
    # 'cats[key]' provides a list or array of indices to select relevant columns from 'Y' and 'E'.
    Y = Y[:, cats[key]]
    E = E[:, cats[key]]
    
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
    # - 'survsc': A keyword to categorize the saved results
    evaluate(nnet, testset, 'survsc')
