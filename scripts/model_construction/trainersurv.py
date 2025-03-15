from utils_surv import *
from config import *
np.random.seed(0)
torch.manual_seed(0)
parser = argparse.ArgumentParser(description="parse")
parser.add_argument("category", type=int)
parser.add_argument("model", type=int)
parser.add_argument("hyperparameter", type=int)
parser.add_argument("Xtype", type=int)
parser.add_argument("gpu", type=int)
parser.add_argument("string", type=str)
args = parser.parse_args()
gpu = args.gpu
if gpu>=0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
category = args.category
model = args.model
hyperparameter = args.hyperparameter
index_number = args.Xtype
loc = args.string
only = False
waiting = 15
epoch = 1000
learning_rate = 0.005




def model_selection(model, shape_data, shape_label):
    """
    Selects and initializes a survival analysis model based on the provided model flag.

    Parameters:
        model (int): An integer flag to select the model type.
                     - 0: Cox Proportional Hazards Model
                     - 1: DeepSurv Model
                     - 2: POPDx Model
                     - 3: Mith Model
        shape_data (int): Number of features in the input data.
        shape_label (int): Number of features in the label data.

    Returns:
        nnet (torch.nn.Module): The initialized neural network model.

    Raises:
        ValueError: If an invalid model flag is provided.
    """
    if model == 0:  # 'cox'
        # Initialize the Cox Proportional Hazards Model
        nnet = Cox(shape_data, shape_label)

    elif model == 1:  # 'deepsurv'
        # Initialize the DeepSurv Model with specific hyperparameters
        # Parameters:
        # - shape_data: Number of input features
        # - shape_label: Number of output features
        # - 5: Number of layers or another hyperparameter (context needed)
        # - 300: Hidden layer size or another hyperparameter (context needed)
        nnet = DeepSurv(shape_data, shape_label, 5, 300)

    elif model == 2:  # 'popdx'
        # Initialize the POPDx Model with label embeddings
        # Load precomputed label embeddings from a NumPy file
        label_emb = np.load(
            "../../data/Embedding/phe.npy", allow_pickle=True
        )

        hidden_size = 400  # Define the size of hidden layers
        # Initialize the EmbeddingModel with the loaded embeddings and device configuration
        nnet = EmbeddingModel(shape_data, shape_label, hidden_size, label_emb, device)

    elif model == 3:  # 'mith'
        # Initialize the Mith Model with label embeddings
        # Load precomputed label embeddings from a NumPy file
        label_emb = np.load(
            "../../data/Embedding/conv.npy", allow_pickle=True
        )
       
        hidden_size = 400  # Define the size of hidden layers
        # Initialize the EmbeddingModel with the loaded embeddings and device configuration
        nnet = EmbeddingModel(shape_data, shape_label, hidden_size, label_emb, device)

    else:
        # Raise an error if an invalid model flag is provided
        raise ValueError(
            f"Invalid model flag '{model}'. Expected values are 0, 1, 2, or 3."
        )

    return nnet


def evaluate(nnet, testset, keyword="surv"):
    """
    Evaluates the trained neural network model on the test dataset, computes the concordance index (c-index),
    saves the evaluation results, and saves the trained model for future use.

    Parameters:
        nnet (torch.nn.Module): The trained neural network model to evaluate.
        testset (torch.utils.data.Dataset): The testing dataset.
        keyword (str, optional): A keyword to categorize the saved results (e.g., "surv" for survival analysis).
                                 Defaults to "surv".

    Returns:
        None

    Raises:
        Exception: Propagates any exception that occurs during evaluation or saving of results.
    """
    # Create a DataLoader for the entire test dataset with a single batch.
    # This allows processing all test samples at once.
    whole_loader = DataLoader(testset, batch_size=len(testset))

    # Retrieve the first (and only) batch from the DataLoader.
    # 'xtest' contains the feature data, 'ytest' contains the labels, and 'etest' contains event indicators.
    xtest, ytest, etest = next(iter(whole_loader))

    # Move 'ytest' and 'etest' tensors to CPU for processing.
    # Note: These lines do not modify 'ytest' and 'etest' in-place.
    # To move tensors in-place, you should assign the result back.
    ytest = ytest.cpu()
    etest = etest.cpu()

    # Move 'xtest' to compute device for model inference, perform forward pass,
    # move the output back to CPU, detach it from the computation graph, and convert to NumPy array.
    # Ensure that 'nnet' is already on the appropriate device or move it as needed.
    out = nnet(xtest.to(device)).cpu().detach().numpy()

    # Initialize an empty list to store c-index values for each output dimension.
    cindex = []

    # Iterate over each output dimension (assumed to be the second dimension of 'out').
    for j in range(out.shape[1]):
        try:
            # Identify indices where the true labels are NaN for the current output dimension.
            na_indices = np.where(np.isnan(ytest[:, j]))[0]

            # Remove NaN entries from the predicted output, true labels, and event indicators.
            o = np.delete(-out[:, j], na_indices)  # Negate 'out' if needed based on model output
            y = np.delete(ytest[:, j], na_indices)
            e = np.delete(etest[:, j], na_indices)

            # Compute the concordance index (c-index) for the current output dimension.
            # 'c_index' is assumed to be a predefined function that computes the c-index.
            cindex.append(c_index(o, y, e))
        except Exception as e:
            # If any exception occurs (e.g., due to insufficient data), append NaN to indicate failure.
            cindex.append(np.nan)
    # Attempt to create a directory named "./{loc}{keyword}" to save evaluation results.
    try:
        os.mkdir(f"{path_prefix}/{loc}{keyword}")
    except:
        # If the directory already exists, ignore the error.
        pass

    # Save the c-index results as a NumPy array.
    # The filename includes 'category', a character representation of the model ('modelchar(model)'),
    # 'index_number', and 'hyperparameter' for identification.
    # These variables are assumed to be predefined global variables or accessible in the scope.
    np.save(f"{path_prefix}/{loc}{keyword}/{category}{modelchar(model)}{index_number}_{hyperparameter}", cindex)

    # Save the true labels and event indicators as a NumPy array for future reference or analysis.
    np.save(f"{path_prefix}/{loc}{keyword}/{index_number}lab", [ytest, etest])

    # Save the trained neural network model to disk for future use or deployment.
    torch.save(
        nnet, f"{path_prefix}/{loc}{keyword}/{category}{modelchar(model)}{index_number}_{hyperparameter}model"
    )

    # Print a completion message to indicate that evaluation is finished.
    print("Evaluation complete.")



if __name__ == "__main__":
    # Load the dataset using the 'load' function with specified parameters.
    # Parameters:
    # - index_number: Identifier or flag related to image data.
    # - category: Category identifier for data selection or processing.
    # - only: Additional flag or parameter for data loading.
    X, Y, E = load(index_number, category=category, only=only)
    
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
    
    # Move the trained model to the computation device for evaluation.
    nnet.to(device)
    
    # Evaluate the trained model using the 'evaluate' function.
    # Parameters:
    # - nnet: The trained neural network model.
    # - testset: The testing dataset.
    evaluate(nnet, testset)