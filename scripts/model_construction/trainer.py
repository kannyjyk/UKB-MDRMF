from utils import *
from config import *

np.random.seed(0)
torch.manual_seed(0)
parser = argparse.ArgumentParser(description="parse")
parser.add_argument("category", type=int)
parser.add_argument("model", type=int)
parser.add_argument("hyperparameter", type=int)
parser.add_argument("index_number", type=int)
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
index_number = args.index_number
loc = args.string

learning_rate = 0.0001
weight_decay = 0





def model_selection(model, shape_data, shape_label, hyperparameter):
    """
    Selects and initializes a machine learning model based on the provided parameters.
    
    Parameters:
        model (int): An integer flag to select the model type.
                     - 0: POPDxModelC1 with label embeddings from 'phe.npy'
                     - 1: pheNN with various hyperparameterarameters
                     - 2: Logistic Regression
                     - 3: POPDxModelC with label embeddings from 'conv.npy'
        shape_data (int): The number of features in the input data.
        shape_label (int): The number of features in the label data.
        hyperparameter (int): An integer index to select hyperparameterarameters from predefined lists.
    
    Returns:
        net: An instance of the selected model initialized with the appropriate parameters.
    
    Raises:
        Exception: If an invalid model flag is provided.
    """
    
    if model == 0:
        # Load label embeddings from 'phe.npy' file
        label_emb = np.load(f"../../data/Embedding/phe.npy", allow_pickle=True)
        # Convert the numpy array to a PyTorch tensor and move it to the specified device
        label_emb = torch.tensor(label_emb, device=device).float()
        
        # Define a list of possible hidden layer sizes
        parampair = [25, 50, 100, 200, 400, 500, 1000]
        # Select the hidden size based on the hyperparameterarameter index
        hidden_size = parampair[hyperparameter]
        
        # Initialize the POPDxModelC1 with the specified parameters
        net = POPDxModelC1(shape_data, shape_label, hidden_size, label_emb)
    
    elif model == 1:
        # Define a list of tuples representing different hyperparameterarameter pairs (e.g., layers and hidden sizes)
        parampair = [
            (1, 25),
            (1, 50),
            (2, 50),
            (2, 100),
            (3, 50),
            (3, 100),
            (3, 200),
            (5, 500),
            (6, 300),
            (6, 400),
            (6, 500),
            (7, 500),
            (8, 500),
            (10, 1000),
        ]
        # Select the hyperparameterarameter pair based on the hyperparameterarameter index
        param = parampair[hyperparameter]
        
        # Initialize the pheNN model with the selected parameters
        net = pheNN(shape_data, shape_label, param[0], param[1])
    
    elif model == 2:
        # Initialize a Logistic Regression model with the specified number of features
        net = LogisticRegression(shape_data, shape_label)
    
    elif model == 3:
        # Load label embeddings from 'conv.npy' file
        label_emb = np.load(f"../../data/Embedding/conv.npy", allow_pickle=True)
        
        # Define a list of possible hidden layer sizes
        parampair = [25, 50, 100, 200, 400, 500, 1000]
        # Select the hidden size based on the hyperparameterarameter index
        try:
            hidden_size = parampair[hyperparameter]
        except:
            raise Exception("Resouce released, moving on to next task, not an error")
        
        
        # Convert the numpy array to a PyTorch tensor and move it to the specified device
        label_emb = torch.tensor(label_emb, device=device).float()
        
        # Initialize the POPDxModelC with the specified parameters
        net = POPDxModelC(shape_data, shape_label, hidden_size, label_emb)
    
    else:
        # Raise an exception if an invalid model flag is provided
        raise Exception("Invalid model selection. Please choose a valid model index.")
    
    # Return the initialized model
    return net


def evaluate(nnnet, testset, keyword=""):
    """
    Evaluates a neural network model on the test dataset, computes AUC scores for each label,
    saves the results, and stores the trained model.

    Parameters:
        nnnet (torch.nn.Module): The neural network model to evaluate.
        testset (torch.utils.data.Dataset): The test dataset.
        keyword (str, optional): An optional keyword to include in the file paths. Defaults to an empty string.

    Returns:
        None

    Raises:
        Exception: Propagates any exceptions that occur during evaluation or saving.
    """
    
    # Create a DataLoader for the entire test set without shuffling
    whole_loader = DataLoader(testset, batch_size=len(testset))
    
    # Retrieve the first (and only) batch from the DataLoader
    inputs, labels = next(iter(whole_loader))
    
    # Move inputs to the specified device (CPU or GPU) and perform a forward pass through the network
    out = nnnet(inputs.to(device)).cpu().detach().numpy()
    
    # Apply the sigmoid activation function to the outputs to obtain probabilities
    out = torch.sigmoid(torch.from_numpy(out)).numpy()
    
    # Initialize a list to store AUC results for each label
    aucresult = []
    
    # Iterate over each label dimension to compute AUC
    for i in range(labels.shape[1]):
        # Compute AUC for the i-th label using the `renderresult` function
        auc = renderresult(labels.cpu().detach().numpy()[:, i], out[:, i])
        aucresult.append(auc)
    
    # Attempt to create a directory specified by `loc`; ignore the error if it already exists
    try:
        os.mkdir(f"{path_prefix}/{loc}{keyword}")
    except:
        pass  # Directory already exists or another error occurred; continue execution
    
    # Compute and print the mean AUC, ignoring NaN values
    print(np.nanmean(aucresult))
    
    # Save the AUC results to a NumPy file with a specific naming convention
    np.save(
        f"{path_prefix}/{loc}{keyword}/{category}{modelchar(model)}{index_number}_{hyperparameter}", aucresult
    )
    
    # Save the labels to a NumPy file for later reference or analysis
    np.save(f"{path_prefix}/{loc}{keyword}/{index_number}lab", labels)
    
    # Save the trained neural network model to a file for future use or deployment
    torch.save(
        nnnet, f"{path_prefix}/{loc}{keyword}/{category}{modelchar(model)}{index_number}_{hyperparameter}model"
    )
    
    # Indicate that the evaluation and saving process is complete
    print("complete")


if __name__ == "__main__":
    # Load the dataset using the `load` function.
    Xdata, _, lab = load(index_number, category, fullimgspec=0)
    
    # Generate datasets and DataLoaders using the `dataset_generation` function.
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
    net = model_selection(model, shape_data, shape_label, hyperparameter)
    
    # Train the selected model using the `train` function.
    nnnet = train(net, [train_loader, val_loader])
    
    # Evaluate the trained model on the test dataset using the `evaluate` function.
    evaluate(nnnet, testset, "")

