# Import necessary utilities from utils_other module and the GPU-accelerated RandomForestClassifier
from utils_other import *
from cuml.ensemble import RandomForestClassifier

# Set the GPU device to be used for processing
os.environ["CUDA_VISIBLE_DEVICES"] = str(7)

# Define a directory path for saving RandomForest results, formatted with the 'folder' variable
path = f"{path_prefix}/{folder}rf/"

# Attempt to create the directory for results; if it exists, ignore the error
try:
    os.mkdir(path)
except:
    pass

# Loop through each combination of image data (image_X) and category
for image_X in imglist:
    for category in catlist:
        # Load feature data (Xdata) and labels (lab) for the specified image and category
        Xdata, _, lab = load(image_X, category)

        # Initialize a dictionary to store results for each disease
        result = {}

        # Loop through each disease in the label matrix (column-wise, each column represents a disease)
        for i in range(lab.shape[1]):
            # Define additional string for file naming based on the image_X value
            addstr = "" if image_X == 0 else "_5w"

            # Construct the output filename using disease index, category, and additional string
            output_name = str(i) + "_" + str(category) + addstr

            # Check if this output file already exists to avoid redundant processing
            if output_name not in os.listdir(path):
                print((str(i), str(category), str(image_X)))  # Print current processing state

                # Split data for the single disease category into train, validation, and test sets
                y_train, X_train, y_val, X_val, y_test, X_test = single_disease_data(
                    lab, Xdata, i, image_X
                )

                # Try to initialize, train, and evaluate the RandomForest model, handling errors gracefully
                try:
                    # Initialize RandomForestClassifier with GPU support
                    model_rf = RandomForestClassifier(
                        n_estimators=100,  # Number of trees in the forest
                    )

                    # Train the model on training data (ensure input is float32 for compatibility)
                    model_rf.fit(
                        np.array(X_train, dtype=np.float32),
                        np.array(y_train, dtype=np.float32),
                    )

                    # Generate probability predictions for test and training sets
                    y_pred = model_rf.predict_proba(X_test)[:, 1]
                    train_pred = model_rf.predict_proba(X_train)[:, 1]

                    # Calculate ROC AUC for test and training predictions
                    res = renderresult(y_test, y_pred, supress=True)
                    train_res = renderresult(y_train, train_pred, supress=True)

                    # Store results in the dictionary
                    result[i] = [i, "Place_holder (not used anymore)", res, train_res]

                    # Save the result using pickle in the specified directory
                    with open(path + output_name, "wb+") as file:
                        pickle.dump(result[i], file)
                        file.close()
                except:
                    pass  # Skip to the next if any error occurs during processing
