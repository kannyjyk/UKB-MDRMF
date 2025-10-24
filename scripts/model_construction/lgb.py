# Import necessary utilities from utils_other module
from utils_other import *

# Define a directory path for saving LightGBM results, formatted with the 'folder' variable
path = f"{path_prefix}/{folder}lgb/"

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
        
        # Load indices for train, validation, and test splits based on image_X
        *_, trainindex, valindex, testindex = loadindex(image_X)
        
        # Initialize a dictionary to store results for each disease
        result = {}

        # Loop through each disease in the label matrix (column-wise, each column represents a disease)
        for i in range(lab.shape[1]):
            # Define additional string for file naming based on image_X value
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

                # Create LightGBM Datasets for training, validation, and test
                train_data = lgb.Dataset(X_train, label=y_train)
                validation_data = lgb.Dataset(X_val, label=y_val)
                test_data = lgb.Dataset(X_test, label=y_test)

                # Set training parameters for LightGBM
                num_round = 200  # Number of boosting rounds
                param = {
                    "objective": "binary",  # Binary classification objective
                    "metric": "auc",        # Use AUC as the evaluation metric
                    "nthread": 100,         # Number of threads for parallel processing
                }

                # Train the LightGBM model with early stopping on validation data
                lgb_model = lgb.train(
                    param,
                    train_data,
                    num_round,
                    valid_sets=[validation_data],  # Evaluate on validation data
                    callbacks=[lgb.early_stopping(stopping_rounds=50)],  # Stop if no improvement for 50 rounds
                )

                # Make predictions on test and training sets using the best iteration
                y_pred = lgb_model.predict(
                    X_test, num_iteration=lgb_model.best_iteration
                )
                train_pred = lgb_model.predict(
                    X_train, num_iteration=lgb_model.best_iteration
                )

                # Try to render the ROC AUC results and store them if successful
                try:
                    # Calculate AUC for test and training predictions
                    res = renderresult(y_test, y_pred, supress=True)
                    train_res = renderresult(y_train, train_pred, supress=True)
                    print(res)  # Print test AUC result
                    
                    # Store results in the dictionary
                    result[i] = [i, "Place_holder (not used anymore)", res, train_res]

                    # Save the result using pickle in the specified directory
                    with open(path + output_name, "wb+") as file:
                        pickle.dump(result[i], file)
                        file.close()
                except:
                    pass  # If any error occurs in rendering or saving, skip to the next loop
