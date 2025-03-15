# Import necessary utilities from utils_other module
from utils_other import *

# Define a directory path for saving results, formatted using the 'folder' variable
path = f"{path_prefix}/{folder}xgb/"

# Attempt to create the directory; if it already exists, ignore the error
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

        # Loop through each disease in the label matrix (each column corresponds to a disease)
        for i in range(lab.shape[1]):
            # Define additional string for file naming based on image_X value
            addstr = "" if image_X == 0 else "_5w"

            # Construct the output filename using disease index, category, and additional string
            output_name = str(i) + "_" + str(category) + addstr

            # Check if this output file already exists in the directory to avoid redundant processing
            if output_name not in os.listdir(path):
                print(
                    (str(i), str(category), str(image_X))
                )  # Print current processing state

                # Split data for the single disease category into train, validation, and test sets
                y_train, X_train, y_val, X_val, y_test, X_test = single_disease_data(
                    lab, Xdata, i, image_X
                )

                # Create XGBoost DMatrix objects for train, validation, and test data
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)
                dtest = xgb.DMatrix(X_test, label=y_test)

                # Set parameters for XGBoost model training
                num_round = 200  # Number of boosting rounds
                param = {
                    "objective": "binary:logistic",  # Binary classification objective
                    "eval_metric": "auc",  # AUC as evaluation metric
                    "nthread": 10,  # Number of parallel threads
                    "tree_method": "gpu_hist",  # Use GPU for training
                    "gpu_id": 2,  # Specify GPU device ID
                }

                # Train the XGBoost model with early stopping on validation data
                xgb_model = xgb.train(
                    param,
                    dtrain,
                    num_round,
                    evals=[(dval, "eval")],
                    early_stopping_rounds=50,  # Stop if no improvement over 50 rounds
                )

                # Make predictions on test and training sets
                y_pred = xgb_model.predict(dtest)
                train_pred = xgb_model.predict(dtrain)

                # Try to render the ROC AUC results and store them if successful
                try:
                    # Calculate AUC for test and training predictions
                    res = renderresult(y_test, y_pred, supress=True)
                    trainres = renderresult(y_train, train_pred, supress=True)

                    # Store the results for the current disease in the result dictionary
                    result[i] = [i, "Place_holder (not used anymore)", res, trainres]

                    # Save the result using pickle in the specified directory
                    with open(path + output_name, "wb+") as file:
                        pickle.dump(result[i], file)
                        file.close()
                except:
                    pass  # If any error occurs in rendering or saving, skip to the next loop
