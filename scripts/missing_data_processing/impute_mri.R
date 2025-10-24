library(tidyverse)
load("../../results/Process_missingness/data_temp/input_mf_train_mri.rda")
load("../../results/Process_missingness/data_temp/input_mf_test_mri.rda")

get_mode <- function(x) {
  # Create a frequency table
  freq_table <- table(x)
  
  # Find the mode value(s)
  mode_value <- names(freq_table)[freq_table == max(freq_table)]
  
  # In case of ties, select one of the mode values
  if(length(mode_value) > 1) mode_value <- mode_value[1]
  
  return(mode_value)
}
impute_mean_mode <- function(input_df) {
  
  # Loop through each column of the dataframe
  for (col_name in names(input_df)) {
    
    # Check if the column is a factor
    if (is.factor(input_df[[col_name]])) {
      
      # Calculate the mode for factor columns
      mode_val <- get_mode(na.omit(input_df[[col_name]]))
      
      # Impute missing values with mode
      input_df[[col_name]][is.na(input_df[[col_name]])] <- mode_val
      
    } else {
      
      # Calculate the mean for numeric columns
      mean_val <- mean(input_df[[col_name]], na.rm = TRUE)
      
      # Impute missing values with mean
      input_df[[col_name]][is.na(input_df[[col_name]])] <- mean_val
    }
  }
  
  return(input_df)
}

predict.missRanger <- function(x, newdata, data_train, n_iter = 5) {
  to_fill <- is.na(newdata[x$visit_seq])
  to_fill_train <- is.na(data_train)
  
  newdata[, setdiff(names(newdata), x$visit_seq)] <- impute_mean_mode(newdata[, setdiff(names(newdata), x$visit_seq)])
  
  # Initialize by mean and mode
  for (v in x$visit_seq) {
    # m <- sum(to_fill[, v])
    if (is.factor(data_train[[v]])) {
      newdata[[v]][to_fill[, v]] <- get_mode(na.omit(data_train[[v]]))
    } else {
      newdata[[v]][to_fill[, v]] <- mean(data_train[[v]], na.rm = TRUE)
    }
  }
  
  for (i in seq_len(n_iter)) {
    for (v in x$visit_seq) {
      m <- sum(to_fill[, v])
      if (m > 0) {
        v_na <- to_fill[, v]
        pred <- predict(x$forests[[v]], newdata[v_na, ])$predictions
        newdata[v_na, v] <- pred
      }
    }
  }
  newdata
}

Sys.time()

# missforest
library(missRanger)
obj_imputed <- missRanger(data = input_mf_train_mri, num.trees = 10, min.node.size = 5000, max.depth = 6, 
                          maxiter = 5, verbose = 2, data_only = FALSE, keep_forests = TRUE)

imputed_train_mri <- obj_imputed$data
save(imputed_train_mri, file = "../../results/Process_missingness/data_temp/imputed_train_mri.rda")
save(obj_imputed, file = "../../results/Process_missingness/data_temp/obj_imputed.rda")
imputed_test_mri <- predict.missRanger(obj_imputed, input_mf_test_mri, input_mf_train_mri)
save(imputed_test_mri, file = "../../results/Process_missingness/data_temp/imputed_test_mri.rda")

Sys.time()

# post_imputation ---------------------------------------------------------

load("../../results/Process_missingness/data_temp/imputed_train_mri.rda")
load("../../results/Process_missingness/data_temp/imputed_test_mri.rda")
mri <- data.table::fread("../../results/Preprocess/MRI.csv", header = T) %>% as_tibble()

imputed_data <- rbind(imputed_train_mri, imputed_test_mri)
df_imp <- imputed_data[order(as.integer(rownames(imputed_data))), ]
mri_imputed <- df_imp[, 1:ncol(mri)]

load("../../results/Process_missingness/data_temp/names_mri.rda")
colnames(mri_imputed) <- names_mri
load("../../results/Process_missingness/data_temp/mri_indicator.rda")
mri_imputed <- cbind(mri_imputed, mri_indicator)
write_csv(mri_imputed, file = "../../results/Process_missingness/mri_imputed.csv")

