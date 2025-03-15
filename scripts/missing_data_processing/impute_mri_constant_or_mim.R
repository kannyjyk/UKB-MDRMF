library(tidyverse)

method <- "mean_mode"
# method <- "median_mode"
# method <- "zero_mode"
# method <- "mim"

mri <- fread("../../results/Preprocess/MRI.csv", header = T) %>% as_tibble()

ind_train <- fread("../../results/Preprocess/image_train_index.csv") %>% as_tibble() %>% pull(train) + 1
ind_val <- fread("../../results/Preprocess/image_val_index.csv") %>% as_tibble() %>% pull(val) + 1
ind_test <- fread("../../results/Preprocess/image_test_index.csv") %>% as_tibble() %>% pull(test) + 1
ind_test <- union(ind_val, ind_test)

get_mode <- function(x) {
  # Create a frequency table
  freq_table <- table(x)
  
  # Find the mode value(s)
  mode_value <- names(freq_table)[freq_table == max(freq_table)]
  
  # In case of ties, select one of the mode values
  if(length(mode_value) > 1) mode_value <- mode_value[1]
  
  return(mode_value)
}
impute_mean_mode <- function(input_df, ind_train) {
  
  # Loop through each column of the dataframe
  for (col_name in names(input_df)) {
    
    # Check if the column is a factor
    if (is.factor(input_df[[col_name]])) {
      
      # Calculate the mode for factor columns
      mode_val <- get_mode(na.omit(input_df[[col_name]][ind_train]))
      
      # Impute missing values with mode
      input_df[[col_name]][is.na(input_df[[col_name]])] <- mode_val
      
    } else {
      
      # Calculate the mean for numeric columns
      mean_val <- mean(input_df[[col_name]][ind_train], na.rm = TRUE)
      
      # Impute missing values with mean
      input_df[[col_name]][is.na(input_df[[col_name]])] <- mean_val
    }
  }
  
  return(input_df)
}


# mim
impute_mim <- function(input_df) {
  
  # Loop through each column of the dataframe
  for (col_name in names(input_df)) {
    
    # Check if the column has any missing values
    if (any(is.na(input_df[[col_name]]))) {
      
      # Check if the column is a factor
      if (is.factor(input_df[[col_name]])) {
        
        # Add a new level "missing" to the factor column
        levels(input_df[[col_name]]) <- c(levels(input_df[[col_name]]), "indicator")
        
        # Replace NA with the new level "missing"
        input_df[[col_name]][is.na(input_df[[col_name]])] <- "indicator"
        
      } else {
        
        # Create a missing indicator column
        indicator_name <- paste0(col_name, "#indicator")
        input_df[[indicator_name]] <- as.integer(is.na(input_df[[col_name]]))
        
        # Impute missing values with 0
        input_df[[col_name]][is.na(input_df[[col_name]])] <- 0
      }
    }
  }
  
  return(input_df)
}

impute_median_mode <- function(input_df, ind_train) {
  
  # Loop through each column of the dataframe
  for (col_name in names(input_df)) {
    
    # Check if the column is a factor
    if (is.factor(input_df[[col_name]])) {
      
      # Calculate the mode for factor columns
      mode_val <- get_mode(na.omit(input_df[[col_name]][ind_train]))
      
      # Impute missing values with mode
      input_df[[col_name]][is.na(input_df[[col_name]])] <- mode_val
      
    } else {
      
      # Calculate the mean for numeric columns
      mean_val <- median(input_df[[col_name]][ind_train], na.rm = TRUE)
      
      # Impute missing values with mean
      input_df[[col_name]][is.na(input_df[[col_name]])] <- mean_val
    }
  }
  
  return(input_df)
}

impute_zero_mode <- function(input_df) {
  
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
      mean_val <- 0
      
      # Impute missing values with mean
      input_df[[col_name]][is.na(input_df[[col_name]])] <- mean_val
    }
  }
  
  return(input_df)
}

# imputation --------------------------------------------------------------

if (method == "mean_mode") {
  mri_imputed <- impute_mean_mode(mri, ind_train)
}

if (method == "median_mode") {
  mri_imputed <- impute_median_mode(mri, ind_train)
}

if (method == "zero_mode") {
  mri_imputed <- impute_zero_mode(mri)
}

if (method == "mim") {
  mri_imputed <- impute_mim(mri)
}

write_csv(mri_imputed, file = "../../results/Process_missingness/mri_imputed.csv")




