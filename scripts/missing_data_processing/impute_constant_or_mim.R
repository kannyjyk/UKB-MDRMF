library(tidyverse)

method <- "mean_mode"
# method <- "median_mode"
# method <- "zero_mode"
# method <- "mim"

load("../../results/Process_missingness/data_temp/input_mf_train.rda")
load("../../results/Process_missingness/data_temp/input_mf_test.rda")

ind_train <- read_csv("../../results/Preprocess/train_index.csv") %>% pull(train) + 1
ind_val <- read_csv("../../results/Preprocess/val_index.csv") %>% pull(val) + 1
ind_test <- read_csv("../../results/Preprocess/test_index.csv") %>% pull(test) + 1
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

predict.missRanger <- function(x, newdata, data_train, n_iter = 5) {
  to_fill <- is.na(newdata[x$visit_seq])
  to_fill_train <- is.na(data_train)
  
  newdata[, setdiff(names(newdata), x$visit_seq)] <- impute_mean_mode(newdata[, setdiff(names(newdata), x$visit_seq)], ind_train = 1:nrow(newdata))
  
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
if (method == "missforest") {
  library(missRanger)
  
  obj_imputed <- missRanger(data = input_mf_train, num.trees = 10, min.node.size = 5000, max.depth = 6, 
                            maxiter = 5, verbose = 2, data_only = FALSE, keep_forests = TRUE)
  
  imputed_train <- obj_imputed$data
  save(imputed_train, file = "../../results/Process_missingness/data_temp/imputed_train.rda")
  save(obj_imputed, file = "../../results/Process_missingness/data_temp/obj_imputed.rda")
  imputed_test <- predict.missRanger(obj_imputed, input_mf_test, input_mf_train)
  save(imputed_test, file = "../../results/Process_missingness/data_temp/imputed_test.rda")
}

if (method == "mean_mode") {
  load("../../results/Process_missingness/data_temp/input_mf.rda")
  imputed_data <- impute_mean_mode(input_mf, ind_train)
}

if (method == "median_mode") {
  load("../../results/Process_missingness/data_temp/input_mf.rda")
  imputed_data <- impute_median_mode(input_mf, ind_train)
}

if (method == "zero_mode") {
  load("../../results/Process_missingness/data_temp/input_mf.rda")
  imputed_data <- impute_zero_mode(input_mf)
}

if (method == "mim") {
  load("../../results/Process_missingness/data_temp/input_mf.rda")
  imputed_data <- impute_mim(input_mf)
}

# path <- paste0("data_output/imputed_", method, "/") 
# if(!file.exists(path)) {dir.create(path, recursive = TRUE)}
# save(imputed_data, file = paste0(path, "imputed_data.rda"))

# post_imputation ---------------------------------------------------------

if (method != "missforest") {
  df_imp <- imputed_data
}

if (method == "missforest"){
  load("../../results/Process_missingness/data_temp/imputed_train.rda")
  load("../../results/Process_missingness/data_temp/imputed_test.rda")
  
  imputed_data <- rbind(imputed_train, imputed_test)
  df_imp <- imputed_data[order(as.integer(rownames(imputed_data))), ]
  
  load("../../results/Process_missingness/data_temp/input_mf.rda")
  colnames(df_imp) <- colnames(input_mf)
}

save(df_imp, file = "../../results/Process_missingness/data_temp/df_imp.rda")

# adjustment and rename ---------------------------------------------------

one_hot_encode <- function(df) {
  factor_cols <- names(df)[sapply(df, is.factor)]
  
  for (col in factor_cols) {
    # Get the one-hot encoding for the factor column
    encoding <- model.matrix(~ 0 + ., data = df[col])
    
    # Rename the columns to the desired format: "original name#factor level"
    col_names <- colnames(encoding)
    new_col_names <- sub("^.*\\.", "", col_names)      # remove the intercept term "0"
    new_col_names <- gsub(col, paste0(col, "#"), new_col_names)   # add '#' between column name and level name
    
    colnames(encoding) <- new_col_names
    
    # Add the one-hot encoded columns to the dataframe and remove the original factor column
    df <- bind_cols(df, as_tibble(encoding))
    df[[col]] <- NULL
  }
  
  return(df)
}

df_one_hot <- one_hot_encode(df_imp)

rename_columns <- function(df) {
  new_names <- gsub("`", "", names(df))
  names(df) <- new_names
  df
}

df_imputed <- rename_columns(df_one_hot)

eid_all <- read_csv("../../results/Process_missingness/data_temp/eid_all.csv")
# df_imputed %>%
#   mutate(eid = eid_all$eid, .before = `31`) %>%
#   write_csv(file = "/data/home/tangbr/UKB/missing_pipeline/code_pipeline/data/imputed_all.csv")

load("../../results/Preprocess/discard.RData")
discardlist[["31"]] <- "0"
discardlist[["1210"]] <- "2"
discard_vec <- sapply(names(discardlist), function(x) paste0(x, "#", discardlist[[x]]))
na_string_columns <- names(df_imputed)[grep("NA", names(df_imputed))]
discard_vec <- union(discard_vec, na_string_columns)

# split -------------------------------------------------------------------

imputed_all <- df_imputed %>%
  select(-all_of(discard_vec)) %>%
  mutate(eid = eid_all$eid) %>%
  select(eid, everything())
showcase <- read_csv("../../results/Preprocess/showcase.csv")

fields_baseline <- showcase %>%
  filter(Class_top == "Baseline characteristics") %>%
  pull(FieldID) %>%
  as.character()

fields_lifestyle <- showcase %>%
  filter(Class_top == "Life") %>%
  pull(FieldID) %>%
  as.character()

fields_measurement <- showcase %>%
  filter(Class_top == "Measures") %>%
  pull(FieldID) %>%
  as.character()

fields_environment <- showcase %>%
  filter(Class_top == "Natural and social environment") %>%
  pull(FieldID) %>%
  as.character()

fields_genetic <- showcase %>%
  filter(Class_top == "Genetic") %>%
  pull(FieldID) %>%
  as.character()


names <- sapply(strsplit(colnames(imputed_all), "[#, -]"), function(ls) ls[1])
inds_prs <- grep("PRSPC", names)

inds_baseline <- which(names %in% fields_baseline)
inds_lifestyle <- which(names %in% fields_lifestyle)
inds_measurement <- which(names %in% fields_measurement)
inds_environment <- which(names %in% fields_environment)
inds_genetic <- which(names %in% fields_genetic)

baseline_imputed <- imputed_all[, c(1, inds_baseline)]
lifestyle_imputed <- imputed_all[, c(1, inds_lifestyle)]
measurement_imputed <- imputed_all[, c(1, inds_measurement)]
environment_imputed <- imputed_all[, c(1, inds_environment)]
genetic_imputed <- imputed_all[, c(1, inds_genetic, inds_prs)]

path <- paste0("../../results/Process_missingness/")
if(!file.exists(path)) {dir.create(path, recursive = TRUE)}

write_csv(baseline_imputed, file = paste0(path, "baseline_imputed.csv"))
write_csv(lifestyle_imputed, file = paste0(path, "lifestyle_imputed.csv"))
write_csv(measurement_imputed, file = paste0(path, "measurement_imputed.csv"))
write_csv(environment_imputed, file = paste0(path, "environment_imputed.csv"))
write_csv(genetic_imputed, file = paste0(path, "genetic_imputed.csv"))
