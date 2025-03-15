library(tidyverse)
library(data.table)

# load index --------------------------------------------------------------
ind_mri <- fread("../../results/Preprocess/image_index.csv") %>% as_tibble() %>% pull(image) + 1
ind_train <- fread("../../results/Preprocess/image_train_index.csv") %>% as_tibble() %>% pull(train) + 1
ind_val <- fread("../../results/Preprocess/image_val_index.csv") %>% as_tibble() %>% pull(val) + 1
ind_test <- fread("../../results/Preprocess/image_test_index.csv") %>% as_tibble() %>% pull(test) + 1

ind_test <- union(ind_val, ind_test)


# load data ---------------------------------------------------------------
mri <- fread("../../results/Preprocess/MRI.csv", header = T) %>% as_tibble()
names_mri <- colnames(mri)
save(names_mri, file = "../../results/Process_missingness/data_temp/names_mri.rda")

# input_mf
load("../../results/Process_missingness/data_temp/input_mf.rda")
data_main <- input_mf[ind_mri, ]
input_mf_mri <- cbind(mri, data_main)


# split -------------------------------------------------------------------
input_mf_mri <- as.data.frame(input_mf_mri)
colnames(input_mf_mri) <- paste0("V", 1:ncol(input_mf_mri))
input_mf_train_mri <- input_mf_mri[ind_train, ]
input_mf_test_mri <- input_mf_mri[ind_test, ]
save(file = "../../results/Process_missingness/data_temp/input_mf_train_mri.rda", input_mf_train_mri)
save(file = "../../results/Process_missingness/data_temp/input_mf_test_mri.rda", input_mf_test_mri)



# mim ---------------------------------------------------------------------

add_mim <- function(input_df) {
  
  # Loop through each column of the dataframe
  for (col_name in names(input_df)) {
    
    # Check if the column has any missing values
    if (any(is.na(input_df[[col_name]]))) {
      
      # Create a missing indicator column
      indicator_name <- paste0(col_name, "#indicator")
      input_df[[indicator_name]] <- as.integer(is.na(input_df[[col_name]]))
    }
  }
  
  return(input_df)
}

mri_argumented <- add_mim(mri)
mri_indicator <- mri_argumented %>%
  select(contains("#indicator"))

save(mri_indicator, file = "../../results/Process_missingness/data_temp/mri_indicator.rda")
