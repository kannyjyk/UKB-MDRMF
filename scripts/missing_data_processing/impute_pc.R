library(tidyverse)
library(data.table)
ind_train <- read_csv("../../results/Preprocess/train_index.csv") %>% pull(train) + 1
ind_val <- read_csv("../../results/Preprocess/val_index.csv") %>% pull(val) + 1
ind_test <- read_csv("../../results/Preprocess/test_index.csv") %>% pull(test) + 1
ind_test <- union(ind_val, ind_test)

ind_mri <- fread("../../results/Preprocess/image_index.csv") %>% as_tibble() %>% pull(image) + 1

load("../../results/Process_missingness/data_temp/df_imp.rda")
pcs <- fread("../../results/Process_missingness/MRIPC.csv")
names_pc <- colnames(pcs)
save(names_pc, file = "../../results/Process_missingness/data_temp/names_pc.rda")

pcs_full <- as.data.frame(matrix(NA, nrow = nrow(df_imp), ncol = ncol(pcs)))
pcs_full[ind_mri,] <- pcs

impute_zero <- function(input_df, ind_train) {
  
  # Loop through each column of the dataframe
  for (col_name in names(input_df)) {
    
    # Impute missing values with 0
    input_df[[col_name]][is.na(input_df[[col_name]])] <- 0
  }
  
  return(input_df)
}

pcs_imp <- impute_zero(pcs_full, ind_train)
colnames(pcs_imp) <- names_pc
PC_indicator <- as.numeric((1:nrow(df_imp)) %in% ind_mri) 
pc_imputed <- cbind(pcs_imp, PC_indicator)
write_csv(pc_imputed, file = "../../results/Process_missingness/mripc_imputed.csv")
