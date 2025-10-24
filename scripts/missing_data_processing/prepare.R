library(tidyverse)
ind_train <- read_csv("../../results/Preprocess/train_index.csv") %>% pull(train) + 1
ind_val <- read_csv("../../results/Preprocess/val_index.csv") %>% pull(val) + 1
ind_test <- read_csv("../../results/Preprocess/test_index.csv") %>% pull(test) + 1
ind_test <- union(ind_val, ind_test)
load("../../results/Process_missingness/data_temp/input_mf.rda")

input_mf <- as.data.frame(input_mf)
colnames(input_mf) <- paste0("V", 1:ncol(input_mf))
input_mf_train <- input_mf[ind_train, ]
input_mf_test <- input_mf[ind_test, ]
save(file = "../../results/Process_missingness/data_temp/input_mf_train.rda", input_mf_train)
save(file = "../../results/Process_missingness/data_temp/input_mf_test.rda", input_mf_test)

