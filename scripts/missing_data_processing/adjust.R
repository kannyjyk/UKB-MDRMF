library(tidyverse)
library(data.table)

# helpful functions -------------------------------------------------------
convert_one_hot_to_factor <- function(df, numbers) {
  # Add a row id for joining purposes
  df <- df %>% mutate(row_id = row_number())
  
  for (num in numbers) {
    # Extract columns matching the pattern
    cols <- grep(paste0("^", num, "#"), colnames(df), value = TRUE)
    
    # Gather, filter, rename, and convert to factor
    df_temp <- df %>%
      select(row_id, all_of(cols)) %>%
      gather(key = "key", value = "value", -row_id) %>%
      filter(value == 1) %>%
      transmute(row_id, !!as.character(num) := factor(gsub(paste0(num, "#"), "", key)))
    
    # Join back to the original dataframe and remove one-hot encoded columns
    df <- df %>%
      left_join(df_temp, by = "row_id") %>%
      select(-all_of(cols))
  }
  
  # Remove row_id before returning the result
  df <- df %>% select(-row_id)
  
  return(df)
}

# load data ---------------------------------------------------------------
data <- fread("../../results/Preprocess/phenofile_final.csv") %>% as_tibble()
phenofile_1 <- fread("../../results/Preprocess/phenofile_1.csv") 

# extract single categorical vars -----------------------------------------

log <- read_lines("../../results/Preprocess/log.txt")
showcase <- read_csv("../../data/Preprocess/showcase.csv")

# Filter lines containing "Binary", split by space and extract the second element
extracted_elements <- log %>%
  # Filter lines that contain "Binary"
  keep(str_detect(., "BINARY")) %>%
  # Split by space and extract the second element
  map_chr(~ str_split(., " ", simplify = TRUE)[4])

binary_tibble <- tibble(extracted = extracted_elements) %>%
  mutate(extracted = as.numeric(extracted))

fields_single_cat <- (showcase %>%
  filter(FieldID %in% binary_tibble$extracted) %>%
  filter(ValueType == "Categorical single"))$FieldID

# convert to factor -------------------------------------------------------
data_factor <- convert_one_hot_to_factor(data, fields_single_cat)
data_factor <- data_factor %>%
  mutate(across(where(is.factor), 
                ~ factor(ifelse(. == "NA", NA, as.character(.)))
  ))

# adjustments -------------------------------------------------------------
data_argumented <- data_factor %>% inner_join(phenofile_1, by = "eid")

# data_factor_processed <- data_factor
ind_train <- read_csv("../../results/Preprocess/train_index.csv") %>% pull(train) + 1
param <- read_csv("../../data/Process_missingness/adjust_manual_param.csv")
param <- param %>%
  mutate(Discrete = as.integer(Discrete),
         Field = as.character(Field),
         Exp = as.character(Exp))

for (i in 1:nrow(param)) {
  param_single <- param[i, ]
  list2env(param_single, envir = .GlobalEnv)
  
  if (Field %in% colnames(data_factor)) {
    if (Discrete == 0) {
      var_mutate <- Field
      data_argumented <- data_argumented %>%
        mutate(!!var_mutate := case_when(eval(parse(text = Exp)) ~ mean(data_factor[[var_mutate]][ind_train], na.rm = T),
                                         .default = !!sym(var_mutate))) 
    }
    
    if (Discrete == 1) {
      var_mutate <- Field
      data_argumented <- data_argumented %>%
        mutate(!!var_mutate := factor(case_when(eval(parse(text = Exp)) ~ "NA",
                                                .default = !!sym(var_mutate)),
                                      levels = c(levels(!!sym(var_mutate)), "NA")))
    } 
    
    if (Discrete == 2) {
      cols <- grep(paste0("^", Field, "#"), colnames(data_argumented), value = TRUE)
      for (var_mutate in cols) {
        data_argumented <- data_argumented %>%
          mutate(!!var_mutate := case_when(eval(parse(text = Exp))  ~ 0,
                                           .default = !!sym(var_mutate)))
      }
    }
  }
}

data_factor_processed <- data_argumented[,1:ncol(data_factor)]

save(data_factor, file = "../../results/Process_missingness/data_temp/data_factor.rda")
save(data_factor_processed, file = "../../results/Process_missingness/data_temp/data_factor_processed.rda")

# prepare data for impute_mf.R ------------------------------------------------------------
input_mf <- data_factor_processed %>%
  select(-eid)

eid_all <- data_factor_processed %>% select(eid)
write_csv(eid_all, file = "../../results/Process_missingness/data_temp/eid_all.csv")
save(input_mf, file = "../../results/Process_missingness/data_temp/input_mf.rda")

