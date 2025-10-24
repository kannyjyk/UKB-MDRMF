Manual_missing <- function(pheno, param){
    fun_environment <- environment()
    if(is.null(param)){
        NULL
    }else{
        for (i in 1:nrow(param)){
            param_single <- param[i,]
            list2env(param_single, envir = fun_environment)
            if (Num_related == 1){
                var_mutate <- paste0(Field, "-0.0")
                var_related <- grep(paste0("^", Related1, "-"), names(pheno), value = TRUE)
                Value_cond1_num <- as.numeric(unlist(strsplit(Value_cond1, ";")))
                if (Cond1 == "IN"){
                    pheno <- pheno %>% mutate(!!var_mutate := case_when(!!sym(var_related) %in% Value_cond1_num ~ Value,
                                                                      .default = !!sym(var_mutate)))
                }

                if (Cond1 == "OUT") {
                  if (length(var_related) > 1) {
                    logical_matrix <- sapply(pheno[var_related], function(column) column %in% c(Value_cond1_num))
                    rows_meet_condition <- apply(logical_matrix, 1, any)
                    indices <- which(!rows_meet_condition)
                    pheno[indices, var_mutate] <- Value
                  } else {
                    pheno <- pheno %>% mutate(!!var_mutate := case_when(!(!!sym(var_related) %in% c(NA,Value_cond1_num)) ~ Value,
                                                                        .default = !!sym(var_mutate)))
                  }
                }
            }

            if (Num_related == 2){
                var_mutate <- paste0(Field, "-0.0")
                var_related1 <- paste0(Related1, "-0.0")
                Value_cond1_num <- as.numeric(unlist(strsplit(Value_cond1, ",")))
                var_related2 <- paste0(Related2, "-0.0")
                Value_cond2 <- as.character(Value_cond2)
                Value_cond2_num <- as.numeric(unlist(strsplit(Value_cond2, ",")))
                if (Cond1 == "IN" & Cond2 == "IN"){
                    pheno <- pheno %>% mutate(!!var_mutate := case_when(!!sym(var_related1) %in% Value_cond1_num & 
                                                                        !!sym(var_related2) %in% Value_cond2_num ~ Value,
                                                                      .default = !!sym(var_mutate)))
                }

                if (Cond1 == "IN" & Cond2 == "OUT") {
                    pheno <- pheno %>% mutate(!!var_mutate := case_when(!!sym(var_related1) %in% Value_cond1_num & 
                                                                        !(!!sym(var_related2) %in% Value_cond2_num)~ Value,
                                                                      .default = !!sym(var_mutate)))
                }

                if (Cond1 == "OUT" & Cond2 == "IN") {
                    pheno <- pheno %>% mutate(!!var_mutate := case_when(!(!!sym(var_related1) %in% Value_cond1_num) & 
                                                                        !!sym(var_related2) %in% Value_cond2_num ~ Value,
                                                                      .default = !!sym(var_mutate)))
                }

                if (Cond1 == "OUT" & Cond2 == "OUT"){
                  pheno <- pheno %>% mutate(!!var_mutate := case_when(!(!!sym(var_related1) %in% Value_cond1_num) & 
                                                                        !(!!sym(var_related2) %in% Value_cond2_num) ~ Value,
                                                                      .default = !!sym(var_mutate)))
                }
            }
        }
      
      # impute alcohol monthly intake variables
      pheno <- pheno %>%
        mutate(`4407-0.0` = case_when(
          !is.na(`1568-0.0`) & is.na(`4407-0.0`) ~ 4.35 * `1568-0.0`,
          `1558-0.0` == 6 & is.na(`4407-0.0`) ~ 0,
          .default = `4407-0.0`
        ))
      
      pheno <- pheno %>%
        mutate(`4418-0.0` = case_when(
          !is.na(`1578-0.0`) & is.na(`4418-0.0`) ~ 4.35 * `1578-0.0`,
          `1558-0.0` == 6 & is.na(`4418-0.0`) ~ 0,
          .default = `4418-0.0`
        ))
      
      pheno <- pheno %>%
        mutate(`4429-0.0` = case_when(
          !is.na(`1588-0.0`) & is.na(`4429-0.0`) ~ 4.35 * `1588-0.0`,
          `1558-0.0` == 6 & is.na(`4429-0.0`) ~ 0,
          .default = `4429-0.0`
        ))
      
      pheno <- pheno %>%
        mutate(`4440-0.0` = case_when(
          !is.na(`1598-0.0`) & is.na(`4440-0.0`) ~ 4.35 * `1598-0.0`,
          `1558-0.0` == 6 & is.na(`4440-0.0`) ~ 0,
          .default = `4440-0.0`
        ))
      
      pheno <- pheno %>%
        mutate(`4451-0.0` = case_when(
          !is.na(`1608-0.0`) & is.na(`1608-0.0`) ~ 4.35 * `1608-0.0`,
          `1558-0.0` == 6 & is.na(`4451-0.0`) ~ 0,
          .default = `4451-0.0`
        ))
      
      pheno <- pheno %>%
        mutate(`4462-0.0` = case_when(
          !is.na(`5364-0.0`) & is.na(`4462-0.0`) ~ 4.35 * `5364-0.0`,
          `1558-0.0` == 6 & is.na(`4462-0.0`) ~ 0,
          .default = `4462-0.0`
        ))
    }
    
    pheno <<- pheno
}

