library(tidyverse)
Center <- function(dataframe, centerpath){
    dat <- dataframe[,c('eid','54-0.0')]
    write.csv(dat, centerpath, row.names=FALSE)
}

Removepilot <- function(phenofile, centerpath, resultDir){
    center <- fread(centerpath, data.table = FALSE)
    eid_remove <- eid_pilot <- filter(center, `54-0.0` == 10003)$eid
    save(eid_remove, file = file.path(resultDir, "eid_remove.rda"))

    pheno_sel <- pheno %>%
        filter(!eid %in% eid_remove)

    pheno <<- pheno_sel
}