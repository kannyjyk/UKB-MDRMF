library(data.table)
LoadData <- function(phenofile,showcasefile, codefile, prspath, missingpath){
  pheno <<- fread(phenofile, data.table = FALSE)
  showcase <<- fread(showcasefile, data.table = FALSE)
  coding <<- fread(codefile, data.table = FALSE)
  if(!is.null(prspath)){
    prs <<- fread(prspath, data.table=FALSE)
  }else{
    prs <<- NULL
  }
  if(!is.null(missingpath)){
    missinginfo <<- fread(missingpath, data.table=FALSE)
  }else{
    missinginfo <<- NULL
  }
}

LoadIndex <- function(indexpath){
  trainindex <<- fread(file.path(indexpath, "train_index.csv"), data.table = FALSE)$train
  valindex <<- fread(file.path(indexpath, "val_index.csv"), data.table = FALSE)$val
  testindex <<- fread(file.path(indexpath, "test_index.csv"), data.table = FALSE)$test
}