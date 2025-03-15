Makedataframe <- function(pheno, varName){
  dat <- data.frame(pheno)
  names(dat) <- as.character(varName)
  return(dat)
}