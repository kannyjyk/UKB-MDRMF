xInteger <- function(pheno, varName, trainindex, valindex, testindex){
  cat("Integer || ")
  if (!is.null(dim(pheno))) {
    phenoAvg = rowMeans(pheno, na.rm=TRUE)
    phenoAvg = ReplaceNaN(phenoAvg)
  }else{
    phenoAvg = pheno
  }
  
  phenoTrain = phenoAvg[trainindex]
  phenoVal = phenoAvg[valindex]
  phenoTest = phenoAvg[testindex]
  
  uniqVar = unique(na.omit(phenoTrain))
  if (length(uniqVar)>=3) {
    cat("To Continuous || ")
    return(xContinuous(phenoAvg, varName, trainindex, valindex, testindex))
  }else if(length(uniqVar) ==2){
    cat("BINARY")
    minValue <- min(uniqVar)
    maxValue <- max(uniqVar)
    phenoVal[which(phenoVal<minValue)] <- minValue
    phenoVal[which(phenoVal>maxValue)] <- maxValue
    phenoTest[which(phenoTest<minValue)] <- minValue
    phenoTest[which(phenoTest>maxValue)] <- maxValue
    return(list(train=Makedataframe(phenoTrain, varName), val=Makedataframe(phenoVal, varName), test=Makedataframe(phenoTest, varName)))
  }else if(length(uniqVar) <= 1){
    cat("Skip (number of levels: ",length(uniqVar),")",sep="")
    return(NULL)
  }
}