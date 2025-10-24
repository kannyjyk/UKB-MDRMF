xContinuous <- function(pheno, varName, trainindex, valindex, testindex){
  cat("Continuous || ")
  if (!is.null(dim(pheno))) {
    phenoAvg = rowMeans(pheno, na.rm=TRUE)
  }
  else {
    phenoAvg = pheno
  }
  phenoAvg = ReplaceNaN(phenoAvg)
  phenoTrain = phenoAvg[trainindex]
  phenoVal = phenoAvg[valindex]
  phenoTest = phenoAvg[testindex]
  
  
  numNotNA=length(na.omit(phenoTrain))
  uniqVar = unique(na.omit(phenoTrain))
  valid = TRUE
  for (uniq in uniqVar) {
    numWithValue = length(which(phenoTrain==uniq))
    if (numWithValue/numNotNA >=0.2) {
      valid = FALSE;
      break
    }
  }
  
  if(valid == FALSE){
    cat(">20% in one category within train set || ")
    numUniqueValues = length(uniqVar)
    if (numUniqueValues<=1) {
      cat("SKIP (number of levels within train set: ",numUniqueValues,")",sep="")
      return(NULL)
    }else if (numUniqueValues == 2){
      cat("BINARY")
      minValue <- min(uniqVar)
      maxValue <- max(uniqVar)
      phenoVal[which(phenoVal<minValue)] <- minValue
      phenoVal[which(phenoVal>maxValue)] <- maxValue
      phenoTest[which(phenoTest<minValue)] <- minValue
      phenoTest[which(phenoTest>maxValue)] <- maxValue
      return(list(train=Makedataframe(phenoTrain, varName), val=Makedataframe(phenoVal, varName), test=Makedataframe(phenoTest, varName)))
    }else{
      temp <- equalSizedBins(phenoAvg, trainindex, valindex, testindex)
      cat("Ordered Categorical")
      return(list(train=Makedataframe(temp$train, varName), val=Makedataframe(temp$val, varName), test=Makedataframe(temp$test, varName)))
    }
  }else{
    cat("Scale || Continuous ")
    meanvalue = mean(phenoTrain, na.rm = TRUE)
    sdvalue = sd(phenoTrain, na.rm = TRUE)
    phenoTrain = (phenoTrain - meanvalue)/sdvalue
    phenoVal = (phenoVal - meanvalue)/sdvalue
    phenoTest = (phenoTest - meanvalue)/sdvalue
    return(list(train=Makedataframe(phenoTrain, varName), val=Makedataframe(phenoVal, varName), test=Makedataframe(phenoTest, varName)))
  }
}