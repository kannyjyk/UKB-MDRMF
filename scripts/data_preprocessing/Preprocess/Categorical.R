xCategorical <- function(pheno, varName, trainindex, valindex, testindex, discardlist){
  cat("Categorical || ")
  
  dataPheno = showcase[which(showcase$FieldID==varName),]
  dataCode = dataPheno$Coding
  
  dataCodeRow = which(coding$coding==dataCode)
  if (length(dataCodeRow)==0) {
    return(pheno)
  }
  else if (length(dataCodeRow)>1) {
    cat("Warning: >1 rows in data code info file || ")
    return(pheno)
  }else{
    dataDataCode = coding[dataCodeRow,]
    order_ind = dataDataCode$ordered
    discard = dataDataCode$todiscard
    if(!is.na(discard)){
      cat("Discard:", as.character(discard),"|| ")
      discardlist[[as.character(varName)]] = as.character(discard)
      discardlist <<- discardlist
    }
    if(order_ind==1){
      if(!is.null(dim(pheno))){
        phenoAvg = rowMeans(pheno, na.rm=TRUE)
        phenoAvg = ReplaceNaN(phenoAvg)
        cat("Ordered Categorical")
        temp <- Makedataframe(phenoAvg, varName)
        return(TrainValTestsplit(temp, trainindex, valindex, testindex))
      }else{
        cat("Ordered Categorical")
        temp <- Makedataframe(pheno, varName)
        return(TrainValTestsplit(temp, trainindex, valindex, testindex))
      }
    }else if(order_ind==0){
      cat("BINARY")
      temp <- cate_to_dum(pheno, varName)
      return(TrainValTestsplit(temp, trainindex, valindex, testindex))
    }else{
      cat("Wrong enter categorical!")
      return(pheno)
    }
  }
}

judge <- function(x,index){
  if(all(is.na(x))){
    return(rep(NA,length(index)))
  }else{
    return(ifelse(index%in%x,1,0))
  }
}

cate_to_dum <- function(data, fieldid){
  index <- sort(unique(as.vector(as.matrix(data))))
  if(length(index) > 0){
    name <- paste(fieldid,index,sep = "#")
  }else{
    name <- NULL
  }
  data = data.frame(data)
  write.csv(data, "data.csv", row.names = FALSE)
  cate <- t(apply(X = data, MARGIN = 1, FUN = judge, index))
  if(all(dim(cate)==c(1,0))){
    frame = data.frame(matrix(nrow = nrow(data), ncol=0))
    nas = as.integer(is.na(data[,1]))
  }else{
    frame <- as.data.frame(cate)
    nas <- as.integer(is.na(frame[,1]))
    frame[is.na(frame)] <- 0
  }
  frame <- cbind(frame,nas)
  name <- c(name,paste(fieldid,"NA",sep = "#"))
  names(frame) <- name
  return(frame)
}