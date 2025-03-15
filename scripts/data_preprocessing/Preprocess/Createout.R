Creatout <- function(pheno, varName){
  dataPheno = showcase[which(showcase$FieldID==varName),]
  dataCode = dataPheno$Coding
  
  dataCodeRow = which(coding$coding==dataCode)
  if (length(dataCodeRow)==0) {
    return(list(pheno, out = NULL))
  }else if (length(dataCodeRow)>1) {
    cat("WARNING: >1 rows in data code info file || ")
    return(list(pheno, out = NULL))
  }else{
    dataDataCode = coding[dataCodeRow,]
    out_code = as.character(dataDataCode$out)
    return(outing(pheno, varName, out_code))
  }
}

outing <- function(pheno, varName, out_code){
  if (!is.na(out_code) && nchar(out_code)>0){
    out_codes <- strtoi(unlist(strsplit(out_code, split = "\\|")))
    cat("Make columns: ", paste(out_codes, collapse = ","), " ||")
    colname <- paste0(varName, "#", out_codes)
  }else{
    return(list(pheno, out = NULL))
  }
  
  if (!is.null(dim(pheno))) {
    row = dim(pheno)[1]
    
  }else{
    row = length(pheno)
  }
  col = length(colname)
  out_frame = data.frame(matrix(0,nrow = row, ncol = col))
  names(out_frame) <- colname
  
  if (!is.null(dim(pheno))){
    i = 0
    for (code in out_codes) {
      i = i + 1
      if(is.na(code)){
        temp = ifelse(apply(is.na(pheno),MARGIN = 1, FUN = all), 1, 0)
        out_frame[,i] <- temp
      }else{
        temp = ifelse(apply(pheno == code, MARGIN = 1, FUN = any), 1, 0)
        temp[is.na(temp)] <- 0
        out_frame[,i] <- temp
        pheno[pheno == code] <- NA
      }
    }
  }else{
    i = 0
    for (code in out_codes) {
      i = i + 1
      if(is.na(code)){
        temp = is.na(pheno)
        out_frame[,i] <- temp
      }else{
        temp = ifelse(pheno == code, 1, 0)
        temp[is.na(temp)] <- 0
        out_frame[,i] <- temp
        pheno[pheno == code] <- NA
      }
    }
  }
  
  return(list(pheno, out = out_frame))
  
}