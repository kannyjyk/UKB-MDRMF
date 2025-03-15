library(plyr)
Reassignvalue <- function(pheno, reassignments){
  if (!is.na(reassignments) && nchar(reassignments)>0) {
    
    reassignParts = unlist(strsplit(reassignments,"\\|"));
    cat(paste("reassignments: ", reassignments, " || ", sep=""))
    
    from = NULL
    to = NULL
    
    for(i in reassignParts) {
      reassign = unlist(strsplit(i,"="))
      
      from = c(from, strtoi(reassign[1]))
      to = c(to, strtoi(reassign[2]))
    }
    if (is.null(dim(pheno))) {
      pheno = mapvalues(pheno, from, to, warn_missing = FALSE)
    }else{
      for (i in 1:ncol(pheno)) {
        pheno[,i] <- mapvalues(pheno[,i], from, to, warn_missing = FALSE)
      }
    }
    
  }
  
  return(pheno)
}

Reassignment <- function(pheno, varName){
  dataPheno = showcase[which(showcase$FieldID==varName),]
  dataCode = dataPheno$Coding
  
  dataCodeRow = which(coding$coding==dataCode)
  if (length(dataCodeRow)==0) {
    return(pheno)
  }
  else if (length(dataCodeRow)>1) {
    cat("WARNING: >1 ROWS IN DATA CODE INFO FILE || ")
    return(pheno)
  }else{
    dataDataCode = coding[dataCodeRow,]
    reassignments = as.character(dataDataCode$reassignment)
    return(Reassignvalue(pheno, reassignments))
  }
  
}