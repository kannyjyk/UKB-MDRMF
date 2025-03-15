Extract_fields <- function(data, showcase, prs){
  index <- NULL
  regexlist <- c("eid", paste0("^",showcase$FieldID,"-",showcase$Instance,"\\."))
  for (reg in regexlist) {
  index <- c(index, grep(reg,x = names(data)))
  }
  index <- sort(index)
  subdat <- data[,index]
  # average England, Scotland and Wales
  if(all(c("26410-0.0","26426-0.0","26427-0.0") %in% names(subdat))){
    temp <- rowMeans(subdat[,c("26410-0.0","26426-0.0","26427-0.0")], na.rm = TRUE)
    temp[is.nan(temp)] <- NA
    subdat$`26410-0.0` <- temp
    subdat$`26426-0.0` <- NULL
    subdat$`26427-0.0` <- NULL
  }
  if(!is.null(prs)){
    subdat <- cbind(subdat, prs)
  }
  pheno <<- subdat
}