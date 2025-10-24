library(data.table)

to_mean <- function(data){
  frame <- as.data.frame(rowMeans(data,na.rm = TRUE))
  frame[sapply(frame, is.nan)] <- NA
  fieldid <- (strsplit(names(data)[1],"-")[[1]])[1]
  names(frame)<-fieldid
  return(frame)
}

group_to_mean <- function(data,fieldlist,contlist){
  c <- as.character(contlist)
  f <- as.character(fieldlist)
  fieldid <- sapply(strsplit(names(data),"-"),function(x) x[1])
  frame <- data.frame(eid = data$eid)
  for (i in f) {
    if((i%in%c)){
      if(sum(i==fieldid)>1){
        temp <- to_mean(data[,i==fieldid])
      }else{
        name <- strsplit(names(data)[i==fieldid],"-")[[1]][1]
        temp <- data[,i==fieldid]
        temp = data.frame(temp)
        names(temp) <- name
      }
    }else{
      temp <- data[,i==fieldid]
      if(is.vector(temp)){
        name <- strsplit(names(data)[i==fieldid],"-")[[1]][1]
        temp <- data.frame(temp)
        names(temp)<- name
      }
    }
    frame <- cbind(frame,temp)
  }
  return(frame)
}


MRI_Select <- function(datapath, MRIlist, removeeid, resultpath){
  resultDir = normalizePath(resultpath)
  table = fread(MRIlist,data.table = FALSE)
  heartlist = table$FieldID[table$Label==1]
  ulsoundlist = table$FieldID[table$Label==2]
  brainlist = table$FieldID[table$Label==3]
  alllist = table$FieldID
  reg = paste0("eid|^(", paste0(alllist, collapse = "|"),")-2")
  dat <- fread(datapath, data.table = FALSE)
  index <- grep(reg, names(dat))
  load(removeeid)
  row_index <- !(dat$eid %in% eid_remove)
  MRIdata <- dat[row_index,index]
  MRI_merge <- group_to_mean(MRIdata,alllist,alllist)
  indicator = !is.na(MRI_merge[,2:ncol(MRI_merge)])
  peopleindex = (rowMeans(indicator) > 0)
  MRI_sub <- MRI_merge[peopleindex,]
  image_index <- data.frame(image = which(peopleindex) - 1)
  image_eid <- dat$eid[row_index][peopleindex]
  write.csv(image_index, file.path(resultDir, "image_index.csv"), row.names = FALSE)
  write.csv(image_eid, file.path(resultDir, "image_eid.csv"), row.names = FALSE)
  n = dim(MRI_sub)[1]
  set.seed(0)
  index <- 0:(n-1)
  length1 <- as.integer(0.1 * n)
  length2 <- n - 2 * length1
  index1 <- sort(sample(index, length2))
  m = colMeans(MRI_sub[index1+1,], na.rm=TRUE)
  std <- apply(MRI_sub[index1+1,], 2, sd, na.rm = TRUE)
  trainMRI <- data.frame(t(apply(MRI_sub[index1+1,], 1, function(row){(row-m)/std})))
  write.csv(data.frame(train=index1),file = file.path(resultDir, "image_train_index.csv"), row.names = FALSE)
  left1 <- index[!index %in% index1]
  index2 <- sort(sample(left1, length1))
  valMRI = data.frame(t(apply(MRI_sub[index2+1,], 1, function(row){(row-m)/std})))
  write.csv(data.frame(val=index2),file = file.path(resultDir, "image_val_index.csv"), row.names = FALSE)
  index3 <- sort(left1[!left1 %in% index2])
  testMRI <- data.frame(t(apply(MRI_sub[index3+1,], 1, function(row){(row-m)/std})))
  write.csv(data.frame(test=index3),file = file.path(resultDir, "image_test_index.csv"), row.names = FALSE)
  MRI_result <- data.frame(matrix(nrow = n, ncol = ncol(MRI_sub)))
  colnames(MRI_result) <- names(MRI_sub)
  MRI_result[index1+1,] <- trainMRI
  MRI_result[index2+1,] <- valMRI
  MRI_result[index3+1,] <- testMRI
  MRI_result$eid <- NULL
  write.csv(MRI_result, file.path(resultDir, "MRI.csv"), row.names = FALSE)
}