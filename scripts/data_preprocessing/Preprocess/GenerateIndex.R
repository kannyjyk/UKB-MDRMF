GenerateIndex <- function(x, path){
  n = dim(x)[1]
  set.seed(0)
  index <- 0:(n-1)
  length1 <- as.integer(0.1 * n)
  length2 <- n - 2 * length1
  index1 <- sort(sample(index, length2))
  write.csv(data.frame(train=index1),file = file.path(path, "train_index.csv"), row.names = FALSE)
  left1 <- index[!index %in% index1]
  index2 <- sort(sample(left1, length1))
  write.csv(data.frame(val=index2),file = file.path(path, "val_index.csv"), row.names = FALSE)
  index3 <- sort(left1[!left1 %in% index2])
  write.csv(data.frame(test=index3),file = file.path(path, "test_index.csv"), row.names = FALSE)
}