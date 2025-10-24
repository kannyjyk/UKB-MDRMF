library(data.table)
library(nFactors)
prspath="../../results/PRS/"
resultpath="../../results/Preprocess/"
filelist = sort(list.files(prspath), method="radix")
phecode_list <- NULL
files <- NULL
first=TRUE
prss = NULL
for (file in filelist) {
  if(endsWith(file, ".P.sscore")){
    dat = fread(paste0(prspath, file), data.table = FALSE)
    phecode_list <- c(phecode_list, strsplit(file, "\\.P")[[1]][1])
    files <- c(files, file)
    if(first){
      prss <- cbind(prss, dat$IID)
      first <- FALSE
    }
    prss <- cbind(prss, dat$SCORE1_AVG)
  }
}
index <- (prss[,1] > 0)
prss <- prss[index,]
eid <- fread(paste0(resultpath, "eid.csv"),data.table = FALSE)$x
indexin <- eid %in% prss[,1]
sub <- matrix(NA, nrow = length(eid), ncol = dim(prss)[2]-1)
sub[indexin,] <- prss[,2:ncol(prss)]
sub <- scale(sub)
data_z = data.frame(sub)
names(data_z) <- phecode_list
prs <- na.omit(data_z)
corpdat1 <- cor(prs, use="pairwise.complete.obs")
pcs <- nScree(x=unname(corpdat1),model="components")
n <- pcs$Components$nparallel
vec = eigen(corpdat1)$values
eigens<-eigen(corpdat1)$vectors[,1:n]
dat <- as.matrix(data_z)
fas <- dat %*% eigens
fas <- data.frame(fas)
names(fas) <- paste0("PRSPC",1:n)
write.csv(fas,file = paste0(prspath, "PRSPC", n, ".csv"),row.names = FALSE)