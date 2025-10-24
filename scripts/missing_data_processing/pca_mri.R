library(data.table)
mridata = "../../results/Process_missingness/mri_imputed.csv"
inputPath = "../../results/Preprocess"
resultPath = "../../results/Process_missingness"
mrishowcasepath = "../../data/Preprocess/MRI_showcase.csv"


trainindex <- fread(file.path(inputPath, "image_train_index.csv"), data.table = FALSE)$train + 1
valindex <- fread(file.path(inputPath, "image_val_index.csv"), data.table = FALSE)$val + 1
testindex <- fread(file.path(inputPath, "image_test_index.csv"), data.table = FALSE)$test + 1
mri <- fread(mridata, data.table = FALSE,header = TRUE)
showcase <- fread(mrishowcasepath, data.table = FALSE)
heart <- as.character(showcase$FieldID[showcase$Label==1])
ultrasound <- as.character(showcase$FieldID[showcase$Label==2])
brain <- as.character(showcase$FieldID[showcase$Label==3])
fieldids <- as.character(sapply(strsplit(names(mri), split = "#"), function(x) return(x[1])))
heartindex <- which(fieldids %in% heart)
ultrasoundindex <- which(fieldids %in% ultrasound)
brainindex <- which(fieldids %in% brain)
train_heartmri <- mri[trainindex,heartindex]
train_ultrasound <- mri[trainindex, ultrasoundindex]
train_brainmri <- mri[trainindex,brainindex]
heart_pca <- eigen(cor(train_heartmri))
ultrasound_pca <- eigen(cor(train_ultrasound))
brain_pca <- eigen(cor(train_brainmri))
for (heart_num in 1:length(heartindex)) {
  if(sum(heart_pca$values[1:heart_num])/sum(heart_pca$values) > 0.9){break}
}
for (ultra_num in 1:length(ultrasoundindex)) {
  if(sum(ultrasound_pca$values[1:ultra_num])/sum(ultrasound_pca$values) > 0.9){break}
}
for (brain_num in 1:length(brainindex)) {
  if(sum(brain_pca$values[1:brain_num])/sum(brain_pca$values) > 0.8){break}
}
heart_loading <- heart_pca$vectors[,1:heart_num]
ultra_loading <- ultrasound_pca$vectors[,1:ultra_num]
brain_loading <- brain_pca$vectors[,1:brain_num]

heart_pcresult <- data.frame(as.matrix(mri[,heartindex])%*%heart_loading)
ultra_pcresult <- data.frame(as.matrix(mri[,ultrasoundindex])%*%ultra_loading)
brain_pcresult <- data.frame(as.matrix(mri[,brainindex])%*%brain_loading)
names(brain_pcresult) <- paste0("Brain_PC",1:brain_num)
names(ultra_pcresult) <- paste0("Ultrasound_PC", 1:ultra_num)
names(heart_pcresult) <- paste0("Heart_PC",1:heart_num)
MRIPC <- cbind(heart_pcresult,ultra_pcresult ,brain_pcresult)
m = colMeans(MRIPC[trainindex,], na.rm=TRUE)
std <- apply(MRIPC[trainindex,], 2, sd, na.rm = TRUE)
MRIPC_std <- data.frame(t(apply(MRIPC, 1, function(row){(row-m)/std})))
write.csv(MRIPC_std, file.path(resultPath, "MRIPC.csv"), row.names = FALSE)
