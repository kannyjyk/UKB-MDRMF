library(data.table)
library(optparse)
option_list = list(
  make_option(c("-p", "--path"), type="character", default=NULL, metavar="character"),
  make_option(c("-f", "--file"), type="character", default=NULL,  metavar="character")
)
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)
data <- fread("/data1/ThibordGWAS/meta_Single_lgddimer.txt.gz",data.table = FALSE)
# file = paste0(opt$path, opt$file)
# data<-fread(file=file,header=T,data.table = FALSE)
index <- which(data$dbSNPID!=".")

data <- data[index, c("dbSNPID","EA","NEA","beta","p")]
new_filename = strsplit(opt$file, "\\.")[[1]][1]
write.table(data,file = paste0(opt$path,"temp/",new_filename), row.names = FALSE, col.names = FALSE, sep = "\t", quote = FALSE)
