library(data.table)
library(optparse)
option_list = list(
  make_option(c("-p", "--path"), type="character", default=NULL, metavar="character"),
  make_option(c("-f", "--file"), type="character", default=NULL,  metavar="character")
)
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

file = paste0(opt$path, opt$file)
data<-fread(file=file,header=T,data.table = FALSE)
selectcol = c("rs_number","reference_allele", "other_allele", "beta", "p-value")
data <- data[,selectcol]
new_filename = strsplit(opt$file, "\\.")[[1]][1]
write.table(data,file = paste0(opt$path,"temp/",new_filename), row.names = FALSE, col.names = FALSE, sep = "\t", quote = FALSE)