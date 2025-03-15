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
index <- nchar(data$Tested_Allele)==1
name = names(data)
beta_names = name[grep("BETA", name)]
freq = numeric(length(beta_names))
i = 1
for (n in beta_names) {
  freq[i] = sum(!is.na(data[[n]]))
  i = i + 1
}
idx = which(freq==max(freq))[1]
select_beta = beta_names[idx]
select_p = sub(pattern = "BETA",replacement = "P", x = select_beta)
selectcol = c("CHR","POS","Tested_Allele","Other_Allele", select_beta, select_p)
data <- data[index,selectcol]

data$CHR[data$CHR==23] <- "X"
new_filename = strsplit(opt$file, "\\.")[[1]][1]
write.table(data,file = paste0(opt$path,"temp/",new_filename), row.names = FALSE, col.names = FALSE, sep = "\t", quote = FALSE)