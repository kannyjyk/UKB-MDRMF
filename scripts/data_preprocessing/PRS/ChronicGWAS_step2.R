library(data.table)
library(optparse)
option_list = list(
  make_option(c("-d", "--data"), type="character", default=NULL, metavar="character"),
  make_option(c("-m", "--map"), type="character", default=NULL,  metavar="character"),
  make_option(c("-o", "--out"), type="character", default=NULL, metavar = "character")
)
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

data = fread(opt$data, data.table = FALSE)
map = fread(opt$map, data.table = FALSE,fill = TRUE)

index = (map$V4!="") & (!is.na(data$V5))
new_data <- data[index,c(3, 5, 6)]
new_data$rsid <- map$V4[index]
write.table(new_data, opt$out, row.names = FALSE, col.names = FALSE, sep = "\t", quote = FALSE)