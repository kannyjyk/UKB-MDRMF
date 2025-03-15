library(data.table)
library(zip)
library(optparse)
option_list = list(
  make_option(c("-p", "--path"), type="character", default=NULL, metavar="character")
)
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)
path = opt$path
dir.create(paste0(path, "process/"))

dat <- fread(paste0(path, "CAC1000G_EA_FINAL_FULL.sumstats.gwascatalog.ssf.tsv.gz"),data.table = FALSE)
index <- (dat$rsid!="#NA") & (nchar(dat$effect_allele)==1)
dat <- dat[index,c("rsid","effect_allele","other_allele","beta","p_value")]
write.table(dat,file = paste0(path, "process/CAC_ready"), row.names = FALSE, col.names = FALSE, sep = "\t", quote = FALSE)

dat <- fread(paste0(path, "T1D_meta_FinnGen_r3_T1D_STRICT.all_chr.hg19.sumstats.txt.gz"),data.table = FALSE)
index <- nchar(dat$REF)==1
dat <- dat[index,c("CHR","POS","REF","ALT","BETA","PVALUE")]
write.table(dat,file = paste0(path, "process/FinnGen_raw"), row.names = FALSE, col.names = FALSE, sep = "\t", quote = FALSE)

dat <- fread(paste0(path, "motor-recovery-gwas-results.tsv"),data.table = FALSE)
index <- nchar(dat$allele1)==1
dat <- dat[index, c("chromosome","physical.pos","allele1","allele2","Estimate","Pvalue")]
write.table(dat,file = paste0(path, "process/Motor_raw"), row.names = FALSE, col.names = FALSE, sep = "\t", quote = FALSE)

dat <- fread(paste0(path, "DIAMANTE-EUR.sumstat.txt.gz"), data.table = FALSE)
index <- nchar(dat$effect_allele)==1
dat <- dat[index,c("rsID","effect_allele","other_allele","Fixed-effects_beta","Fixed-effects_p-value")]
dat$effect_allele <- toupper(dat$effect_allele)
dat$other_allele <- toupper(dat$other_allele)
write.table(dat,file = paste0(path, "process/DIAMANTE_ready"), row.names = FALSE, col.names = FALSE, sep = "\t", quote = FALSE)

i = 1
for(f in list.files(path)){
  if(startsWith(f, "cade_et_al_2021")){
    dat <- fread(paste0(path, f), data.table = FALSE)
    index <- nchar(dat$ALLELE1)==1
    dat <- dat[index, c("CHR","BP","ALLELE1","ALLELE0","BETA","P")]
    write.table(dat, file = paste0(path, "process/", "CADE", i, "_raw"), row.names = FALSE, col.names = FALSE, sep = "\t", quote = FALSE)
    i = i + 1
  }
}