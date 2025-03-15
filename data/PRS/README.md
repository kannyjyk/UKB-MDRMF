# PRS Data

## Note

Here are the data needed to compute PRS, including GWAS data and Genotype calls.

GWAS Data from other researches are needed to compute PRS. Here we use GWAS from four different sources.

Genotype calls are data in UKB, i.e. `Field 22418`.

In addition, `scripts/PRS/downloaddbSNP.sh` will download two files: `hg19.dbSNP150.bed` and `hg38.dbSNP150.bed` here, which record how rsid responds to chr and pos by `hg19` or `hg38`.