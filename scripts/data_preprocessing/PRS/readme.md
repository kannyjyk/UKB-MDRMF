# Guide on Computing *PRS*

## Step1: Download corresponding datasets

Download the following datsets:

- Different kinds of GWAS summary(go to the subfolders of `data/PRS` and click on the links to open the website to download them)

- UKB DataField 22418(into `data/PRS/Field22418`)

- dbSNP(`source downloaddbSNP.sh`)

## Step2: Download some tools

The following tools should be downloaded:

- [plink2](https://www.cog-genomics.org/plink/2.0/)

- [bedops](https://bedops.readthedocs.io/en/latest/)

## Step3: Merge all `.bed` files of DataField22418

```shell
source plink_merge.sh
```

## Step4: Use `plink2` to compute *PRS*

This step may take a lot of time, but the main time consumption lies in data preprocessing. `.sh` files integrate preprocessing and computing PRS. Source them to process *PRS*.

```shell
source {Cancer/Chen/Chronic/Thibord}GWAS.sh
```

## Step5: PCA and selecting the first several PCs

```shell
Rscript PRSPC.R
```