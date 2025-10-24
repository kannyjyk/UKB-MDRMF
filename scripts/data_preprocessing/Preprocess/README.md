# Data Preprocess

## Step1: Compute PRS according to the guidance of *[scripts/PRS/](../PRS)*(optional)

## Step2: Arrange the files in *[data/Preprocess/](../../data/Preprocess)*

| File | Content |
| ---- | ---- |
|ukbxxxxxx.csv|**UKB Phenotypes file.** It's the origin file downloaded from UK Biobank. The first column should be `eid`, and other columns should have column names with format `FieldID-Instance.Array`.|
|showcase.csv|**A showcase for the datafields in `ukbxxxxxx.csv`.** The showcase should display the data type of each datafield and show how the program is going to deal with it.|
|data-coding.csv|**A showcase for the Data-Coding that datafields in `ukbxxxxxx.csv` use.** This file shows if a data-coding is ordered or not, and if we want to reassign value of some special codes.|
|impute_manual_param.csv|**A showcase for the manual imputation of some specific datafields.**|
|MRI_showcase.csv|**A showcase for the datafields related to image.**|

## Step3: Go to the directory of `preprocess.sh`, edit it and run it

```shell
cd scripts/Preprocess
```

```shell
source preprocess.sh
```

The parameters provided in `preprocess.sh` are listed as follows, the path provided to `preprocess.sh` can either be absolute path or path related to `preprocess.sh`.

| Parameter | Meaning |
| --------- | ------- |
|datafile|Path of `ukbxxxxxx.csv` in **Step2**|
|variablelistfile|Path of `showcase.csv` in **Step2**|
|datacodingfile|Path of `data-coding.csv` in **Step2**|
|prsDir|Path where the output of *[scripts/PRS/](../PRS)* is placed in.|
|missingParam|Path of the manual imputation parameters, corresponding to `impute_manual_param.csv` in **Step2**|
|resultDir|Path where you want to place the output of the programme in.|
|MRIlistfile|Path of `MRI_showcase.csv` in **Step2**|

