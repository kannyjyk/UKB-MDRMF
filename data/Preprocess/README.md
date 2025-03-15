# Data Preprocess

## Files to be placed here

| File | Content |
| ---- | ---- |
|ukbxxxxxx.csv|**UKB Phenotypes file.** It's the origin file downloaded from UK Biobank. The first column should be `eid`, and other columns should have column names with format `FieldID-Instance.Array`.|
|showcase.csv|**A showcase for the datafields in `ukbxxxxxx.csv`.** The showcase should display the data type of each datafield and show how the program is going to deal with it.|
|data-coding.csv|**A showcase for the Data-Coding that datafields in `ukbxxxxxx.csv` use.** This file shows if a data-coding is ordered or not, and if we want to reassign value of some special codes.|
|impute_manual_param.csv|**A showcase for the manual imputation of some specific datafields.**|
|MRI_showcase.csv|**A showcase for the datafields related to image.**|