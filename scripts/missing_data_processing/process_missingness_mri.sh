Rscript prepare_mri.R
Rscript impute_mri.R # MissForest by default. For other imputation methods, specify variable `method` in impute_mri_constant_or_mim.R and run `Rscript impute_mri_constant_or_mim.R` instead.
Rscript pca_mri.R
Rscript impute_pc.R 