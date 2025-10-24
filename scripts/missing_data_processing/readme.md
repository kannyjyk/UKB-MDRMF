# Missingness process

**Note:** Set the working directory to the current directory before running.

## Workflow

#### Step1

Arrange Input files in [*data/Process_missingness/data_input*](../../data/Process_missingness/data_input), see [*data/Process_missingness/data_input/readme.md*](../../data/data_input/Process_missingness.md) for details.

#### Step2

1.  Run **process_missingness.sh** to deal with missingness for all phenotypes except MRI.

2.  Run **process_missingness_mri.sh** to deal with missingness for MRI data.

#### Generated Files

-   Temporary files generated in [*data/Process_missingness/data_temp*](../../data/Process_missingness/data_temp)

-   Imputed data saved in [*results/Process_missingness*](../../results/Process_missingness).

## Code Files

**adjust.R**

-   gather the one-hot encoded variables.
-   convert single categorical variables to factor type.
-   manual adjustment for Type-II related variables, create new factors. (NA for categorical, mean value for continuous)
-   other detailed adjustments

**prepare.R**

-   prepare data for imputation via MissForest.

**impute.R**

-   Impute via MissForest.

**impute_constant_or_mim.R**

-   Constant imputation or missing indicator method.
-   Specify imputation method with variable `method`.
-   Unused by default.

**prepare_mri.R**

-   prepare data for imputation via MissForest.

**impute_mri.R**

-   Impute via MissForest.

**impute_mri_constant_or_mim.R**

-   Constant imputation or missing indicator method.
-   Specify imputation method with variable `method`.
-   Unused by default.

**pca_mri.R**

-   Apply PCA for MRI data, generate PCs for downstreaming analysis.

**impute_pc.R**

-   Impute missing values for PCs of MRI data.
