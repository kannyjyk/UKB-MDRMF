# UKB-MDRMF: A Multi-Disease Risk and Multimorbidity Framework Based on UK Biobank Data

## Overview

UKB-MDRMF is a comprehensive framework for health risk prediction and multimorbidity assessment, leveraging multimodal data from the UK Biobank. By integrating diverse data types — including demographic, lifestyle, clinical measurements, environmental, genetic, and imaging data — UKB-MDRMF enables simultaneous risk prediction for 1,560 diseases. Unlike traditional single-disease models, this framework captures inter-disease connections, revealing shared and disease-specific risk factors. Through joint modeling of disease risk and multimorbidity mechanisms, UKB-MDRMF achieves superior predictive accuracy. This holistic approach provides a deeper understanding of health dynamics, supporting the discovery of novel disease associations and risk pathways.

## Features
- **Multi-Disease Prediction**: Predict risks for 1,560 diseases simultaneously.
- **Individual Risk Assessment**: Assess health risks based on a comprehensive set of variables.
- **Multimodal Data Integration**: Utilize data from various sources including lifestyle, measurements, genetics, and imaging.
- **High Predictive Performance**: Achieve significant improvements in predictive accuracy for a vast majority of disease types.
- **Connections among risk factors and diseases**: Offer a broader perspective on health and multimorbidity mechanisms.
- **Interactive Platform**: Explore detailed results and variable importance through our [interactive platform](https://luminite.shinyapps.io/ukb-mdrmf/).

## Repository Structure
This repository is organized into five main parts:

1. **Data Preprocessing**: Cleaning and preparing the data.
2. **Missing Data Processing**: Handling missing data to ensure robust model performance.
3. **Disease Diagnosis**: Predicting various diseases.
4. **Risk Assessment**: Assessing individual health risks.
5. **Others**: The impact of various risk factors and the interrelationships between multiple diseases, comparative analysis and figures.

--------
## Getting Started

### Prerequisites
- Python 3.7 or higher
- R 4.3 or higher
- Required Python libraries (specified in `requirements.txt`)
- Required R packages:
    - tidyverse 2.0.0 
    - data.table 1.14.8
    - missRanger 2.4.0
    - scales 1.2.1
    - forcats 1.0.0
    - ggsci 3.0.0
    - ggpubr 0.6.0
    - optparse 1.7.3
    - plyr 1.8.9
    - zip 2.3.0
    - nFactors 2.4.1.1

### Installation
Clone the repository to your local machine:
```sh
git clone https://github.com/kannyjyk/UKB-MDRMF.git
cd UKB-MDRMF
```

Install the required Python libraries:
```sh
pip install -r requirements.txt
```
The environment setup is expected to be completed within 30 minutes, depending on the network speed.

### Usage
For quick testing and demonstration of the modeling process, we provide a script to generate UKB-like data and simulate the modeling workflow: `scripts/model_construction/randomdatademo.ipynb`. Once the environment is set up, you can run it directly.

To reproduce the results and generate the figures in the manuscript, follow these steps:
#### Data Preparation
Before running the code, place the main table from the UKB dataset (`ukbxxxxxx.csv`, required) and the primary care file (`gp_clinical.csv`, optional) in the `./data/` directory. If available, store the SNP data (optional) in the `./data/PRS/Field22418` directory.

#### Data Preprocessing
Navigate to the `scripts/data_preprocessing` folder and run the preprocessing scripts in the following order:
1.	**Preprocess**: Run all scripts in the Preprocess folder.
2.	**PRS**: If genetic information is available, run the scripts in the PRS folder; otherwise, this step can be skipped.
3.	**Label Creation**: Finally, run the scripts in the Label_creation folder to generate the necessary labels for downstream analysis.

#### Missing Data Processing
Handle missing data using the scripts in the `scripts/missing_data_processing` folder.

#### Disease Prediction and Risk Assessment
Run disease prediction and health risk assessment models from the `scripts/model_construction` folder. This unified structure streamlines the process of predicting disease diagnoses and assessing individual health risks, enabling a more efficient and comprehensive analysis.

#### Risk Factors and Multimorbidity Analysis
Analyze risk factors and the relationships between multiple diseases using the scripts in the `scripts/disease_diagnosis` and `scripts/plot` folders.

#### Comparative Analysis
Conduct comparative analysis using the scripts in the `scripts/comparative_analysis` folder.

#### Figures
Generate figures in the manuscript using the scripts in the `scripts/plot` folder.

### Machine Learning Methods and Intermediate Results
The parameters, weights, and intermediate results used for plotting are available on Zenodo. You can access them via the following link: [Model parameters and outcomes of "UKB-MDRMF: A Multi-Disease Risk and Multimorbidity Framework Based on UK Biobank Data"](https://zenodo.org/records/15027261).

- **Data Preprocessing**: GPU is not required for preprocessing. On a machine with 64GB+ of RAM, the preprocessing step typically takes about one day to complete.
- **Model Training**: If a GPU server is available, the entire model training process is expected to be completed within 2-3 days.
   
------
## Contributing
We welcome contributions to improve UKB-MDRMF. Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

## Acknowledgements
We thank the UK Biobank for providing the data and all contributors.

