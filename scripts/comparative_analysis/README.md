# Comparative Analysis
- `disease_code_check.ipynb` is used to verify the ICD-PHECODE mapping.
- `auc_generator.py` is used to compare methods from various studies on disease prediction.
- `cindex_generator.py` is used to compare methods from various studies on survival analysis.

- **Input**: 
  - target disease name and ICD10 code(regular expression)
  - trained all-disease network
  - icd10-phecode mapping file and input-phecode mapping file
- **Output**:
  - disease-specific network
  - disease-specific auc/cindex
