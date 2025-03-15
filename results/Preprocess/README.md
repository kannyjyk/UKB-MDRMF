# Result of Preprocess

## Files to be generated here

|File|Content|
|----|-------|
|phenofile_origin.csv|The origin dataset file after selection.|
|phenofile_1.csv| The dataset processed after removing the pilot.|
|phenofile_final.csv| The output dataset after preprocess.|
|{train/val/test}_index.csv|The corresponding index file for train/val/test set, starting from zero.|
|discard.RData|The output information about columns to be discarded.|
|eid_remove.rda|The eids to be removed when we remove the pilot.|
|log.txt| Record how the programme deal with the data. |
|center.csv| A data frame containing data field 53 (assessment center each participant attends).|
|eid.csv|The eids of the whole cohort.|
|MRI.csv|The selected MRI data.|
|image_index.csv|The index indicating whether someone has Image data or not.|
|image_eid.csv|The corresponding eid of people with Image data.|
|image_{train/val/test}_index.csv|The corresponding index file of train/val/test set for `MRI.csv`.|