## Label creation

Simply run `wrapper.py` to execute the entire label creation process. However, please note that you need to replace the line `file_path = "../../../data/ukbxxxxxx.csv"` with the path to your own UKB base CSV file. If you also have the primary care file (`gp_clinical.csv`), save it in the `./data/` directory and uncomment the line `run_script('./primary_care.py')` in `wrapper.py`.

-------
### cachefile.py

Depending on your RAM size, you might need to cache your files first. This file will help you to convert the CSV file from UKBiobank to separated column files

### firstocc.py

You will need this file only if your first occurrence data is in separate CSV file.

### primary_care.py

This file will process GP_Clinical data, resulting a dictionary that holds all the information of the original txt.

### imgtime.py

The code that creates the final label using processed data. 

Note that you may specify the time threshold by setting its parameter, 0 for image time, 1 for original time.

### wrapper.py
The execution wrapper.