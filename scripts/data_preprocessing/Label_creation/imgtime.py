import pandas as pd
import numpy as np
import os
import argparse
import pickle

# Enable extended mapping for ICD codes not included, such as those containing ‘X’ or custom-defined rules
# Please be aware that this may lead to non-standard mappings and may result in incorrect PheCodes
# Set to False to use only the default mapping
extended_mapping = False
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_type", type=int, default=0)
    args = parser.parse_args()
    time_type = args.time_type

    # Load 'eid' column from the baseline imputed CSV and convert it to a NumPy array
    inceid = pd.read_csv("../../../results/Process_missingness/baseline_imputed.csv")[
        "eid"
    ].to_numpy()

    # Create a mapping from ALT_CODE to PheCode using the phecode_icd10.csv file
    icdmap = dict(
        zip(
            pd.read_csv("../../../data/phecode_icd10.csv")["ALT_CODE"].to_numpy(),
            pd.read_csv("../../../data/phecode_icd10.csv")["PheCode"].to_numpy(),
        )
    )

    # Load PheCodes from phecode.csv and concatenate them into a single NumPy array
    phe = np.concatenate(pd.read_csv("../../../data/phecode.csv").to_numpy())

    # Initialize a dictionary to store valid ICD to PheCode mappings
    valid = {}
    for k, v in icdmap.items():
        if v in phe:
            valid[k] = v

    if extended_mapping:
        # Initialize a dictionary to store valid ICD to PheCode mappings where keys end with 'X'
        validX = {}
        for k, v in icdmap.items():
            if k.endswith("X") and v in phe:
                # Remove the trailing 'X' from the key, could lead to incorrect mappings
                validX[k[:-1]] = v  

    def icdtophe(icd):
        """
        Maps an ICD code to its corresponding PheCode.

        Parameters:
            icd (str): The ICD code to be mapped.

        Returns:
            tuple:
                - bool: Indicates whether a valid PheCode was found.
                - str or None: The corresponding PheCode if found, else None.
        """
        # Check if the full ICD code exists in the valid mapping
        if icd in valid:
            return True, valid[icd]

        elif extended_mapping==True:
            if icd[:3] in validX:
                return True, validX[icd[:3]]

        # If neither condition is met, return False and None
        else:
            return False, None

    cd3 = pd.read_csv("../../../data/cd3.tsv", delimiter="\t")
    icdmapnonalt = dict(
        zip(
            pd.read_csv("../../../data/phecode_icd10.csv")["ICD10"].to_numpy(),
            pd.read_csv("../../../data/phecode_icd10.csv")["PheCode"].to_numpy(),
        )
    )
    phemeaning = dict(
        zip(
            pd.read_csv("../../../data/phecode_definitions1.2.csv")[
                "phecode"
            ].to_numpy(),
            pd.read_csv("../../../data/phecode_definitions1.2.csv")[
                "phenotype"
            ].to_numpy(),
        )
    )

    def resolve_intv(intv):
        """
        Resolves an interval of ICD10 codes to their corresponding PheCodes.

        Parameters:
            intv (str): A string representing the ICD10 code interval in the format 'start-end'.

        Returns:
            list: A list of PheCodes that fall within the specified ICD10 interval.
        """
        inc = []
        try:
            # Split the interval into start and end ICD10 codes
            start, end = intv.split("-")
        except ValueError:
            # print(f"Invalid interval format: '{intv}'. Expected format 'start-end'.")
            return inc

        # Iterate through the ICD10 to PheCode mapping to find codes within the interval
        for k, v in icdmapnonalt.items():
            if start <= k <= end:
                inc.append(v)

        # Handle specific case where no PheCodes are found and the interval starts with 'C07', your case may vary
        if not inc:
            if intv[:3] == "C07":
                # Attempt to map the first three characters of the start code
                truncated_code = start[:3]
                phe_code = icdmapnonalt.get(truncated_code)
                if phe_code:
                    inc.append(phe_code)
                else:
                    print(f"No PheCode found for truncated code: '{truncated_code}'.")

        # If still no PheCodes are found, log the unresolved interval
        if not inc:
            print(f"Unresolved interval: '{intv}'.")

        return inc

    def resolve_code(cd):
        """
        Resolves a single ICD10 code to its corresponding PheCode.

        Parameters:
            cd (str): The ICD10 code to be resolved.

        Returns:
            list: A list containing the corresponding PheCode if found, else [np.nan].
        """
        phe = ""

        # Direct mapping from ICD10 to PheCode using icdmapnonalt
        if cd in icdmapnonalt:
            phe = icdmapnonalt[cd]

        # Mapping based on the first three characters of the ICD10 code using icdmap
        elif cd[:3] in icdmap:
            phe = icdmap[cd[:3]]

        # Direct mapping from the first three characters of the ICD10 code using icdmapnonalt
        elif cd[:3] in icdmapnonalt:
            phe = icdmapnonalt[cd[:3]]

        else:
            phe = np.nan
            # Attempt to append '.0' if not present and try resolving again
            if "." not in cd:
                cd_extended = cd + ".0"
                phe = resolve_code(cd_extended)[0]
            # else:
            #     # Log unresolved ICD10 codes, try to handle it manually
            #     print(f"Unresolved ICD10 code: {cd}")

        return [phe]

    cdcancer = {}
    mDF = []

    for row in cd3.to_numpy():
        # Extract the ICD10 code(s) from the third column of the row
        icd_codes = row[2]

        # Determine the type of ICD10 entry and resolve accordingly
        if "-" not in icd_codes:
            # Single ICD10 code without a range
            resolved = resolve_code(icd_codes)
        elif "," not in icd_codes:
            # ICD10 code represents an interval (range)
            resolved = resolve_intv(icd_codes)
        else:
            # ICD10 codes include multiple entries separated by commas
            resolved = []
            for itm in icd_codes.split(", "):
                if "-" not in itm:
                    # Resolve individual ICD10 code
                    resolved += resolve_code(itm)
                else:
                    # Resolve interval of ICD10 codes
                    resolved += resolve_intv(itm)

        # Remove duplicate PheCodes
        unique_resolved = np.unique(resolved)

        # Filter PheCodes to include only those present in the 'phe' array
        valid_phe_codes = [t for t in unique_resolved if t in phe]

        # Append each valid PheCode along with relevant information to 'mDF'
        for phe_code in valid_phe_codes:
            mDF.append([row[0], row[1], phe_code, phemeaning[phe_code]])

        # Assign the first valid PheCode to the 'cdcancer' dictionary for the given 'eid'
        if valid_phe_codes:
            cdcancer[row[0]] = valid_phe_codes[0]
        else:
            # Handle cases where no valid PheCode is found
            cdcancer[row[0]] = np.nan
            # print(f"No valid PheCode found for eid: {row[0]}")

    pd.DataFrame(mDF).to_csv("../../../data/cancermap.csv")
    include = ["20001", "20002", "40001", "41270"]
    includewtime = ["40001,40007", "41270,41280", "20001,20007", "20002,20009"]
    cod6 = pd.read_csv(
        "../../../data/Self_report_FO_mappings_Jan2022.tsv", delimiter="\t"
    )
    cod6icd = {}
    for i in cod6.to_numpy():
        cod6icd[i[0]] = i[1]

    def handle_code(datai, data):
        """
        Handles the resolution of codes based on the provided 'data' type.

        Parameters:
            datai (float or str): The input code identifier to be resolved.
            data (str): A string indicating the type of data ('20001', '20002', or others).

        Returns:
            tuple:
                - status (bool): Indicates whether the code was successfully resolved.
                - code (str or None): The corresponding PheCode if resolved; otherwise, None.
        """
        status = False  # Initialize the status as False by default
        code = None  # Initialize code as None

        # Handle the case when 'data' is '20002'
        if data == "20002":
            try:
                key = int(float(datai))
            except ValueError:
                print(f"Invalid datai value for data '20002': {datai}")
                return status, code

            # Check if the converted key exists in the 'cod6icd' dictionary
            if key in cod6icd:
                # Retrieve the ICD code from 'cod6icd' and map it to a PheCode using 'icdtophe'
                status, code = icdtophe(cod6icd[key])
            else:
                print(f"Key {key} not found in 'cod6icd' for data '20002'.")

        # Handle the case when 'data' is '20001'
        elif data == "20001":
            try:
                key = int(float(datai))
            except ValueError:
                print(f"Invalid datai value for data '20001': {datai}")
                return status, code

            # Check if the converted key exists in the 'cdcancer' dictionary
            if key in cdcancer:
                status = True  # Set status to True as the key exists
                code = cdcancer[
                    key
                ]  # Retrieve the corresponding PheCode from 'cdcancer'
            else:
                print(f"Key {key} not found in 'cdcancer' for data '20001'.")

        # Handle all other cases
        else:
            # Directly map 'datai' to a PheCode using 'icdtophe'
            status, code = icdtophe(datai)

        # Return the status and code based on whether the mapping was successful
        if status:
            return status, code
        else:
            return status, None

    handle_code(1002.0, "20001")

    # Initialize the 'overall' dictionary to store results (assumed from previous context)
    overall = {}

    # Directory containing the CSV files
    ukb_csv_dir = "../../../results/cache/ukbcsv/"

    # Iterate over each 'data,time' pair in 'includewtime'
    for entry in includewtime:
        # Split the entry into 'data_code' and 'time_code'
        data_code, time_code = entry.split(",")

        # Initialize lists to store DataFrames for 'data_code' and 'time_code'
        data_dfs = []
        time_dfs = []

        # Iterate over sorted list of filenames in the 'ukbcsv' directory
        for filename in sorted(os.listdir(ukb_csv_dir)):
            file_path = os.path.join(ukb_csv_dir, filename)

            # Check if 'data_code' is present in the filename
            if data_code in filename:
                try:
                    # Read the CSV file without headers and append to 'data_dfs'
                    df = pd.read_csv(file_path, header=None)
                    data_dfs.append(df)
                except:
                    print(f"Error reading file {file_path}")

            # Check if 'time_code' is present in the filename
            if time_code in filename:
                try:
                    # Read the CSV file without headers and append to 'time_dfs'
                    df = pd.read_csv(file_path, header=None)
                    time_dfs.append(df)
                except:
                    print(f"Error reading file {file_path}")

        # Concatenate all DataFrames in 'data_dfs' along columns to form 'dataDF'
        dataDF = pd.concat(data_dfs, axis=1)

        timeDF = pd.concat(time_dfs, axis=1)

        # Convert DataFrames to NumPy arrays for efficient processing
        tnp = timeDF.to_numpy()
        dnp = dataDF.to_numpy()

        # Initialize a list to store disease information per file
        per_file_disease = []

        # Iterate over each row in the NumPy arrays
        for row_index in range(tnp.shape[0]):
            # Initialize a dictionary to store diseases and their earliest times for this row
            disease_dict = {}

            # Iterate over each column in the current row
            for col_index in range(tnp.shape[1]):
                # Extract time and data values as strings
                time_i = str(tnp[row_index, col_index])
                data_i = str(dnp[row_index, col_index])

                # Proceed only if both time and data are not 'nan'
                if time_i != "nan" and data_i != "nan":
                    # Call 'handle_code' function to resolve the data code
                    status, code = handle_code(data_i, data_code)

                    # If the code was not successfully resolved, skip to next iteration
                    if not status:
                        continue

                    # Attempt to convert 'time_i' to float and check if it's negative
                    try:
                        time_float = float(time_i)
                        if time_float < 0:
                            continue  # Skip if time is negative (indicating an error code)
                    except ValueError:
                        # If 'time_i' cannot be converted to float (e.g., it's text from 41270), ignore
                        pass

                    # Update 'data_i' with the resolved code
                    data_i = code

                    # If the disease code is already in 'disease_dict', keep the earliest time
                    if data_i in disease_dict:
                        existing_time = disease_dict[data_i]
                        # Formatted time, compare as strings
                        if time_i < existing_time:
                            disease_dict[data_i] = time_i
                    else:
                        # Add the disease code and its time to 'disease_dict'
                        disease_dict[data_i] = time_i

            # Append the 'disease_dict' dictionary to 'per_file_disease' list
            per_file_disease.append(disease_dict)

        # Assign the 'per_file_disease' list to the 'overall' dictionary with 'data_code' as the key
        overall[data_code] = per_file_disease

    priprocessed = []
    # pricare = pickle.load(open("../../../results/cache/pricare", "rb"))
    pricare = {}
    fulleid = np.concatenate(
        pd.read_csv("../../../results/cache/ukbcsv/eid.csv", header=None).to_numpy()
        # pd.read_csv("../../../results/cache/ukbcsv/eid.csv").to_numpy()
    )
    fulleid = list(fulleid)
    # Initialize an empty list to store the processed results
    

    # Iterate over each key in the 'fulleid' list
    for k in fulleid:
        # Check if the current key 'k' is not present in the 'pricare' dictionary
        if k not in pricare.keys():
            # If 'k' is not in 'pricare', append an empty dictionary to 'priprocessed'
            priprocessed.append({})
        else:
            # If 'k' exists in 'pricare', retrieve its associated value (a dictionary)
            v = pricare[k]
            # Initialize an empty dictionary to store processed data for this key
            disedict = {}

            # Iterate over each key-value pair in the dictionary 'v'
            for c, date in v.items():
                status, code = handle_code(c, "42040")
                # Proceed only if 'handle_code' returned a successful status
                if status:
                    datai = code
                    if datai in disedict.keys():
                        # If the current 'date' is earlier than the stored date, update it
                        if date < disedict[datai]:
                            disedict[datai] = date
                    else:
                        disedict[datai] = date
            # After processing all items, append the 'disedict' to 'priprocessed'
            priprocessed.append(disedict)

    focc = np.load("../../../results/cache/firstocc.npy", allow_pickle=1).item()
    # Initialize an empty list to store the processed results for 'focc'
    foccprocessed = []

    # Iterate over each key in the 'fulleid' list
    for k in fulleid:
        # Check if the current key 'k' is not present in the 'pricare' dictionary
        if k not in pricare.keys():
            # If 'k' is not in 'pricare', append an empty dictionary to 'foccprocessed'
            foccprocessed.append({})
        # If 'k' is present in 'pricare', check if its string representation is not in the 'focc' dictionary
        elif str(k) not in focc.keys():
            # If the string version of 'k' is not in 'focc', append an empty dictionary to 'foccprocessed'
            foccprocessed.append({})
        else:
            # If 'str(k)' exists in 'focc', retrieve its associated value (a dictionary)
            v = focc[str(k)]
            # Initialize an empty dictionary to store processed data for this key
            disedict = {}

            # Iterate over each key-value pair in the dictionary 'v'
            for c, date in v.items():
                # Call the 'handle_code' function with 'c' and a fixed string '130000'
                # It returns a tuple: (status, code)
                status, code = handle_code(c, "130000")

                # Proceed only if 'handle_code' returned a successful status
                if status:
                    # Assign the returned 'code' to 'datai'
                    datai = code

                    # Check if 'datai' already exists in 'disedict'
                    if datai in disedict.keys():
                        # If the current 'date' is earlier than the stored date, update it
                        if date < disedict[datai]:
                            disedict[datai] = date
                    else:
                        # If 'datai' is not in 'disedict', add it with the current 'date'
                        disedict[datai] = date

            # After processing all items, append the 'disedict' to 'foccprocessed'
            foccprocessed.append(disedict)
    overall["42040"] = priprocessed
    overall["130000"] = foccprocessed
    for k, v in overall.items():
        newv = []
        for s in v:
            # Initialize an empty dictionary to store valid key-time pairs
            noerror = {}
            for tempphe, time in s.items():
                # Check if 'time' is one of the specified invalid dates
                if time in ["1901-01-01", "1902-02-02", "1903-03-03", "2037-07-07"]:
                    # If 'time' is invalid, skip this key-time pair
                    continue
                else:
                    # If 'time' is valid, add the pair to 'noerror'
                    noerror[tempphe] = time

            # After processing all key-time pairs in 's', append 'noerror' to 'newv'
            newv.append(noerror)

        # Update the 'overall' dictionary for key 'k' with the processed list 'newv'
        overall[k] = newv
    fulleid_dict = {int(value): index for index, value in enumerate(fulleid)}
    # print(fulleid_dict)

    incidx = [fulleid_dict[int(i)] for i in inceid]

    f34 = np.concatenate(
        pd.read_csv("../../../results/cache/ukbcsv/34-0.0.csv", header=None).to_numpy()
    )
    f52 = np.concatenate(
        pd.read_csv("../../../results/cache/ukbcsv/52-0.0.csv", header=None).to_numpy()
    )
    f53 = np.concatenate(
        pd.read_csv("../../../results/cache/ukbcsv/53-0.0.csv", header=None).to_numpy()
    )
    f53i = np.concatenate(
        pd.read_csv("../../../results/cache/ukbcsv/53-2.0.csv", header=None).to_numpy()
    )
    eid = np.concatenate(
        pd.read_csv("../../../results/cache/ukbcsv/eid.csv", header=None).to_numpy()
    )
    imgeid = pd.read_csv("../../../results/Preprocess/image_eid.csv")["x"].to_numpy()
    f53final = []

    # Iterate over the indices of the 'eid' list
    for i in range(len(eid)):
        # Check if the current 'eid' exists in the 'imgeid' collection
        if eid[i] in imgeid:
            # If present, append the corresponding value from 'f53i' to 'f53final'
            f53final.append(f53i[i])
        else:
            # If not present, append the corresponding value from 'f53' to 'f53final'
            f53final.append(f53[i])

    f53final = np.array(f53final)

    if time_type == 0:
        # Update the 'f53' list with the 'f53final' values if 'time_type' is 0, which is imaging time, otherwise keep the original values, which is the record time
        f53 = f53final

    # Initialize empty lists to store calculated ages
    age_of_record = []
    age_till_now = []

    # Iterate over the 'f53' list with both index 'i' and element 'd'
    for i, d in enumerate(f53):
        year = int(d.split("-")[0])
        month = int(d.split("-")[1])
        day = int(d.split("-")[2])
        age_record = year - f34[i] + month / 12 - f52[i] / 12 + day / 365
        age_of_record.append(age_record)

        age_now = 2024 - f34[i] - f52[i] / 12
        age_till_now.append(age_now)

    # age_till_now

    def calctime(date, year, mo):
        """
        Calculates the time difference between a given date and a reference year and month.

        Parameters:
            date (str): A date string in the format 'YYYY-MM-DD'.
            year (int): The reference year to subtract from the date's year.
            mo (int): The reference month to subtract from the date's month.

        Returns:
            float: The calculated time difference in years, including fractional years based on months and days.
        """

        y, m, d = np.array(date.split("-"), dtype=int)
        time_diff = y - year + (m - mo) / 12 + d / 365

        return time_diff

    for i in range(len(overall["41270"])):
        for k, v in overall["41270"][i].items():
            overall["41270"][i][k] = calctime(v, f34[i], f52[i])
    for i in range(len(overall["42040"])):
        for k, v in overall["42040"][i].items():
            overall["42040"][i][k] = calctime(v, f34[i], f52[i])
    for i in range(len(overall["130000"])):
        for k, v in overall["130000"][i].items():
            overall["130000"][i][k] = calctime(v, f34[i], f52[i])

    count_dict = {}

    # Iterate over each key-value pair in the 'overall' dictionary
    for k, v in overall.items():
        # Initialize an empty dictionary to count occurrences of sub-keys for the current main key
        patient_dict = {}

        # Print the current main key and the number of entries it has
        print(k, len(v))

        # Iterate over each dictionary in the list 'v'
        for i in v:
            # Iterate over each sub-key and its associated value in the inner dictionary 'i'
            for s, t in i.items():
                # If the sub-key 's' is already in 'patient_dict', increment its count
                if s in patient_dict.keys():
                    patient_dict[s] += 1
                else:
                    # If the sub-key 's' is not in 'patient_dict', initialize its count to 1
                    patient_dict[s] = 1

        # Assign the count dictionary 'patient_dict' to the main dictionary 'count_dict' under key 'k'
        count_dict[k] = patient_dict

    # Assign the count dictionary to the variable 'data' for clarity
    data = count_dict

    # Uncomment the following block to write the data to a CSV file
    # Define the output file path for the CSV
    # output_file = "../../../results/imgsumdata.csv"

    # # Open the output CSV file in write mode
    # with open(output_file, "w", newline="") as csvfile:
    #     # Create a CSV writer object
    #     writer = csv.writer(csvfile)

    #     # Write the header row: 'Key' followed by sorted main keys from 'data'
    #     writer.writerow(["Key"] + sorted(list(data.keys())))

    #     # Read the 'phecode.csv' file and extract inner keys
    #     # Note: Assumes that 'phecode.csv' has a single column without a header
    #     inner_keys = np.concatenate(pd.read_csv("../../../data/phecode.csv").to_numpy())

    #     # Iterate over each inner key to create rows for the CSV
    #     for inner_key in inner_keys:
    #         # Initialize the row with the current inner key as the first element
    #         row = [str(inner_key)]

    #         # Iterate over each main key in 'data' to append counts or default values
    #         for key in data.keys():
    #             # Append the count if 'inner_key' exists in 'data[key]', else append an empty string
    #             row.append(data[key].get(inner_key, ""))

    #         # Write the constructed row to the CSV file
    #         writer.writerow(row)
    phe = list(phe)
    matrix = []
    for i in range(len(age_till_now)):

        # Create a temporary array 'temp' filled with ones multiplied by the current age
        temp = np.ones(len(phe)) * age_till_now[i]
        for k, v in overall.items():
            for code, time in v[i].items():

                # Find the index of the current 'code' in the 'phe' list
                # This index corresponds to the position in the 'temp' array
                dindex = phe.index(code)

                time = float(time)

                # Check if the 'time' is greater than 'age_of_record'
                if time - age_of_record[i] > 0:
                    if temp[dindex] > time:
                        temp[dindex] = time
                else:
                    temp[dindex] = np.nan
        matrix.append(temp)
    matrix0_1 = []
    for i in matrix:
        current_time = np.nanmax(i)
        nan = np.where(np.isnan(i))[0]
        diseases_sign = np.where(i != current_time)[0]
        # Not being the current time indicates the disease has happened
        temp = np.zeros(len(phe))
        temp[diseases_sign] = 1
        temp[nan] = np.nan
        matrix0_1.append(temp)
    matrix0_1 = np.array(matrix0_1)
    fulleid_dict = {value: index for index, value in enumerate(fulleid)}
    incidx = [fulleid_dict[i] for i in inceid]
    print(incidx)
    incidx = np.array(incidx)
    os.makedirs("../../../results/cache/", exist_ok=True)
    if time_type == 0:
        np.save("../../../results/cache/0-1img", matrix0_1[incidx, :])
        np.save("../../../results/cache/coximg", np.array(matrix)[incidx, :])
    else:
        np.save("../../../results/cache/0-1", matrix0_1[incidx, :])
        np.save("../../../results/cache/cox", np.array(matrix)[incidx, :])
