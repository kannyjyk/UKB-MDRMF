import csv
import os
import pandas as pd
import numpy as np


# Create a mapping from FieldID to Field by reading the TSV data dictionary
print("Loading FieldID to Field mapping...")
data_dict_path = "../../../data/Data_Dictionary_Showcase.tsv"
data_dict = pd.read_csv(data_dict_path, delimiter="\t")
field_map = dict(zip(data_dict["FieldID"], data_dict["Field"]))
print(f"Mapping loaded: {len(field_map)} entries found.")

# Path to the input CSV file containing only first occurrence columns
file_path = os.getenv('BASIC_FILE_PATH')  # Replace with actual path if needed
output_file = "../../../results/cache/firstocc.npy"

foccdict = {}
print(f"Processing file: {file_path}")

with open(file_path, "r", newline="") as csvfile:
    reader = csv.reader(csvfile)

    # Extract the header row to map column indices
    indexmap = next(iter(reader))
    total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract 1 for header row
    print(f"Total rows to process (excluding header): {total_rows}")

    for row_index, row in enumerate(reader, start=1):
        if row_index % 1000 == 0:
            print(f"Processed {row_index}/{total_rows} rows...")

        nonempty = {}

        # Enumerate over each column in the row
        for i, info in enumerate(row):
            if i == 0:
                continue  # Skip the first column (eid)

            if info != "":
                # Extract the FieldID from the header and get the corresponding Field name
                field_id = int(indexmap[i].split("-")[0])
                line = field_map.get(field_id, f"Unknown FieldID: {field_id}")

                # If the Field name contains 'Date', process it
                if "Date" in line:
                    # Extract the second word from the Field name and assign the value
                    key = line.split(" ")[1]
                    nonempty[key] = info

        foccdict[row[0]] = nonempty

    print(f"Finished processing all {total_rows} rows.")

# Save the dictionary to a .npy file
print(f"Saving results to {output_file}...")
np.save(output_file, foccdict)
print("Save complete.")