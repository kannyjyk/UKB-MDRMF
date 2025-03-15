import pandas as pd
import os

# Define input file path and output directory
# file_path = "path/to/ukb*****.csv"
file_path = os.getenv("BASIC_FILE_PATH")
output_dir = "../../../results/cache/ukbcsv"

os.makedirs(output_dir, exist_ok=True)

# Set the number of rows per chunk to manage memory usage
chunk_size = 10000

# Retrieve all column names from the CSV file
all_columns = pd.read_csv(file_path, nrows=0).columns.tolist()
necessary_columns = []

# for cols in all_columns:
#     if cols == "eid":
#         necessary_columns.append(cols)
#     else:
#         icol = int(cols.split("-")[0])
#         if icol > 130000:
#             necessary_columns.append(cols)
#         if icol < 100:
#             necessary_columns.append(cols)
#         if icol in [40001, 40007, 41270, 41280, 20001, 20007, 20002, 20009]:
#             necessary_columns.append(cols)

for cols in reversed(all_columns):
    if cols == "eid":
        necessary_columns.append(cols)
    else:
        icol = int(cols.split("-")[0])
        if icol > 130000:
            necessary_columns.append(cols)
        if icol < 100:
            necessary_columns.append(cols)
        if icol in [40001, 40007, 41270, 41280, 20001, 20007, 20002, 20009]:
            necessary_columns.append(cols)


# Process necessary columns to save time, remove the next line if you want to cache all columns, or if you had any issues
all_columns = necessary_columns

batch_size = 10
total_batches = len(all_columns) // batch_size + (
    1 if len(all_columns) % batch_size != 0 else 0
)

print(f"Total columns: {len(all_columns)}")
print(f"Processing in batches of {batch_size} columns...")
print(f"Total batches: {total_batches}")

for batch_index, i in enumerate(range(0, len(all_columns), batch_size)):
    current_columns = all_columns[i : i + batch_size]
    column_data = {col: [] for col in current_columns}

    print(
        f"Processing batch {batch_index + 1}/{total_batches}: Columns {i + 1} to {min(i + batch_size, len(all_columns))}"
    )

    # ---------------
    # Skip columns where output file already exists
    current_columns = [
        col for col in current_columns
        if not os.path.exists(os.path.join(output_dir, f"{col}.csv"))
    ]

    if not current_columns:
        print(f"All columns in batch {batch_index + 1} already processed. Skipping...")
        continue
    # ---------------

    # Read the CSV file in chunks
    chunks = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)

    for chunk_number, chunk in enumerate(chunks):
        print(f"  Processing chunk {chunk_number + 1} for batch {batch_index + 1}...")
        for column in current_columns:
            if column in chunk.columns:
                column_data[column].append(chunk[[column]])

    # Save each column's data to a separate CSV file
    for column, data_list in column_data.items():
        column_df = pd.concat(data_list)
        output_file = os.path.join(output_dir, f"{column}.csv")
        column_df.to_csv(output_file, index=False, header=False)
        print(f"Saved column '{column}' to {output_file}")

print("All columns have been saved to individual files.")
