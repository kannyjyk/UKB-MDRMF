import os
import subprocess

# Define the basic file path as an environment variable
file_path = "../../../data/ukbxxxxxx.csv"


os.environ['BASIC_FILE_PATH'] = file_path 

def run_script(script_name, args=None):
    """
    Runs a script with optional arguments and prints separators for clarity.
    """
    print(f"\n{'=' * 50}")
    print(f"Executing {script_name} {' '.join(args) if args else ''}")
    print(f"{'=' * 50}")
    if args:
        subprocess.run(['python', script_name] + args)
    else:
        subprocess.run(['python', script_name])

# Execute each script with separators
run_script('./cachefile.py')
run_script('./firstocc.py')
# run_script('./primary_care.py')  # Uncomment if needed
run_script('./imgtime.py', ['--time_type', '0'])
run_script('./imgtime.py', ['--time_type', '1'])

print("\nAll scripts executed successfully.")
