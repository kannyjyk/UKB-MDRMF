import subprocess
from concurrent.futures import ThreadPoolExecutor
import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

from config import *  # Import all variables and functions from the config module

# List of commands to execute. Replace these with the paths to your specific Python scripts.
commands = [
    'python ./xgb.py',
    'python ./lgb.py',
    'python ./rf.py' # Change this line to your rapids-23.10 Python path. Find it with 'which python' in its conda environment
]  

print(commands)  # Print the list of commands to be executed

max_concurrent = 20  # Maximum number of concurrent threads to run

def execute_command(command):
    """
    Executes a shell command and waits for it to complete.

    Args:
        command (str): The command to execute.
    """
    # Print the command being executed and the total number of commands
    print(f"Executing command: {command}, total {len(commands)}")
    
    # Start the subprocess with the given command
    process = subprocess.Popen(command, shell=True)
    
    # Wait for the subprocess to finish
    process.wait()
    
    # Print a message indicating the command has finished
    print(f"Command finished: {command}")

# Create a ThreadPoolExecutor to manage concurrent execution of commands
with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
    # Submit all commands to the executor for execution
    futures = [executor.submit(execute_command, command) for command in commands]
    
    # Iterate over the futures to ensure all commands have completed
    for future in futures:
        future.result()  # Wait for each future to complete

print("All commands executed.")  # Print a final message after all commands have been executed
