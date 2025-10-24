import subprocess
from concurrent.futures import ThreadPoolExecutor
from config import *
import os
# This file generates the commands to run the varimp and varimpsurv scripts,
# which are used to calculate the variable importance of the models

commands = []  # Initialize an empty list to store the commands
gpu = 0  # Initialize GPU counter
folder='../../results/importance'
os.makedirs(folder, exist_ok=True)  # Create the folder if it does not exist
# Generate commands for the varimpsurv.py script
for i in range(0, 1560, 30):
    gpu += 1  # Increment GPU counter
    # Append a command string to the commands list
    commands.append(
        f'python ./varimpsurv.py {i} {i + 30} {ava_gpus(gpu)} {folder}'
    )

max_concurrent = 10  # Maximum number of concurrent threads to run

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
    # Submit all varimpsurv commands to the executor for execution
    futures = [executor.submit(execute_command, command) for command in commands]
    
    # Iterate over the futures to ensure all commands have completed
    for future in futures:
        future.result()  # Wait for each future to complete

# Reset the commands list and GPU counter for the next set of commands
commands = []
gpu = 0

# Generate commands for the varimp.py script
for i in range(0, 1560, 30):
    gpu += 1  # Increment GPU counter
    # Append a command string to the commands list
    commands.append(
        f'python ./varimp.py {i} {i + 30} {ava_gpus(gpu)} {folder}'
    )

max_concurrent = 10  # Maximum number of concurrent threads to run

# Create another ThreadPoolExecutor to manage concurrent execution of varimp commands
with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
    # Submit all varimp commands to the executor for execution
    futures = [executor.submit(execute_command, command) for command in commands]
    
    # Iterate over the futures to ensure all commands have completed
    for future in futures:
        future.result()  # Wait for each future to complete

