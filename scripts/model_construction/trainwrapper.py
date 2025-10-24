import subprocess
from concurrent.futures import ThreadPoolExecutor
import os
import time
import argparse
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
from config import *


def modelchar(x):
    if x >= 0 and x <= 9:
        return str(x)
    elif x >= 10:
        return chr(65 + x - 10)


def cmdfull(category, model, hyperparameter, index_number, location, gpu):
    return f"python ./trainer.py {category} {model} {hyperparameter} {index_number} {gpu} {location}\n"


def cmdimp(category, model, hyperparameter, index_number, location, gpu):
    return f"python ./trainerimpute.py {category} {model} {hyperparameter} {index_number} {gpu} {location}\n"


def cmdpri(category, model, hyperparameter, index_number, location, gpu):
    return f"python ./trainerpriority.py {category} {model} {hyperparameter} {index_number} {gpu} {location}\n"


def cmdsurvpri(category, model, hyperparameter, index_number, location, gpu):
    return f"python ./trainersurvpriority.py {category} {model} {hyperparameter} {index_number} {gpu} {location}\n"


def cmdsc(category, model, hyperparameter, index_number, location, gpu):
    return f"python ./trainersc.py {category} {model} {hyperparameter} {index_number} {gpu} {location}\n"


def cmdsurv(category, model, hyperparameter, index_number, location, gpu):
    return f"python ./trainersurv.py {category} {model} {hyperparameter} {index_number} {gpu} {location}\n"


def cmdsurvsc(category, model, hyperparameter, index_number, location, gpu):
    return f"python ./trainersurvsc.py {category} {model} {hyperparameter} {index_number} {gpu} {location}\n"


def cmdspecial(category, model, hyperparameter, index_number, location, gpu):
    return f"python ./trainerspecialdata.py {category} {model} {hyperparameter} {index_number} {gpu} {location}\n"


def cmdsurvspecial(category, model, hyperparameter, index_number, location, gpu):
    return f"python ./trainersurvspecial.py {category} {model} {hyperparameter} {index_number} {gpu} {location}\n"


def execute_command(command):
    print(f"Executing command: {command},total {len(commands)}")
    process = subprocess.Popen(command, shell=True)
    process.wait()
    print(f"Command finished: {command}")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select the task you want, input the according number, starts from 0: dummy, standard, survival, priority, survival priority, sub category, survival sub category, importance, special")
    parser.add_argument('numbers', type=int, nargs='+', help="Input multiple integers to select the tasks you want to run")
    args = parser.parse_args()
    numbers = args.numbers
    commands = []
    gpu = 0
    list_full=['dummy','standard','survival','priority','survival priority','sub category','survival sub category','importance','special']
    list_of_training=[]
    for i in numbers:
        list_of_training.append(list_full[i])
        print(f"Task {list_full[i]} selected")
    if 'dummy' in list_of_training:
        dummy=7
        for category in [dummy]:# specify the category of the dataset here, 7 for dummy dataset, see above output for more details
            for model in [1]:# choose your model, 0 for POPDX, 1 for FCNN, 2 for logit, 3 for MITH, note that only 1 and 2 are available for the dummy dataset since other two require pretrained embeddings
                for hyperparameterarameter in range(15):# choose the hyperparameterarameter setting, use range(15) for all parameters, see details in trainer.py
                    if category ==7:
                        index_number=3
                    else:
                        index_number=0
                    c=cmdfull(category,model,hyperparameterarameter,index_number,folder,ava_gpus(gpu))# this generates the shell command to use the standard (cmdfull) training model, see more in trainwrapper.py
                    commands.append(c)# add to the list of commands to be executed
                    gpu+=1# call next gpu to avoid overloading

        # the following code is for the survival analysis, with the same structure as above
        for category in [dummy]:
            for model in [1]:
                if category ==7:
                    index_number=3
                else:
                    index_number=0
                c=cmdsurv(category,model,0,index_number,folder,ava_gpus(gpu))
                commands.append(c)
                gpu+=1

    categories = [1,2,3,4,5,6]
    if 'special' in list_of_training:
        for cat in categories:
            for label_category in [1]:
                for hyperparameter in [0]:
                    for index_number in [0]:
                        for imputation_method in range(7):
                            c = cmdspecial(
                                cat,
                                label_category,
                                hyperparameter,
                                index_number,
                                f"{folder}_{imputation_method}",
                                ava_gpus(gpu),
                            )
                            commands.append(c)
                            gpu += 1
    if 'survival' in list_of_training:
        for cat in categories:
            for model in [0, 1, 2, 3]:
                for hyperparameter in [0]:
                    for index_number in [0]:
                        c = cmdsurv(
                            cat, model, hyperparameter, index_number, folder, ava_gpus(gpu)
                        )
                        commands.append(c)
                        gpu += 1
    if 'standard' in list_of_training:
        for cat in categories:
            for model in [0, 1, 2, 3]:
                for hyperparameter in  range(15):
                    for index_number in [0]:
                        c = cmdfull(
                            cat, model, hyperparameter, index_number, folder, ava_gpus(gpu)
                        )
                        commands.append(c)
                        gpu += 1
    if 'survival priority' in list_of_training:
        for cat in categories:
            for model in [1]:
                for hyperparameter in [1, 2]:
                    for index_number in [0]:
                        c = cmdsurvpri(
                            cat, model, hyperparameter, index_number, folder, ava_gpus(gpu)
                        )
                        commands.append(c)
                        gpu += 1
    if 'priority' in list_of_training:
        for cat in categories:
            for model in [1]:
                for hyperparameter in [1, 2]:
                    for index_number in [0]:
                        c = cmdpri(
                            cat, model, hyperparameter, index_number, folder, ava_gpus(gpu)
                        )
                        commands.append(c)
                        gpu += 1
    if 'sub category' in list_of_training:
        for cat in categories:
            for label_category in range(30):
                for hyperparameter in [1]:
                    for index_number in [0]:
                        c = cmdsc(
                            cat,
                            label_category,
                            hyperparameter,
                            index_number,
                            folder,
                            ava_gpus(gpu),
                        )
                        commands.append(c)
                        gpu += 1
    if 'importance' in list_of_training:
        for cat in categories:
            for model in [1]:
                for hyperparameter in [1, 2, 3, 4, 5, 6]:
                    for index_number in [0]:
                        c = cmdimp(
                            cat, model, hyperparameter, index_number, folder, ava_gpus(gpu)
                        )
                        commands.append(c)
                        gpu += 1
    if 'survival sub category' in list_of_training:
        for cat in categories:
            for label_category in range(30):
                for hyperparameter in [1]:
                    for index_number in [0]:
                        c = cmdsurvsc(
                            cat,
                            label_category,
                            hyperparameter,
                            index_number,
                            folder,
                            ava_gpus(gpu),
                        )
                        commands.append(c)
                        gpu += 1


    max_concurrent = 1

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [executor.submit(execute_command, command) for command in commands]
        for future in futures:
            future.result()
    print("All commands executed.")
