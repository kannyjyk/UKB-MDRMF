from datetime import datetime  
# Get the current date and time
current_date = datetime.now()

# Format the current date as "MMDD", for example, "1027" for October 27th
formatted_date = current_date.strftime("%m%d")

path_prefix='../../results/Disease_diagnosis'
# Assign the formatted date to the variable 'folder', which can be used as a folder name
folder = formatted_date

# Initialize the image list with a single element 0
imglist = [0]

# Initialize the category list with categories numbered from 1 to 6
catlist = [1, 2, 3, 4, 5, 6]

rawXlocation = ''
 
cpumode=True
# Set the path where Xblock is stored
Xblocklocation = '../../results/xblock/'
def ava_gpus(g):
    """
    Returns an available GPU number based on the input index 'g'.
    Uses modulo operation to ensure the index wraps within the available GPU list.
    
    Parameters:
        g (int): Input GPU index
    
    Returns:
        int: Corresponding available GPU number
    """
    if cpumode:
        return -1
    ava = [1, 2, 3, 4, 5, 6, 7]  # Define a list of available GPU numbers
    g = g % len(ava)  # Ensure 'g' is within the range of the available GPU list
    return ava[g]  # Return the GPU number corresponding to the adjusted index

# Set the neural network depth to 2 layers
NNdepth = 2

# Set the neural network shape, for example, 100 neurons in the hidden layer
NNshape = 100