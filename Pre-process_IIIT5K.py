# This script loads the IIIT5K data set and prepares it for PARSeq

import sys, os
from scipy.io import loadmat

# Get the dataset location
if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
    print("Usage: {sys.argv[0]} \[path_to_IIIT5K\]")
    sys.exit()
IIIT5K = sys.argv[1]
train_file = os.path.join(IIIT5K, "traindata.mat")
test_file = os.path.join(IIIT5K, "testdata.mat")
if not os.path.isfile(train_file) or not os.path.isfile(test_file):
    print("The given folder is not IIIT5K")
    sys.exit()

# Load the data
print("Loading data...")
train_mat = loadmat(train_file)["traindata"][0]
test_mat = loadmat(test_file)["testdata"][0]

# Some extra unused data
#header = train_data["__header__"]
#version = train_data["__version__"]
#version = train_data["__globals__"]

# Process the data
print("Processing data...")
train_data = [(os.path.split(x[0][0])[1], x[1][0]) for x in train_mat]
test_data = [(os.path.split(x[0][0])[1], x[1][0]) for x in test_mat]

# Output data information
print(f"{len(train_mat)} training instances loaded")
print(f"{len(test_mat)} testing instances loaded")

# Save the training data
with open("IIIT5K_train_data.txt", "w") as output_file:
    output_file.write(os.path.join(IIIT5K, "train"))
    output_file.write("\n")
    for image_data in train_data:
        output_file.write(f"{image_data[0]} {image_data[1]}\n")
        
# Save the testing data
with open("IIIT5K_test_data.txt", "w") as output_file:
    output_file.write(os.path.join(IIIT5K, "test"))
    output_file.write("\n")
    for image_data in test_data:
        output_file.write(f"{image_data[0]} {image_data[1]}\n")
