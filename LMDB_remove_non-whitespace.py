# This script extracts whitespace STR data from PARSeq datasets.
# This script opens all the LMDB databases found in the given folders and then saves the data instances
#   which contain whitespace in the label. The data is re-organized into train, val, and test splits.
#   The new datasets have the same structure as the original ones.
# This script has the option to include some percentage of non-whitespace data instances to be included.
# Arguments:
#   output database: a folder will be created here to store the new LMBD
#   input folders: one or more folders that contains the LMBDs to be read
#   -subset: If this flag is included, 90% of data will be unused at random
#   --non_whitespace_prob: The probability that any non-whitespace data instance will be included.

# Standard library imports
import sys, os, re, glob, io, random, unicodedata, argparse

# Local/external library imports
import lmdb
from PIL import Image

# This assumes parseq and the folder containing this script are in the same folder
sys.path.append("../parseq")
from strhub.data.utils import CharsetAdapter

# Parse the arguments:
# 1. Path for the output files.
# 2. One or more input paths.
# 3. An optional 'subset' flag.
# 4. An optional probability for non-whitespace data.
parser = argparse.ArgumentParser()
parser.add_argument("output_path", help="Output file path")
parser.add_argument("input_paths", nargs="+")
parser.add_argument("-subset", action="store_true")
parser.add_argument("--prob", type=float, default=0, help="The probability for each non-whitespace data instance to be included (0-1, default is 0)")
args = parser.parse_args()

# Initialize the random seed
random.seed()

# Define the data split ratios
train_ratio = .8
val_ratio = .1
test_ratio = .1

# Create directories for the output LMDB files based on the data splits.
os.mkdir(args.output_path)
os.mkdir(os.path.join(args.output_path, "train"))
os.mkdir(os.path.join(args.output_path, "val"))
os.mkdir(os.path.join(args.output_path, "test"))

# Define paths for each data split.
train_split = os.path.join(args.output_path, "train/real")
val_split = os.path.join(args.output_path, "val/val")
test_split = os.path.join(args.output_path, "test/test")

# Some default settings, with normalization and size filtering disabled
normalize_unicode = False
min_image_dim = 0
max_label_len = 999999
charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ " # Note the space added
charset_adapter = CharsetAdapter(charset)

# Returns true if the given string contains whitespace
def contains_whitespace(s: str):
    pattern = r'\S\s+\S'
    match = re.search(pattern, s)
    return match is not None

# This size is large enough to store the PARSeq's original data
# The sizes are currently hardcoded making it hard to run this script on other data
train_map_size = 650000000 * train_ratio
val_map_size = 650000000 * val_ratio
test_map_size = 650000000 * test_ratio
# Add space for non-whitespace data
train_map_size += 127733 * 3100000 * args.prob
val_map_size += 127733 * 3100000 * args.prob
test_map_size += 127733 * 3100000 * args.prob
# Remove space when making a subset of the data
if args.subset:
    import random
    train_map_size *= .1
    val_map_size *= .1
    test_map_size *= .1
train_map_size = int(train_map_size)
val_map_size = int(val_map_size)
test_map_size = int(test_map_size)

# Open the LMBDs
train_env = lmdb.open(train_split, map_size=train_map_size)
val_env = lmdb.open(val_split, map_size=val_map_size)
test_env = lmdb.open(test_split, map_size=test_map_size)

# Create transactions for writing to the output databases
with train_env.begin(write=True) as train_txn, val_env.begin(write=True) as val_txn, test_env.begin(write=True) as test_txn:
    train_sample_index = 1  # Index for the new database
    val_sample_index = 1  # Index for the new database
    test_sample_index = 1  # Index for the new database

    # For each input folder, open it and find the databases
    for folder in args.input_paths:
        found_datasets = glob.glob(f"{folder}/**/data.mdb", recursive=True)

        # For each  database, open it and add its contents to the output database
        for mdb in found_datasets:
            db_path = os.path.abspath(os.path.join(mdb, os.pardir))
            db_name = os.path.relpath(db_path, start=folder)
            num_instances = 0

            # Open the found database
            env = lmdb.open(db_path)
            with env.begin() as txn:
                num_samples = int(txn.get('num-samples'.encode()))
                for index in range(1, num_samples + 1):  # lmdb indexing starts at 1
                    if args.subset and random.random() > .1:
                        continue
                    # Get label
                    label_key = f'label-{index:09d}'.encode()
                    label = txn.get(label_key).decode()

                    # Normalize unicode composites (if any) and convert to compatible ASCII characters
                    if normalize_unicode:
                        label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()

                    # Filter by length before removing unsupported characters. The original label might be too long.
                    if len(label) > max_label_len:
                        continue
                    label = charset_adapter(label)

                    # We filter out samples which don't contain any supported characters
                    if not label:
                        continue
            
                    # Filter images that are too small.
                    if min_image_dim > 0:
                        img_key = f'image-{index:09d}'.encode()
                        buf = io.BytesIO(txn.get(img_key))
                        w, h = Image.open(buf).size
                        if w < min_image_dim or h < min_image_dim:
                            continue

                    # Save data instaces which have whitespace in the label
                    # Randomly save other data instances too
                    if contains_whitespace(label) or random.random() < args.prob:
                        # Get image
                        image_key = f'image-{index:09d}'.encode()
                        image = txn.get(image_key)

                        # Write data to the new database
                        random_val = random.random()
                        if random_val < train_ratio:
                            # Get keys
                            new_label_key = f'label-{train_sample_index:09d}'.encode()
                            new_image_key = f'image-{train_sample_index:09d}'.encode()
                            train_txn.put(new_label_key, label.encode())
                            train_txn.put(new_image_key, image)
                            train_sample_index += 1
                        elif random_val < train_ratio + val_ratio:
                            # Get keys
                            new_label_key = f'label-{val_sample_index:09d}'.encode()
                            new_image_key = f'image-{val_sample_index:09d}'.encode()
                            val_txn.put(new_label_key, label.encode())
                            val_txn.put(new_image_key, image)
                            val_sample_index += 1
                        else:
                            # Get keys
                            new_label_key = f'label-{test_sample_index:09d}'.encode()
                            new_image_key = f'image-{test_sample_index:09d}'.encode()
                            test_txn.put(new_label_key, label.encode())
                            test_txn.put(new_image_key, image)
                            test_sample_index += 1

                        num_instances += 1
                print(f"{db_name} loaded, {num_instances} data instances found")
    # Write the number of samples
    train_txn.put('num-samples'.encode(), str(train_sample_index - 1).encode())
    val_txn.put('num-samples'.encode(), str(val_sample_index - 1).encode())
    test_txn.put('num-samples'.encode(), str(test_sample_index - 1).encode())

print(f"\nTraining dataset has {train_sample_index - 1} data instances")
print(f"Validation dataset has {val_sample_index - 1} data instances")
print(f"Testing dataset has {test_sample_index - 1} data instances")
print(f"Dataset saved to {os.path.abspath(args.output_path)}")