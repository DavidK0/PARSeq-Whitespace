# This script is used to count how many data instances there are in a PARSeq data LMDB,
#   and how many of those data instances have whice space.

# This script takes one argument, a folder containing LMBDs with data readable by PARSeq

import sys, os, re, glob
import lmdb

# This assumes parseq and the folder containing this script are in the same folder
sys.path.append("../parseq")
from strhub.data.utils import CharsetAdapter

# Check usage and get the dataset location
if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
    print("Usage: {sys.argv[0]} \[PARSeq data path]")
    sys.exit()
data_path = sys.argv[1]

# Returns true if the given string contains whitespace
def contains_whitespace(s: str):
    pattern = r'\S\s+\S'
    match = re.search(pattern, s)
    return match is not None

# Opens one database and returns information about the number of data instances
def process_database(db_path):
    # Initialize variables for image size tracking
    total_image_size = 0
    num_images = 0

    # Some default settings, with normalization and size filtering disabled
    normalize_unicode = False
    min_image_dim = 0
    max_label_len = 999999
    charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
    charset_adapter = CharsetAdapter(charset)

    labels = []
    filtered_index_list = []

    whitespace_labels = []

    env = lmdb.open(db_path)
    with env.begin() as txn:
        num_samples = int(txn.get('num-samples'.encode()))
        for index in range(1, num_samples + 1): # lmdb indexing starts at 1
            # Get label
            label_key = f'label-{index:09d}'.encode()
            label = txn.get(label_key).decode()

            # Get image and track the size
            image_key = f'image-{index:09d}'.encode()
            image_data = txn.get(image_key)
            image_size = len(image_data) # size in bytes
            total_image_size += image_size
            num_images += 1

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
            #if min_image_dim > 0:
            #    img_key = f'image-{index:09d}'.encode()
            #    buf = io.BytesIO(txn.get(img_key))
            #    w, h = Image.open(buf).size
            #    if w < min_image_dim or h < min_image_dim:
            #        continue

            labels.append(label)
            filtered_index_list.append(index)

            if contains_whitespace(label):
                whitespace_labels.append(label)

    # show the average iamges size
    if num_images:
        avg_image_size_kib = total_image_size / num_images / 1024  # Convert bytes to KiB
        print(f"Average image size: {avg_image_size_kib:.2f} KiB")

    # Close the database and return the lists
    env.close()
    return labels, filtered_index_list, whitespace_labels

for mdb in glob.glob(f"{data_path}/**/data.mdb", recursive=True):
    # Open the database
    db_path = os.path.abspath(os.path.join(mdb, os.pardir))

    print(f"Opening {os.path.relpath(db_path, start=data_path)}")
    labels, _, whitespace_labels = process_database(db_path)

    print(f"total labels: {len(labels)}")
    print(f"  with whitespace: {len(whitespace_labels)} ({len(whitespace_labels)/len(labels):.2%})")
    print(f"  without whitespace: {len(labels) - len(whitespace_labels)} ({(len(labels) - len(whitespace_labels))/len(labels):.2%})")
    #print(whitespace_labels[:100]) # print a sample of labels
    print()