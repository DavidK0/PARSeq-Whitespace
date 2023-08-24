# This script opens a LMDB used to store data for PARSeq.
# The data is analyzed and the amount of whitespace present is recorded.

import sys, os, re, glob
import lmdb

# This assumes parseq and the folder containing this script are in the same folder
sys.path.append("../parseq")
from strhub.data.utils import CharsetAdapter

# Check usage and get the dataset location
if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
    print("Usage: {sys.argv[0]} \[path_to_lmdb]")
    sys.exit()
data_path = sys.argv[1]

# The paths to the data splits
train_real_path = os.path.join(data_path, "train/real")
train_synth_path = os.path.join(data_path, "train/synth")
val_path = os.path.join(data_path, "val")
test_path = os.path.join(data_path, "test")

# Returns true if the given string contains whitespace
def contains_whitespace(s: str):
    pattern = r'\S\s+\S'
    match = re.search(pattern, s)
    return match is not None

def process_database(db_path):
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

            # Get image
            #image_key = f'image-{index:09d}'.encode()
            #image = txn.get(image_key).decode()

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

    # Close the database and return the lists
    env.close()
    return labels, filtered_index_list, whitespace_labels

for split_path in [train_real_path, train_synth_path, val_path, test_path]:
    for mdb in glob.glob(f"{split_path}/**/data.mdb", recursive=True):
        # Open the database
        db_path = os.path.abspath(os.path.join(mdb, os.pardir))

        print(f"Opening {os.path.relpath(db_path, start=data_path)}")
        labels, _, whitespace_labels = process_database(db_path)

        print(f"total labels: {len(labels)}")
        print(f"labels w/ whitespace: {len(whitespace_labels)}")
        print(f"{len(whitespace_labels)/len(labels):.2%}")
        print(whitespace_labels[:100])
        print()



