# This script extracts a few sample images and labels from the LMBDs used to store PARSeq's data.
# This script takes two arguments:
#   The LMDB path
#   The folder to save the the images to

import sys, os, glob, io, random
import argparse

import lmdb
from PIL import Image

# This assumes parseq and the folder containing this script are in the same folder
sys.path.append("../parseq")
from strhub.data.utils import CharsetAdapter

# Parse the two arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_path")
parser.add_argument("output_path")
args = parser.parse_args()

# Some default settings, with normalization and size filtering disabled
normalize_unicode = False
min_image_dim = 0
max_label_len = 999999
charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ " # Note the space added
charset_adapter = CharsetAdapter(charset)

# Search for a LMDB database
found_dataset = glob.glob(f"{args.input_path}/**/data.mdb", recursive=True)[0]
db_path = os.path.abspath(os.path.join(found_dataset, os.pardir))

# Open the found database
env = lmdb.open(db_path)

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)
with env.begin() as txn:
    num_samples = int(txn.get('num-samples'.encode()))
    #for index in range(1, num_samples + 1):  # lmdb indexing starts at 1

    for i in range(10):
        index = random.randrange(num_samples)

        # Get label
        label_key = f'label-{index:09d}'.encode()
        label = txn.get(label_key).decode()

        # Get image
        image_key = f'image-{index:09d}'.encode()
        image_buf = txn.get(image_key)
        buf = io.BytesIO(image_buf)
        image = Image.open(buf).convert('RGB')

        # Save image
        image.save(os.path.join(args.output_path, f"{label}.jpg"))