# This script is used to qualitatively inspect PARSeq's performance.
# It takes three arguments:
#   A pre-trained PARSeq .ckpt
#   An LMBD containing data readable by PARSeq
#   The folder to save the the images and labels to
# The model is loaded and 10 random images are selected to be read by PARSeq.
#   Those 10 images, the correct label, and PARSeq's predicted label and saved in a the given folder.

import sys, os, glob, io, random, argparse

import lmdb
from PIL import Image

# This assumes parseq and the folder containing this script are in the same folder
sys.path.append("../parseq")
from strhub.models.utils import load_from_checkpoint, parse_model_args
from strhub.data.utils import CharsetAdapter
from strhub.data.module import SceneTextDataModule

# Parse the two arguments
parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
parser.add_argument("lmbd_path", help="An LMDB with PARSeq data")
parser.add_argument("output_path", help="The path to save the results to")
parser.add_argument('--device', default='cuda')
args, unknown = parser.parse_known_args()
kwargs = parse_model_args(unknown)

# Load the model
model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

# Some default settings, with normalization and size filtering disabled
normalize_unicode = False
min_image_dim = 0
max_label_len = 999999
charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ " # Note the space added
charset_adapter = CharsetAdapter(charset)

# Search for a LMDB database
found_dataset = glob.glob(f"{args.lmbd_path}/**/data.mdb", recursive=True)[0]
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
        image = img_transform(image).unsqueeze(0).to(args.device)

        # Make prediction
        p = model(image).softmax(-1)
        pred, p = model.tokenizer.decode(p)
        print(f'{index}\t{label.ljust(30)}\t{pred[0]}')

        # Save image
        #image.save(os.path.join(args.output_path, f"{label}.jpg"))