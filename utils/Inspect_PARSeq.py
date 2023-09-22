# This script is used to qualitatively inspect PARSeq's performance.
# It takes three arguments:
#   A pre-trained PARSeq .ckpt
#   An input folder containing iamges
# The model is loaded and all images in the input folder are read by PARSeq.

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
parser.add_argument("images_folder", help="A folder containing images")
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
    
# Get a list of all files in the input folder
file_list = os.listdir(args.images_folder)
    
# Filter out non-image files based on file extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
image_files = [f for f in file_list if any(f.lower().endswith(ext) for ext in image_extensions)]
    
# Iterate through image files and open them using PIL
for image_file in image_files:
    # Load image and prepare for input
    image_path = os.path.join(args.images_folder, image_file)
    image = Image.open(image_path).convert('RGB')
    image = img_transform(image).unsqueeze(0).to(args.device)

    # Get predicted label
    p = model(image).softmax(-1)
    pred, p = model.tokenizer.decode(p)

    # Output
    image_name = os.path.splitext(os.path.split(image_path)[1])[0]
    print(f'{image_name.ljust(30)}\t{pred[0]}')