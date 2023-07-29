# This script loads a pre-trained PARSeq model and evaluates it on the given dataset.
# The given dataset must have the working folder as the first line and
#   it each line after that must have a space delimited tuple of (image_name, text_label)

import torch
import os,sys
from PIL import Image
import numpy
from strhub.data.module import SceneTextDataModule
from torchvision import transforms

from memory_profiler import profile

# Get the images location
if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
    print("Usage: {sys.argv[0]} [dataset]")
    sys.exit()
dataset = sys.argv[1]

# Load model and image transforms
print("Loading model...")
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

# Load the processed IIIT5K dataset file
with open(dataset) as input_file:
    images_root = input_file.readline().strip()
    data = [line.strip().split(" ") for line in input_file.readlines()]

# Track accuracy
performance = [0,0]
performance_case_insensitive = 0
performance_lower_alphabet_only = [0, 0]

# For each images in the give images folder
print("Evaluating images...")
amount_of_data = len(data)
for image_data in data[:amount_of_data]:
    # Open the image
    file_path = os.path.join(images_root, image_data[0])
    img = Image.open(file_path).convert('RGB')

    # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
    trans1 = transforms.Resize(parseq.hparams.img_size, transforms.InterpolationMode.BICUBIC)
    trans2 = transforms.ToTensor()
    trans3 = transforms.Normalize(0.5, 0.5)
    composed_transform = transforms.Compose([trans1, trans2, trans3])
    im_tensor = composed_transform(img).unsqueeze(0)

    # logits has shape=torch.Size([1, 26, 95]), 94 characters + [EOS] symbol
    logits = parseq(im_tensor) 

    # Greedy decoding
    pred = logits.softmax(-1)
    label, confidence = parseq.tokenizer.decode(pred)

    # Track accuracy
    performance[1] += 1
    if label[0] == image_data[1]:
        performance[0] += 1
    if label[0].lower() == image_data[1].lower():
        performance_case_insensitive += 1
    if label[0].isalpha():
        performance_lower_alphabet_only[1] += 1
        if label[0].lower() == image_data[1].lower():
            performance_lower_alphabet_only[0] += 1
        else:
            print(f"\n{label[0]}, {image_data[1]}")
    
    print(f"Evaluating images: {performance[1]/amount_of_data:.1%}", end="\r")
print()
print(f"{performance[1]} images evaluated, accuracy: {performance[0]/performance[1]:.1%}")
print(f"Case insensitive accuracy: {performance_case_insensitive/performance[1]:.1%}")
print(f"Small letters only accuracy: {performance_lower_alphabet_only[0]/performance_lower_alphabet_only[1]:.1%}")