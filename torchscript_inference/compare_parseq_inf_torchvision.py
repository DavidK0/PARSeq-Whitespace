# This model takes two TorchScript PARSeq models and compares their outputs.
# It only prints results if the models differ.
# It takes two arguments, the two models.

import cv2
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

import os, argparse
import time
import numpy as np

# Parse the two arguments
parser = argparse.ArgumentParser()
parser.add_argument('model1')
parser.add_argument('model2')
args, unknown = parser.parse_known_args()

# reading configuration
charset_whitespace = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
charset_normal = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
BOS = '[B]'
EOS = '[E]'
PAD = '[P]'
specials_first = (EOS,)
specials_last = (BOS, PAD)

# whitespace
itos_whitespace = specials_first + tuple(charset_whitespace) + specials_last
stoi = {s: i for i, s in enumerate(itos_whitespace)}
eos_id_whitespace, bos_id, pad_id = [stoi[s] for s in specials_first + specials_last]
itos_whitespace = specials_first + tuple(charset_whitespace) + specials_last

# normal
itos = specials_first + tuple(charset_normal) + specials_last
stoi = {s: i for i, s in enumerate(itos)}
eos_id, bos_id, pad_id = [stoi[s] for s in specials_first + specials_last]
itos = specials_first + tuple(charset_normal) + specials_last


res_norm = T.Compose([
        T.ToTensor(),
        T.Resize((32, 128), T.InterpolationMode.BICUBIC, antialias=True),
        T.Normalize(0.5, 0.5)
        ])

def __call__(self, batch):
    batch = filter(lambda x: x is not None, batch)
    images, labels = zip(*batch)
    image_tensors = [self.res_norm(image) for image in images]
    image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

# decode the model output
def tokenizer_filter(probs, ids, whitespace_charset = False):
    ids = ids.tolist()

    try:
        if whitespace_charset:
            eos_idx = ids.index(eos_id_whitespace)
        else:
            eos_idx = ids.index(eos_id)
    except ValueError:
        eos_idx = len(ids)  # Nothing to truncate.
    # Truncate after EOS
    ids = ids[:eos_idx]
    probs = probs[:eos_idx + 1]  # but include prob. for EOS (if it exists)
    return probs, ids

def ids2tok(token_ids, whitespace_charset = False):
    if whitespace_charset:
        tokens = [itos_whitespace[i] for i in token_ids]
    else:
        tokens = [itos[i] for i in token_ids]
    return ''.join(tokens)

def decode(token_dists, whitespace_charset = False):
    """Decode a batch of token distributions.
    Args:
        token_dists: softmax probabilities over the token distribution. Shape: N, L, C
        raw: return unprocessed labels (will return list of list of strings)

    Returns:
        list of string labels (arbitrary length) and
        their corresponding sequence probabilities as a list of Tensors
    """
    batch_tokens = []
    batch_probs = []
    for dist in token_dists:
        probs, ids = dist.max(-1)  # greedy selection
        probs, ids = tokenizer_filter(probs, ids, whitespace_charset)
        tokens = ids2tok(ids, whitespace_charset)
        batch_tokens.append(tokens)
        batch_probs.append(probs)
    return batch_tokens, batch_probs

def parseq_batch_inference(images, model, eps, batch_size, device, whitespace_charset = False):
    N = len(images)
    # print("number of text {}".format(N))
    if N//batch_size == N/batch_size:
        n_batch = N//batch_size
    else:
        n_batch = N//batch_size + 1
    labels = []
    Total_time = 0
    for i in range(n_batch):
        # print(list(range((i-1)*batch_size,batch_size*i)))
        if batch_size*(i+1) <= N:
            input_holder = torch.cat(images[(i)*batch_size:batch_size*(i+1)], 0)
        else:
            input_holder = torch.cat(images[(i)*batch_size:N], 0)
        
        start = time.time()
        with torch.no_grad():  
            logits = model(input_holder.to(device))
        end = time.time()     
        Total_time += (end-start)
        pred = logits.softmax(-1)
        readings, confidences = decode(pred, whitespace_charset)
        
        for i, reading in enumerate(readings):
            # print(reading)
            confidence = confidences[i]
            labels.append("".join([reading[i] for i in range(len(reading)) if confidence[i] > eps]))
    # print("The total time cost is {}".format(Total_time))
    return labels

if __name__ == "__main__":
    img_folder = "img_demo"

    img_paths = [os.path.join(img_folder, x) for x in os.listdir(img_folder) if x.endswith("jpg")]
    txt_paths = [x.replace("jpg", "txt") for x in img_paths]

    images = [cv2.imread(x) for x in img_paths]
    
    images = [cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB) for cv2_image in images]
    image_tensors = [res_norm(image) for image in images]
    image_tensors = [t.unsqueeze(0) for t in image_tensors]
    
    model_1 = torch.jit.load(args.model1)
    model_2 = torch.jit.load(args.model2)

    labels_1 = parseq_batch_inference(image_tensors, model_1, eps=0.2, batch_size=16, device=torch.device("cuda"))
    labels_2 = parseq_batch_inference(image_tensors, model_2, eps=0.2, batch_size=16, device=torch.device("cuda"), whitespace_charset=True)
    
    # Compare and print only differing labels
    num_same = 0
    for img, label1, label2 in zip(img_paths, labels_1, labels_2):
        if label1 != label2:
            print(f"{img}: Model 1: {label1}\tModel 2: {label2}")
        else:
            num_same += 1
    print(f"Amount same: {num_same/len(img_paths):.2%} out of {len(img_paths)}")
