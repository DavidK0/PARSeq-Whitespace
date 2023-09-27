import cv2
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

import os
import time
import numpy as np

# reading configuration
charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
BOS = '[B]'
EOS = '[E]'
PAD = '[P]'
specials_first = (EOS,)
specials_last = (BOS, PAD)
itos = specials_first + tuple(charset) + specials_last
stoi = {s: i for i, s in enumerate(itos)}
eos_id, bos_id, pad_id = [stoi[s] for s in specials_first + specials_last]
itos = specials_first + tuple(charset) + specials_last


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
def tokenizer_filter(probs, ids):
    ids = ids.tolist()
    try:
        eos_idx = ids.index(eos_id)
    except ValueError:
        eos_idx = len(ids)  # Nothing to truncate.
    # Truncate after EOS
    ids = ids[:eos_idx]
    probs = probs[:eos_idx + 1]  # but include prob. for EOS (if it exists)
    return probs, ids

def ids2tok(token_ids):
    tokens = [itos[i] for i in token_ids]
    return ''.join(tokens)

def decode(token_dists):
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
        probs, ids = tokenizer_filter(probs, ids)
        tokens = ids2tok(ids)
        batch_tokens.append(tokens)
        batch_probs.append(probs)
    return batch_tokens, batch_probs

# def img_preprocess(img_orig, nH=32, nW=128):
#     # img_orig is a numpy array from cv2.imread(img_path)
#     original_image = img_orig[:, :, ::-1]
#     try:
#         resized_image = cv2.resize(original_image, (nW, nH), interpolation=cv2.INTER_CUBIC)
#     except:
#         resized_image = np.zeros([32, 128, 3])
#         print("resize failed!")
#     resized_image = (resized_image - 0.5 * 255) / (0.5 * 255)

#     image = torch.as_tensor(resized_image.astype("float32").transpose(2, 0, 1)).unsqueeze(0)

#     return image


def parseq_batch_inference(images, model, eps, batch_size, device):
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
        readings, confidences = decode(pred)
        
        for i, reading in enumerate(readings):
            # print(reading)
            confidence = confidences[i]
            labels.append("".join([reading[i] for i in range(len(reading)) if confidence[i] > eps]))
    # print("The total time cost is {}".format(Total_time))
    return labels

if __name__ == "__main__":
    img_folder = "img_demo"
    parseq_path = "parseq_cuda_b16_torchscript.pt"
    
    img_paths = [os.path.join(img_folder, x) for x in os.listdir(img_folder) if x.endswith("jpg")]
    txt_paths = [x.replace("jpg", "txt") for x in img_paths]

    images = [cv2.imread(x) for x in img_paths]
    
    images = [cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB) for cv2_image in images]
    image_tensors = [res_norm(image) for image in images]
    image_tensors = [t.unsqueeze(0) for t in image_tensors]
    
    model = torch.jit.load(parseq_path)

    print(image_tensors[0].shape)
    labels = parseq_batch_inference(image_tensors, model, eps=0.2, batch_size=16, device=torch.device("cuda"))

    # print(labels)
    for i, t in enumerate(labels):
        with open(txt_paths[i], "w") as f:
            print(t)
            f.writelines(t)
            