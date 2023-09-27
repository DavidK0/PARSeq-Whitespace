#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a custom version of the original test script. In addition to outputing the usual metrics,
#   it also breaks those metrics down according to whether or not the label had whitespace.
# This script does not batch images for processing but instead process them individually, making it
#   noticably slower than the original.

import sys
import time
import re
import argparse

import torch
from torchvision import transforms as T

from tqdm import tqdm

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

# Returns true if the given string contains whitespace
def contains_whitespace(s: str):
    pattern = r'\S\s+\S'
    match = re.search(pattern, s)
    return match is not None

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--std', action='store_true', default=False, help='Evaluate on standard benchmark datasets')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--custom', action='store_false', default=True, help='Evaluate on custom personal datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)


    kwargs.update({'charset_test': "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "})
    #print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)

    hp = model.hparams
    #print([(x,hp[x]) for x in hp])
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation,
                                     remove_whitespace=False)

    test_set = tuple()
    if args.std:
        test_set = SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK
    if args.custom:
        test_set += SceneTextDataModule.TEST_CUSTOM
    if args.new:
        test_set += SceneTextDataModule.TEST_NEW

    test_set = sorted(set([test_set[0]]))

    results = {}
    non_whitespace_results = {}
    whitespace_results = {}
    max_width = max(map(len, test_set))
    for name, dataloader in datamodule.test_dataloaders(test_set).items():

        results[name] = {'total': 0, 'correct': 0}
        non_whitespace_results[name] = {'total': 0, 'correct': 0}
        whitespace_results[name] = {'total': 0, 'correct': 0}

        for batch_imgs, batch_labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):

            list_imgs = [img.unsqueeze(dim=0) for img in batch_imgs]
            batch_pred_labels = parseq_batch_inference(list_imgs , model, eps=0.2, batch_size=16, device=torch.device("cuda"))

            for img, label, predicted_label in zip(batch_imgs, batch_labels, batch_pred_labels):
                results[name]['total'] += 1

                if label == predicted_label:
                    results[name]['correct'] += 1
                
                if contains_whitespace(label):
                    whitespace_results[name]['total'] += 1
                    if label == predicted_label:
                        whitespace_results[name]['correct'] += 1
                else:
                    non_whitespace_results[name]['total'] += 1
                    if label == predicted_label:
                        non_whitespace_results[name]['correct'] += 1
                #else:
                #    print(label, predicted_label)
    for name in results:
        accuracy = results[name]['correct'] / results[name]['total']
        accuracy_non_whitespace = non_whitespace_results[name]['correct'] / non_whitespace_results[name]['total']
        accuracy_whitespace = whitespace_results[name]['correct'] / whitespace_results[name]['total']
        print(name)
        print(f"Overall {accuracy:.2%} out of {results[name]['total']}")
        print(f"Non whitespace {accuracy_non_whitespace:.2%} out of {non_whitespace_results[name]['total']}")
        print(f"Whitespace {accuracy_whitespace:.2%} out of {whitespace_results[name]['total']}")
        print()

# reading configuration
charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
BOS = '[B]'
EOS = '[E]'
PAD = '[P]'
specials_first = (EOS,)
specials_last = (BOS, PAD)
itos = specials_first + tuple(charset) + specials_last
stoi = {s: i for i, s in enumerate(itos)}
eos_id, bos_id, pad_id = [stoi[s] for s in specials_first + specials_last]
itos = specials_first + tuple(charset) + specials_last

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

if __name__ == '__main__':
    main()