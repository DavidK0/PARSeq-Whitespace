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

import argparse
import string
import sys
import re
from dataclasses import dataclass
from typing import List

import torch

from tqdm import tqdm

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

# Returns true if the given string contains whitespace
def contains_whitespace(s: str):
    pattern = r'\S\s+\S'
    match = re.search(pattern, s)
    return match is not None

@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float


def print_results_table(results: List[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0, 0, 0, 0)
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} |', file=file)
    if c.num_samples != 0:
        c.accuracy /= c.num_samples
        c.ned /= c.num_samples
        c.confidence /= c.num_samples
        c.label_length /= c.num_samples
    if len(results) > 1:
        print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|'.format('----', w=w), file=file)
        print(f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
              f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} |', file=file)


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

    charset_test = string.digits + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += string.punctuation
    #kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)

    hp = model.hparams
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
    test_set = sorted(set(test_set))

    results = {}
    non_whitespace_results = {}
    whitespace_results = {}
    max_width = max(map(len, test_set))
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        # all data
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0

        # no whitespace
        non_whitespace_total = 0
        non_whitespace_correct = 0
        non_whitespace_ned = 0
        non_whitespace_confidence = 0
        non_whitespace_label_length = 0
        
        # white-space
        whitespace_total = 0
        whitespace_correct = 0
        whitespace_ned = 0
        whitespace_confidence = 0
        whitespace_label_length = 0

        for batch_imgs, batch_labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            for img, label in zip(batch_imgs, batch_labels):
                img = img.unsqueeze(0).to(model.device)
                res = model.test_step((img, [label]), -1)['output']

                # all data
                total += res.num_samples
                correct += res.correct
                ned += res.ned
                confidence += res.confidence
                label_length += res.label_length

                if not contains_whitespace(label):
                    # no whitespace
                    non_whitespace_total += res.num_samples
                    non_whitespace_correct += res.correct
                    non_whitespace_ned += res.ned
                    non_whitespace_confidence += res.confidence
                    non_whitespace_label_length += res.label_length
                else:
                    # white-space
                    whitespace_total += res.num_samples
                    whitespace_correct += res.correct
                    whitespace_ned += res.ned
                    whitespace_confidence += res.confidence
                    whitespace_label_length += res.label_length

        # all data
        accuracy = 100 * correct / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length)

        # no whitespace
        non_whitespace_accuracy = 100 * non_whitespace_correct / non_whitespace_total if non_whitespace_total != 0 else 0
        non_whitespace_mean_ned = 100 * (1 - non_whitespace_ned / non_whitespace_total) if non_whitespace_total != 0 else 0
        non_whitespace_mean_conf = 100 * non_whitespace_confidence / non_whitespace_total if non_whitespace_total != 0 else 0
        non_whitespace_mean_label_length = non_whitespace_label_length / non_whitespace_total if non_whitespace_total != 0 else 0
        non_whitespace_results[name] = Result(name, non_whitespace_total, non_whitespace_accuracy, non_whitespace_mean_ned, non_whitespace_mean_conf, non_whitespace_mean_label_length)

        # white-space
        whitespace_accuracy = 100 * whitespace_correct / whitespace_total if whitespace_total != 0 else 0
        whitespace_mean_ned = 100 * (1 - whitespace_ned / whitespace_total) if whitespace_total != 0 else 0
        whitespace_mean_conf = 100 * whitespace_confidence / whitespace_total if whitespace_total != 0 else 0
        whitespace_mean_label_length = whitespace_label_length / whitespace_total if whitespace_total != 0 else 0
        whitespace_results[name] = Result(name, whitespace_total, whitespace_accuracy, whitespace_mean_ned, whitespace_mean_conf, whitespace_mean_label_length)


    result_groups = dict()

    if args.std:
        result_groups.update({'Benchmark (Subset)': SceneTextDataModule.TEST_BENCHMARK_SUB})
        result_groups.update({'Benchmark': SceneTextDataModule.TEST_BENCHMARK})
    if args.custom:
        result_groups.update({'Custom': SceneTextDataModule.TEST_CUSTOM})
    if args.new:
        result_groups.update({'New': SceneTextDataModule.TEST_NEW})

    with open(args.checkpoint + '.log.txt', 'w') as f:
        for out in [sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print("All data", file=out)
                print_results_table([results[s] for s in subset], out)
                print("Non-whitespace data", file=out)
                print_results_table([non_whitespace_results[s] for s in subset], out)
                print("Whitespace data", file=out)
                print_results_table([whitespace_results[s] for s in subset], out)
                print('\n', file=out)


if __name__ == '__main__':
    main()
