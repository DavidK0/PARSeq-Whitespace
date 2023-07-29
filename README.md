# PARSeq-Experiments
In this repository I experiment with running a few datasets through PARSeq.

## The IIIT 5K-word dataset
The first dataset I tried was the [IIIT 5K-word dataset](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset) \(IIIT5K\). Upon manual inspection I found that the training split of this dataset has at least 9 out of the 2000 images mis-labeled. For example the official logo for Flicker is included in the dataset, but it is labeled "FLICKER". All labels are case-insensitive and in all-caps. Many images contain punctuation which is not present in the label.

To evaluate IIIT5K I used three accuracies. The first is raw word accuracy without processing labels. The second is case-insensitive accuracy. The third is case-insensitive accuracy when only considering images in which PARSeq does not detect a non-alphabetic character. The performance of IIIT5K on each of the accuracies are given in the table below.
