# PARSeq Experiments
In this repository I experiment with running a few datasets through [PARSeq](https://github.com/baudm/parseq). The datasets I use are the evaluation datasets from the [original PARSeq paper](https://link.springer.com/chapter/10.1007/978-3-031-19815-1_11). 

## The IIIT 5K-word Dataset
The first dataset I tried was the [IIIT 5K-word dataset](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset) \(IIIT5K\). Upon manual inspection I found that the training split of this dataset has at least 9 out of the 2000 images mis-labeled. For example the official logo for Flicker is included in the dataset, but it is labeled "FLICKER". All labels are case-insensitive and in all-caps. Many images contain punctuation which is not present in the label.

To evaluate IIIT5K I used three accuracies. The first is raw word accuracy without processing labels. The second is case-insensitive accuracy. The third is case-insensitive accuracy when only considering images in which PARSeq does not detect a non-alphabetic character. The performance of a pretrained PARSeq on the test split of IIIT5K \(3000 images\) with each of the accuracies is given in the table below.

|Raw  |Case-insensitive|Alphabet only|
|-----|----------------|-------------|
|51.4%|87.7%           |98.3%        |

The results show that PARSeq has terrible on the raw IIIT5K, but this is due to shortcomings of the dataset. When ignoring capitalization performance increases dramatically. Performance would likely increase even more if IIIT5K did not include un-labeled punctuation.

## The Curve Text Dataset
The second dataset I tried was the [Curve Text dataset](http://cs-chan.com/downloads_cute80_dataset.html) \(CUTE80\). This dataset appears to be for the task of scene text detection rather than scene text recognition. The dataset contains labels for the position of text in images but not for the text itself. I am unsure how to procede from here.

## The Street View Text Dataset
The third dataset I tried was the [Street View Text dataset](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset) \(SVT\). Like CUTE80, this dataset is intended to be used with scene text detection or uncropped scene text recognition. I pre-processed this dataset to produce word-level cropped images and evaluated PARSeq in the same manner as I did with IIIT5K. The three accuracies are given below.

|Raw  |Case-insensitive|Alphabet only|
|-----|----------------|-------------|
|70.5%|96.1%           |97.8%        |

The results show that, compared to IIIT5K, PARSeq had a high raw score amd case-insensitive score but lower alphabet-only score. This suggests that fewer images in SVT were challenging due to un-expected case differences or un-expected puncuation difference, but also that the images for which PARSeq did not detect punctuation were harder to recognize.
