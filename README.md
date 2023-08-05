# PARSeq Experiments
In this repository I experiment with evaluating [PARSeq](https://github.com/baudm/parseq) on three of the evaluation datasets from the [original PARSeq paper](https://link.springer.com/chapter/10.1007/978-3-031-19815-1_11). I also analyze the original training, validation, and testing datat of PARSeq to find whitespace in the labels. 

## The IIIT 5K-word Dataset
The first dataset I tried was the [IIIT 5K-word dataset](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset) \(IIIT5K\). Upon manual inspection I found that the training split of this dataset has at least 9 out of the 2000 images mis-labeled. For example the official logo for Flickr is included in the dataset, but it is labeled "FLICKER". All labels are case-insensitive and in all-caps. Many images contain punctuation which is not present in the label.

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

## Whitespace in PARSeq's data
By default, PARSeq's dataset loader will remove all whitespace from labels (ex. "Main St" becomes "MainSt"). Disabling the removal of whitespace could allow PARSeq to recognize spaces. To better understand the presence of whitespace in PARSeq's data I counted the number of data instances that countain whitespace in each of the non-synthetic training datasets. Those counts are given below, along with some examples.

|Dataset |Total Instances|w/ whitespace|Example            |
|--------|---------------|-------------|-------------------|
|ArT     |32,239         |2.19%        |Clinic of Dentistry|
|COCOv2.0|73,180         |1.57%        |World Acclaimed    |
|LSVT    |42,407         |14.4%        |Fashionable dress  |
|MLT19   |56,776         |0.00%        |N/A                |
|OpenVINO|1,929,716      |0.01%        |6.21 KM            |
|RCTW17  |10,467         |18.8%        |Please Save Water  |
|ReCTS   |26,291         |0.00%        |N/A                |
|TextOCR |817,520        |0.00%        |N/A                |
|Uber    |128,262        |18.2%        |Main St            |

In total there are about 3.1 million data instances but only 33.5k\(1.1%\) of them contain whitespace. 33.5k instances could be enough to fine-tune on just those images, but it also might be helpful to produce even more whitespace training data with the help of synthetic data generators like in [MJSynth](https://arxiv.org/abs/1406.2227) and [SynthText](https://arxiv.org/abs/1604.06646).
