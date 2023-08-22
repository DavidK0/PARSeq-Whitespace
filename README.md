## Whitespace in PARSeq's data
In this repository I experiment with improving PARSeq's ability to recognize whitespace. By default, PARSeq's dataset loader will remove all whitespace from labels (ex. "Main St" becomes "MainSt"). Disabling the removal of whitespace could allow PARSeq to recognize spaces. To better understand the presence of whitespace in PARSeq's data I first counted the number of data instances that countain whitespace in each of the non-synthetic training datasets. Those counts are given below, along with some examples.

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

I also counted the number of data instances with whitespace in PARSEq's validation and test split. The validation split contained only 55 data instances with whitespace, and the test split has 17061 instances, 16976 of which come from Uber. With all splits combines, there are 50.6k data instances with whitespace. If these were re-organized into new splits with a 80-10-10 ratio that would leave 40.5k for training and 5.1k for validation and training.

Next I tried training a PARSeq model from scratch on a subset of the whitespace data. I used 3858 images for training and 473 for validation. The model was trained for 2000 epochs using with the same settings as the original model, except that whitespace removal was disabled and the character set was extended to include the space character " ". 3858 is not enough images to train this model, but it is enough to show that the model's loss converges. Below is a graph of the training loss over training iterations.

<img src="https://github.com/DavidK0/PARSeq-Experiments/assets/9288945/496cd8d9-92c4-4918-bb5c-c3944968755e" alt="Alt Text" width="315" height="225">

And here is a graph of the validation loss.

<img src="https://github.com/DavidK0/PARSeq-Experiments/assets/9288945/e1bb495e-5ab3-44c1-b8e2-fbf9b6133ec9" alt="Alt Text" width="315" height="225">

The two graphs above show that the model is training on the whitespace data, but this small test run is too small for the model to generalize to the validation data.
