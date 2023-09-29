## Whitespace in PARSeq's data
In this repository train a PARSeq from scratch so that it can predict whitespace in addition to the original set of 94 characters. By default, PARSeq's dataset loader will remove all whitespace from labels (ex. "Main St" becomes "MainSt"). Disabling the removal of whitespace could allow PARSeq to recognize spaces. To better understand the presence of whitespace in PARSeq's data I first counted the number of data instances that countain whitespace in each of the non-synthetic training datasets. Those counts are given below, along with some examples.

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

In total there are about 3.1 million data instances but only 33.5k\(1.1%\) of them contain whitespace. I also counted the number of data instances with whitespace in PARSEq's validation and test split. The validation split contained only 55 data instances with whitespace, and the test split has 17061 instances, 16976 of which come from Uber.

# Methadology


# Results
![image](https://github.com/DavidK0/PARSeq-Whitespace/assets/9288945/d2a95537-0edb-421c-9c56-c9daa13582e7)

The performance of my model when tested on non-whitespace data is quite comparable to the pre-trained model. My model is within 1 percentage point of the pre-trained model's accuracy for the first two experiments.

However when it comes to whitespace data my model's accuracy is only 7.32%. One explanation for this low accuracy is that only a very small portion of the training (about 1%) has whitespace. Also when I inspect some of my model's predictions on whitespace images, I see some unreadable images like the top two here: 

![image](https://github.com/DavidK0/PARSeq-Whitespace/assets/9288945/f85f8b2e-9c05-4273-ab56-3cb5b9847f77)

The bottom two examples show that the model is able to recognize whitespace on easier examples.

# Future Endeavours
It might be helpful to produce even more whitespace training data with the help of synthetic data generators like in [MJSynth](https://arxiv.org/abs/1406.2227) and [SynthText](https://arxiv.org/abs/1604.06646).

Lastly, I tried fine tuning my model on just whitespace data, but that caused overfitting and after a about 30 epochs the whitespace accuracy was up to 57% but the non-whitespace accuracy was down to 48%.
