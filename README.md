# Smart Home Security Camera 

## 1. Introduction
The Smart Home Security Camera is designed and developed in a way that it can able to detect unknown individual within the valid detection range of the camera. It is intended to place in every entry point of the house(e.g. main door). Once an unknown individual is detected, the system will take a snapshot photo and send the information to those authorized recipient.

Deep learning for computer vision and image classification is used and served as the heart and soul of the system. I used supervised learning to train my deep learning network/model.

In deep learning, dataset is very important. Achieving higher accucary and less loss is not solely dependent on the network architecture or the correct use of hyperparameters but also on how good the datasets are.

For this project, I made the datasets manually using smartphone. I took several photos of me and my wife. I captured different photos from different angle of our face. And for the unknown class, I simply downloaded different faces from the internet. As this is a supervised learning, I assigned class label for each images.
In addition, I created different set of datasets that solely used for evaluation.

Overall, I have datasets used for training and testing and another set of datasets that used for evaluation.

Please note: For privacy purposes, I did not upload the photos that I used for my datasets.

## 2. Development Environment
 - OS: Ubuntu 16.04
 - IDE: Sublime Text
 - Python 3.5.2
 - tensorflow 2.0.0
 - tf.keras 2.2.4-tf (when Goolge announced TensorFlow 2.0, Keras is now official high-level API of tensorflow)
 - OpenCv 4.4.0
 - skimage 0.15.0
 - sklearn 0.22.2.post1
 - numpy 1.18.5
 - matplotlib 3.0.3
