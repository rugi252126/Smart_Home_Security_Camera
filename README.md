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


## 3. Hardware Setup
 - Laptop or Jetson nano or Jetson Tx2 or anything a like
 - USB camera


## 4. Camera

### 4.1. Logitech C310

![Logitech_c310](https://user-images.githubusercontent.com/47493510/98442767-bf3d7980-2141-11eb-9098-040037f16686.png)


### 4.2. Intel RealSense SR300
The depth frame captured from realsense camera is used to measure the distance between the face and the camera.
At short range, the measured distance is pretty much accurate.

![Intel_RealSense1](https://user-images.githubusercontent.com/47493510/98442683-04ad7700-2141-11eb-99da-c7f6e5b6abaa.png)

![Intel RealSense2](https://user-images.githubusercontent.com/47493510/98442754-9ae19d00-2141-11eb-9bdf-4db48373279c.png)


## 5. Useful Links
1. To setup the development environment (e.g. TensorFlow 2.0) and virtual environment, please follow below link
- https://www.pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/

2. Intel RealSense SDK 2.0 installation package for Ubuntu 16/18 LTS
- https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md

3. To compile the librealsense, please follow below link
- https://dev.intelrealsense.com/docs/compiling-librealsense-for-linux-ubuntu-guide

4. Windows 10 camera driver and Depth Camera Manager
- https://support.intelrealsense.com/hc/en-us/articles/360022951533-Windows-10-Issues-with-Intel-RealSense-Cameras-SR300-and-F200
- https://downloadcenter.intel.com/download/25044/Intel-RealSense-Depth-Camera-Manager


## 6. Troubleshooting

### Intel RealSense Camera
1. After driver installation [refer to Useful Links item #1] and if camera still not working, try to narrow down the problem. 
Try to setup the camera on Windows 10 [refer to Useful Links item #4] and test if the camera is working as expected. 
From here, you can tell if the problem is coming from the camera or from your Linux environment.
Setting-up the camera in Windows 10 is quite straight forward compare to Linux.

2. If camera is working fine on Windows 10 but not on Ubuntu Linux, try the link on item #3 [refer to Useful Links]

3. If camera is switching ON then OFF, it might be the power on the USB port is not enough.
Try to use USB hub that has dedicated power source. Connect the camera on the USB hub.

4. In case you are doing the development inside virtual environment and some packages are not found, try to install or re-install the missing
package inside virtual environment. Log-in to your virtual environment then install the missing package.



