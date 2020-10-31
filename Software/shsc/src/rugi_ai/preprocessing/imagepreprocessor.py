# import the necessary packages
import cv2


"""
Class that handles the resizing of images according to desired size.
OpenCV is used here to do the trick.
"""
class ImagePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect ratio
        return cv2.resize(image, (self.width, self.height),
            interpolation=self.inter)
