# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array


"""
Class that handles the arrangement of image channels ordered.
Using tensorflow as backend, channel ordered is by default channels last.
It would be good to have an extra handler in software to make the 
software design more flexible when adapting to other channels ordered(e.g. channels first)
"""
class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # apply the Keras utility function that correctly rearranges
        # the dimensions of the image
        return img_to_array(image, data_format=self.dataFormat)