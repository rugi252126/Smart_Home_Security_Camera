# USAGE
# python shsc_train_ann.py --dataset ../datasets/faces --model ../model/shsc_ann.hdf5 --output ../output/result/shsc_ann.png
# python shsc_train_ann.py --dataset ../datasets/faces --model ../model/shsc_ann.hdf5
# python shsc_train_ann.py --dataset ../datasets/faces

# import the necessary packages
# LabelBinarizer is used to one-hot encode the integer labels as vector labels
from sklearn.preprocessing import LabelBinarizer
# train_test_split is used to create training and testing splits
from sklearn.model_selection import train_test_split
# classification_report is used to help evaluate the performance of classifier
from sklearn.metrics import classification_report
# Sequential class indicates that the network will be feedforward and layers 
# will be added to the class sequentially, one on top of the other
from tensorflow.keras.models import Sequential
# Dense is the implementation of the fully-connected layers
from tensorflow.keras.layers import Dense
# SGD is used to optimize the parameters of the network
from tensorflow.keras.optimizers import SGD
# plot_model is used to visualize the architecture
from tensorflow.keras.utils import plot_model
from rugi_ai.preprocessing import ImageToArrayPreprocessor
from rugi_ai.preprocessing import ImagePreprocessor
from rugi_ai.datasets_loader import DatasetLoader
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
import argparse

# constant declaration
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_DEPTH = 3
INPUT_SHAPE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH
NUM_OF_CLASSES = 3
BATCH_SIZE = 32
NUM_OF_EPOCHS = 100
LEARNING_RATE = 0.01


class SmartHomeSecurityCamera:
    def __init__(self, dataset, model=None, output=None):
        # /path/to/dataset/{class}
        self.image_paths = dataset
        self.model_paths = model
        self.output_paths = output

    def start(self):
        shsc.load_images()
        shsc.images_preprocessing()
        shsc.assign_class_label()
        shsc.compile_model()
        shsc.train_network()
        shsc.save_network()
        shsc.evaluate_network()
        shsc.plot_results()

    def load_images(self):
        # grab the list of images from datasets
        # the paths of each images for each idividual classes
        # /path/to/dataset/{class}/{image}.jpg
        print("[INFO] loading images...")
        self.image_paths = list(paths.list_images(self.image_paths))

    def images_preprocessing(self):
        print("[INFO] preprocessing images...")
        # create an instances of image preprocessors
        # set the target size of the image(width and height)
        # note: ignoring the aspect ratio of the image 
        ip = ImagePreprocessor(IMAGE_WIDTH, IMAGE_HEIGHT)
        iap = ImageToArrayPreprocessor()

        # load the dataset from the disk then
        # resize the image, convert the image to array form and
        # scale the raw pixel intensities to the range [0, 1]
        dl = DatasetLoader(preprocessors=[ip, iap])
        (data, labels) = dl.load(self.image_paths, verbose=200)
        data = data.astype("float") / 255.0

        # after resizing each image, it is represented as a IMAGE_WIDTHxIMAGE_HEIGHTxIMAGE_DEPTH image, 
        # but in order to apply a standard neural network we must
        # first "flatten" the image to be simple list of IMAGE_WIDTHxIMAGE_HEIGHTxIMAGE_DEPTH = number of pixels
        # e.g. 32x32x3 = 3072
        data = data.reshape((data.shape[0], INPUT_SHAPE))

        # partition the data into training and testing splits using 75% of
        # data for training and the remaining 25% for testing
        # common training and testing data splits are 66.7% and 33.3%, 75% and 25%, 90% and 10%
        (self.trainX, self.testX, self.trainY, self.testY) = train_test_split(data, labels, 
            test_size=0.25, random_state=42)

        # convert the labels from integers to vectors
        self.trainY = LabelBinarizer().fit_transform(self.trainY)
        self.testY = LabelBinarizer().fit_transform(self.testY)

    def assign_class_label(self):
        # initialize the class label for shsc datasets
        self.class_label = ["edgi", "rudy", "unknown"]

    def compile_model(self):
        # initialize the optimizer and model
        print("[INFO] compiling model...")
        # define the INPUT_SHAPE-1024-512-NUM_OF_CLASSES architecture using Keras
        self.model = Sequential()
        self.model.add(Dense(1024, input_shape=(INPUT_SHAPE,), activation="relu"))
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(NUM_OF_CLASSES, activation="softmax"))
        # visualization graph to disk
        model_arch_paths = "../output/architecture/ruginet_ann.png"
        plot_model(self.model, to_file=model_arch_paths, show_shapes=True)

        # train the model using SGD
        sgd = SGD(LEARNING_RATE)
        # compile the model
        # loss categorical_crossentropy is commonly used for 
        # image classification where classes size is greater than 2
        self.model.compile(loss="categorical_crossentropy", optimizer=sgd,
            metrics=["accuracy"])

    def train_network(self):
        print("[INFO] training network...")
        self.H = self.model.fit(self.trainX, self.trainY, validation_data=(self.testX, self.testY),
            epochs=NUM_OF_EPOCHS, batch_size=BATCH_SIZE)

    def save_network(self):
        if self.model_paths is not None:
            # save the network to disk (.hdf5 format)
            print("[INFO] serializing network...")
            # network will be saved on the provided path
            self.model.save(self.model_paths)

    def evaluate_network(self):
        # evaluate the network
        print("[INFO] evaluating network...")
        predictions = self.model.predict(self.testX, batch_size=BATCH_SIZE)
        print(classification_report(self.testY.argmax(axis=1),
            predictions.argmax(axis=1), target_names=self.class_label))

    def plot_results(self):
        print("[INFO] plotting the results...")
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, NUM_OF_EPOCHS), self.H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, NUM_OF_EPOCHS), self.H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, NUM_OF_EPOCHS), self.H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, NUM_OF_EPOCHS), self.H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()

        if self.output_paths is not None:
            # save the plotted image
            plt.savefig(self.output_paths)

        plt.show()


# entry point if executed from file level
if __name__ == '__main__':
    # create an instance of argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
        help="path to imput datasets")
    ap.add_argument("-m", "--model", required=False,
        help="path of output model")
    ap.add_argument("-o", "--output", required=False,
        help="path to output of result image")
    args = vars(ap.parse_args())

    # create an instance of class SmartHomeSecurityCamera
    shsc = SmartHomeSecurityCamera(args["dataset"], args["model"], args["output"])
    shsc.start()



