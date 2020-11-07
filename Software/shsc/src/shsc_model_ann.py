# USAGE
# python shsc_model_ann.py --cascade ../cascade/haarcascade_frontalface_default.xml --model ../model/shsc_ann.hdf5

# import the necessary packages
# img_to_array is used to orders the image channels correctly
from tensorflow.keras.preprocessing.image import img_to_array
# load_model is used to load the trained model
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import time
import imutils
import cv2


# constant declaration
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_DEPTH = 3
INPUT_SHAPE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH
NUM_OF_CLASSES = 3
ROI_PAD_SIZE = 30
BATCH_SIZE = 32
CAMERA_SOURCE_INDEX = 1


class SmartHomeSecurityCameraModel:
    def __init__(self, cascade, model, video=None):
        self.video_paths = video
        # load the face detector cascade and the trained model
        self.detector = cv2.CascadeClassifier(cascade)
        self.model = load_model(model)

        self.assign_class_label()

    def assign_class_label(self):
        # initialize the class label
        self.class_label = ["Edgi", "Rudy", "Unknown"]

    def check_input_source(self):
        if args.get("image", False):
            # grab the reference to video
            self.camera = cv2.VideoCapture(self.video_paths)
        else:
            # otherwise, grab the reference to the webcam
            self.camera = cv2.VideoCapture(CAMERA_SOURCE_INDEX)
        

    def start(self):
        self.check_input_source()

        # keep looping
        while True:
            # grab the current frame
            (grabbed, frame) = self.camera.read()

            # if we are viewing a video and we did not grab a frame, then we
            # have reached the end of the video
            if args.get("video") and not grabbed:
                break

            # resize the frame, convert it to grayscale, and then clone the
            # original frame so we can draw on it later in the program
            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frameClone = frame.copy()

            # detect faces in the input frame
            rects = self.detector.detectMultiScale(gray, scaleFactor=1.05, 
                minNeighbors=5, minSize=(34, 34),
                flags=cv2.CASCADE_SCALE_IMAGE)

            # loop over the face bounding boxes
            for (fX, fY, fW, fH) in rects:
                # extract the ROI of the face from the original image
                roi = frame[fY:fY + fH , fX:fX + fW]
                # check the size of the image and make sure it's within 
                # the desired size
                (h, w) = roi.shape[:2]
                if  h < IMAGE_HEIGHT or w < IMAGE_WIDTH:
                    # pad the ROI image
                    roi = frame[fY:fY + fH + ROI_PAD_SIZE, fX:fX + fW + ROI_PAD_SIZE]

                # resize it to a fixed pixels, and then prepare the
                # ROI for classification via the ANN
                roi = cv2.resize(roi, (IMAGE_WIDTH, IMAGE_HEIGHT))
                # convert the image to array form and
                # scale the raw pixel intensities to the range [0, 1]
                roi = img_to_array(roi)
                roi = roi.astype("float") / 255.0
                # flatten the image 
                roi = roi.reshape((1, INPUT_SHAPE))

                # begin prediction by supplying the ROI of image as input
                # predict method of model will return a list of probabilities for every
                # image in data – one probability for each class label, respectively. 
                predictions = self.model.predict(roi, batch_size=BATCH_SIZE)
                # Taking the argmax on axis=1 finds the index of the class label with the 
                # largest probability for each image
                #predictions = self.model.predict(roi, batch_size=32).argmax(axis=1)
                
                lp = 0
                for r in range(len(predictions)):
                    for c in range(NUM_OF_CLASSES):
                        # check the class that has the largest probability
                        # convert the predicted value into percentage
                        lp_raw = predictions[r][c] * 100
                        if lp_raw > lp:
                            lp = lp_raw
                            # save the class label index
                            idx = c

                # convert the largest probability value from numpy.float64 to
                # integer then into string format to be able to display on 
                # the output frame
                preds = str(int(lp))

                # display the label and bounding box rectangle on the output
                # frame
                img_txt = self.class_label[idx] + ' - ' + preds + '%'
                cv2.putText(frameClone, img_txt, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                # start point = (fX, fY)
                # end point = (fX + fW, fY + fH)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                    (255, 0, 0), 2)

            # show our detected faces along with sthe predicted labels
            cv2.imshow("Logitech", frameClone)

            # if the 'q' key is pressed, stop the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # cleanup the camera and close any open windows
        camera.release()
        cv2.destroyAllWindows()



# entry point if executed from file level
if __name__ == '__main__':
    # create an instance of argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--cascade", required=True,
        help="path to where the face cascade resides")
    ap.add_argument("-m", "--model", required=True,
        help="path to pre-trained model")
    ap.add_argument("-v", "--video", required=False,
        help="path to video file")
    args = vars(ap.parse_args())

    # create an instance of class SmartHomeSecurityCameraModel
    shscm = SmartHomeSecurityCameraModel(args["cascade"], args["model"], args["video"])
    shscm.start()
