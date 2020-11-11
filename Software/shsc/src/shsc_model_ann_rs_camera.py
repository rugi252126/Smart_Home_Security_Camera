# USAGE
# python shsc_model_ann_rs_camera.py --cascade ../cascade/haarcascade_frontalface_default.xml --model ../model/shsc_ann.hdf5

# import the necessary packages
# img_to_array is used to orders the image channels correctly
from tensorflow.keras.preprocessing.image import img_to_array
# load_model is used to load the trained model
from tensorflow.keras.models import load_model
import pyrealsense2 as rs
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
CAMERA_SOURCE_INDEX = 0
MIN_VALID_DISTANCE_IN_CM = 30
NUM_OF_UNKNOWN_PREDICTION = 5


class SmartHomeSecurityCameraModel:
    def __init__(self, cascade, model, video=None):
        self.video_paths = video
        # load the face detector cascade and the trained model
        self.detector = cv2.CascadeClassifier(cascade)
        self.model = load_model(model)

        self.invalid_dist_status_b = False
        self.unknown_preds_status_b = False
        self.unknown_preds_ctr = 0
        self.dist_to_center = 0

        self.assign_class_label()

    def assign_class_label(self):
        # initialize the class label
        self.class_label = ["Edgi", "Rudy", "Unknown"]

    def check_input_source(self):
        if self.video_paths is not None: #args.get("video", False):
            # grab the reference to video
            self.camera = cv2.VideoCapture(self.video_paths)
        else:
            # otherwise, grab the reference to the Intel RealSense Camera
            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # Start streaming
            #self.profile = self.pipeline.start(config)
            self.pipeline.start(config)

            # Skip 5 first frames to give the Auto-Exposure time to adjust
            for x in range(5):
                self.pipeline.wait_for_frames()
        
    def evaluate_prediction(self, dist_meter, preds, snapshot):
        if self.dist_to_center < MIN_VALID_DISTANCE_IN_CM:
            self.invalid_dist_status_b = True
        else:
            # further evaluation will only be done if the detected person is 
            # within valid range
            self.invalid_dist_status_b = False

            if preds == "Unknown":
                if self.unknown_preds_ctr < NUM_OF_UNKNOWN_PREDICTION:
                    self.unknown_preds_ctr +=1
                else:
                    self.unknown_preds_status_b = True
                    print("[INFO] Detected unknown person")
                    # TODO: save the snapshot
                    #path = "../snapshot/unidentified_person.jpg"
            else:
                self.unknown_preds_ctr = 0

    def start(self):
        try:
            # Wait for a coherent pair of frames: depth and color
            frameset = self.pipeline.wait_for_frames()
            depth_frame = frameset.get_depth_frame()
            color_frame = frameset.get_color_frame()

            # check if depth and color frames are available
            if not depth_frame or not color_frame:
                pass
            else:
            #if depth_frame and color_frame:
                # RGB Data
                # start with accessing the color componnent of the frameset
                color = np.asanyarray(color_frame.get_data())
                #cv2.imshow("RGB Data", color)

                # Depth Data
                # visualize the depth map captured by the RealSense camera
                colorizer = rs.colorizer()
                colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                #cv2.imshow("Colorized Depth Data", colorized_depth)

                # Stream Alignment
                # align in case that the two frames are not captured from the same physical viewport
                # and then combine them into a single RGBD image
                # align depth data to color viewport
                # and then create alignment primitive with color as its target stream
                align = rs.align(rs.stream.color)
                frameset = align.process(frameset)

                # after alignment, update color and depth frames
                aligned_depth_frame = frameset.get_depth_frame()
                colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
                #cv2.imshow("Aligned depth", colorized_depth)

                # Show the two frames together
                #images = np.hstack((color, colorized_depth))
                #cv2.imshow("Aligned images", images)

                # make a copy of the color frame
                frame = color.copy()

                # resize the frame, convert it to grayscale, and then clone the
                # original frame so we can draw on it later in the program
                frame = imutils.resize(frame, width=500)
                colorized_depth = imutils.resize(colorized_depth, width=500)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # make another copy of resized frame to be used later
                frameClone = frame.copy()

                # detect faces in the input frame
                rects = self.detector.detectMultiScale(gray, scaleFactor=1.05, 
                    minNeighbors=5, minSize=(34, 34),
                    flags=cv2.CASCADE_SCALE_IMAGE)

                # init ROI status
                valid_roi_b = False

                # loop over the face bounding boxes
                for (fX, fY, fW, fH) in rects:
                    # ROI is extracted successfully
                    valid_roi_b = True

                    # extract the ROI of the face from the original image
                    roi = frame[fY:fY + fH , fX:fX + fW]

                    # check the size of the image and make sure it's within 
                    # the desired size
                    (h, w) = roi.shape[:2]
                    if  h < IMAGE_HEIGHT or w < IMAGE_WIDTH:
                        # pad the ROI image
                        roi = frame[fY:fY + fH + ROI_PAD_SIZE, fX:fX + fW + ROI_PAD_SIZE]

                    # Find the center x,y coordinates from ROI
                    # fW = fH as this is the size of the rectangle
                    center_x = fX + (fW // 2)
                    center_y = fY + (fH // 2)
                    # Once center coordinate is located, get the distance using depth image
                    dist_meter = aligned_depth_frame.get_distance(center_x, center_y)
                    # Intel RealSense camera the distance is in meter form
                    # convert the distance to centimeter and then to integer type
                    self.dist_to_center = int(dist_meter * 100)

                    # resize it to a fixed pixels, and then prepare the
                    # ROI for classification via the ANN
                    roi = cv2.resize(roi, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    roi_copy = roi
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

                    # add distance information
                    if self.invalid_dist_status_b:
                        dist_txt = "Invalid distance"
                    else:
                        dist_txt = "Distance: {}".format(self.dist_to_center) + "cm"

                    cv2.putText(colorized_depth, dist_txt, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    # create bounding box on the colorized depth image
                    cv2.rectangle(colorized_depth, (fX, fY), (fX + fW, fY + fH),
                    (255, 0, 0), 2)

                    # evaluate further the predicted data
                    self.evaluate_prediction(self.dist_to_center, self.class_label[idx], frameClone)

                if valid_roi_b:
                    # Stack both images together horizontally
                    images = np.hstack((frameClone, colorized_depth))

                    # show our detected faces along with predicted labels
                    cv2.namedWindow("Intel RealSense", cv2.WINDOW_AUTOSIZE)
                    cv2.imshow("Intel RealSense", images)

        finally:
            pass
            

    def get_prediction_status(self):
        return self.unknown_preds_status_b

    # once snapshot has been sent, clear the flag
    def clear_prediction_status(self):
        self.unknown_preds_status_b = False  

    def exit(self):  
        # Stop streaming
        self.pipeline.stop()
        # cleanup the camera and close any open windows
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
    shscm.check_input_source()

    while True:
        shscm.start()

        # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break      
    
    shscm.exit()