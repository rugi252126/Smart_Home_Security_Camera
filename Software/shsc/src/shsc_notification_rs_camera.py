# USAGE
# python shsc_notification_rs_camera.py --cascade ../cascade/haarcascade_frontalface_default.xml --model ../model/shsc_ann.hdf5

# import related packages
import os
import smtplib
import imghdr
import argparse
import cv2
from email.message import EmailMessage
from shsc_model_ann_rs_camera import SmartHomeSecurityCameraModel


# get email and password defined in environment variables
# in Ubuntu 16.04, it can be done inside .profile and not in .bash_profile
# the password here is the one generated from google app
# in this way, it will create secure path between this script and the email account
EMAIL_USER = os.environ.get("EMAIL_ADDRESS")
EMAIL_PWD = os.environ.get("EMAIL_PASSWORD")
RCPT_1 = os.environ.get("RECIPIENT_1")
RCPT_2 = os.environ.get("RECIPIENT_2")


# create an instance of argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
    help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
    help="path to pre-trained model")
args = vars(ap.parse_args())

# create an instance of class SmartHomeSecurityCameraModel
shscm = SmartHomeSecurityCameraModel(args["cascade"], args["model"])
# check and initialize input source
shscm.check_input_source()

def send_message():
    # create message instance
    msg = EmailMessage()
    msg["Subject"] = "[NOTIFICATION] Unidentified person"
    msg["From"] = EMAIL_USER
    msg["To"] = RCPT_1, RCPT_2

    msg.set_content("Image attached!!!")

    # get image file name
    img_name = "unidentified_person.png"

    # attached the snapshot photo of the unknown person
    path = "../snapshot/" + img_name
    with open(path, "rb") as f:
        # get the image
        file_data = f.read()
        # get the extention of the image (e.g. .jpg, .png)
        file_type = imghdr.what(f.name)
        file_name = f.name
        # take only the name of the file
        file_name = path.split(os.path.sep)[-1]


    msg.add_attachment(file_data, maintype="image", subtype=file_type, 
        filename=file_name)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:

        smtp.login(EMAIL_USER, EMAIL_PWD)
        smtp.send_message(msg)

        print("[INFO] Message sent")

# start streaming
while True:
    shscm.start()

    # check if unidentified person is detected
    if shscm.get_prediction_status():
        # send notification
        send_message()
        # clear the flag to be ready for next check
        shscm.clear_prediction_status()

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# do clean exit
shscm.exit()



