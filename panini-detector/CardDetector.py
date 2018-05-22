############## Python-OpenCV Playing Card Detector ###############
#
# Author: Lorenz Haenggi
# Date: 5/20/18
# Description: Python script to detect and identify panini playing cards in a video stream and detect the number
# from a PiCamera or USB video feed.
#

# Import necessary packages
import cv2
import numpy as np
import time
import os
import Cards
import VideoStream


### ---- INITIALIZATION ---- ###
# Define constants and initialize variables

## Camera settings
IM_WIDTH = 320
IM_HEIGHT = 180 
FRAME_RATE = 30

## Initialize calculated frame rate because it's calculated AFTER the first time it's displayed
frame_rate_calc = 1
freq = cv2.getTickFrequency()

## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera object and video feed from the camera. The video stream is set up
# as a seperate thread that constantly grabs frames from the camera feed. 
# See VideoStream.py for VideoStream class definition
## IF USING USB CAMERA INSTEAD OF PICAMERA,
## CHANGE THE THIRD ARGUMENT FROM 1 TO 2 IN THE FOLLOWING LINE:
videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,2,0).start()
time.sleep(1) # Give the camera time to warm up

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
#train_pos = Cards.load_ranks( path + '/images/pos')
#train_suits = Cards.load_suits( path + '/Card_Imgs/')


### ---- MAIN LOOP ---- ###
# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify panini card and display them as a rect.

cam_quit = 0 # Loop control variable

test_image = ""
test_image = Cards.load_image(path, '/test/image1.jpg')

train_cards = Cards.load_compare_cards()

# Begin capturing frames
while cam_quit == 0:

    # Grab frame from video stream
    image = videostream.read()
    if (test_image != ""):
        image = np.copy(test_image)

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Pre-process camera image (gray, blur, and threshold it)
    pre_proc = Cards.preprocess_image(image)
    Cards.show_thumb("preprocess", pre_proc, 1, 0)

    # Find and sort the contours of all cards in the image (query cards)
    cards = Cards.find_cards(pre_proc, train_cards, image)

    # If there are no contours, do nothing
    if len(cards) != 0:
        # Draw card contours on image (have to do contours all at once or
        # they do not show up properly for some reason)
        temp_cnts = []
        for i in range(len(cards)):
            temp_cnts.append(cards[i].contour)
            print(cards[i])

    # Draw framerate in the corner of the image. Framerate is calculated at the end of the main loop,
    # so the first time this runs, framerate will be shown as 0.
    cv2.putText(image,"FPS: "+str(int(frame_rate_calc)),(10,26),font,0.7,(255,0,255),2,cv2.LINE_AA)

    # Finally, display the image with the identified cards!
    Cards.show_thumb("", image, 0, 0);
    cv2.imshow("Card Detector",image)
    Cards.save_image("/images/pos/final.jpg", np.copy(image))
    cv2.moveWindow("Card Detector", 0, 320);

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1
    
    # Poll the keyboard. If 'q' is pressed, exit the main loop.
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1
        

# Close all windows and close the PiCamera video stream.
cv2.destroyAllWindows()
videostream.stop()

