############## Python-OpenCV Playing Card Detector ###############
#
# Author: Lorenz Haenggi
# Date: 5/20/18
# Description: Python script to test if detect correct orientation using diff of image
#

# Import necessary packages
import cv2
import numpy as np
import time
import os
import Cards
from skimage.measure import compare_ssim
import imutils


THRESHOLD_MIN = 127
CARD_LONG_2_SHORT_FACTOR = 6.5 / 5
CARD_WRAP_LONG_MAX = 300
CARD_WRAP_SHORT_MAX = int(CARD_WRAP_LONG_MAX / CARD_LONG_2_SHORT_FACTOR)

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
test_image = Cards.load_image_bw(path, '/test/image10.jpg')
test_w = test_image.shape[1]
test_h = test_image.shape[0]
test_image_norm = []
if (test_w > test_h):
  test_image_norm = cv2.resize(test_image, (CARD_WRAP_LONG_MAX, CARD_WRAP_SHORT_MAX))
else:
  test_image_norm = cv2.resize(test_image, (CARD_WRAP_SHORT_MAX, CARD_WRAP_LONG_MAX))

(thresh, test_image_binary) = cv2.threshold(test_image_norm, THRESHOLD_MIN, 255, cv2.THRESH_BINARY)
Cards.show_thumb("diff-{}".format(-1), test_image_binary, 0, 0)

##from https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/

names = [
  "top", "right", "down", "left",
  "landscape-right", "landscape-down", "landscape-left", "landscape-top"
]

images = []
images_threshold = []
for i in range(len(names)):
  print ("/test/diff/{}.jpg".format(names[i]))
  image = Cards.load_image_bw(path, "/test/diff/{}.jpg".format(names[i]))
  w = image.shape[1]
  h = image.shape[0]
  image_norm = []
  if (w > h):
    image_norm = cv2.resize(image, (CARD_WRAP_LONG_MAX, CARD_WRAP_SHORT_MAX))
  else:
    image_norm = cv2.resize(image, (CARD_WRAP_SHORT_MAX, CARD_WRAP_LONG_MAX))
  images.append(image)
  (thresh, image_threshold) = cv2.threshold(image_norm, THRESHOLD_MIN, 255, cv2.THRESH_BINARY)
  images_threshold.append(image_threshold)


portrait = True
if (test_w > test_h):
  portrait = False

best_match = 0.0
best_match_index = -1

for i in range(len(images)):
  # compute the Structural Similarity Index (SSIM) between the two
  # images, ensuring that the difference image is returned
  if (((i % 2) == 0) == portrait):
    compareImage_binary = images_threshold[i]
    (score, diff) = compare_ssim(compareImage_binary, test_image_binary, full=True)
    diff = (diff * 255).astype("uint8")
    print("image {}-{} SSIM: {}".format(i, names[i], score))
    if (score > best_match):
      best_match = score
      best_match_index = i
      Cards.show_thumb("diff-{}".format(i), diff, 0, 0)

  
cam_quit = 0
while cam_quit == 0:
  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
      cam_quit = 1
        

# Close all windows and close the PiCamera video stream.
cv2.destroyAllWindows()
