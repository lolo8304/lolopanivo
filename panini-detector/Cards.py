############## Panini cards image functions ###############
#
# Author: Lorenz Haenggi
# Date: 5/20/18
# Description: Functions and classes for CardDetector.py that perform 
# various steps of the card detection algorithm


# Import necessary packages
import numpy as np
import cv2
import time
import os
from skimage.measure import compare_ssim
import imutils

### Constants ###

# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

# Width and height of card corner, where rank and suit are
CORNER_WIDTH = 32
CORNER_HEIGHT = 84

# Dimensions of rank train images
RANK_WIDTH = 70
RANK_HEIGHT = 125

# Dimensions of suit train images
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700

CARD_MAX_AREA = 400000
CARD_MIN_AREA = 25000

THRESHOLD_MIN = 127
CARD_LONG_2_SHORT_FACTOR = 6.5 / 5
CARD_WRAP_LONG_MAX = 300
CARD_WRAP_SHORT_MAX = int(CARD_WRAP_LONG_MAX / CARD_LONG_2_SHORT_FACTOR)

font = cv2.FONT_HERSHEY_SIMPLEX
path = os.path.dirname(os.path.abspath(__file__))

### Structures to hold query card and train card information ###

class Query_card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.index = 0
        self.contour = [] # Contour of card
        self.width, self.height = 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.box = [] # box rectangle of card
        self.center = [] # Center point of card
        self.warp = [] # 200x300, flattened, grayed, blurred image
        self.number_img = [] # Thresholded, sized image of corner image to detect number
        self.best_number_match = "Unknown" # Best matched number

class Train_Card:
    """Structure to store information about trained oriented images."""

    def __init__(self):
        self.img = [] # grey, sized rank image loaded from hard drive
        self.image_threshold = [] # Thresholded, sized rank image loaded from hard drive
        self.name = "Placeholder"
        self.portrait = True

### Functions ###
def load_compare_cards():
    """Loads rank images from directory specified by filepath. Stores
    them in a list of Train_ranks objects."""

    train_ranks = []
    i = 0
    for Rank in [   "top", "right", "down", "left",
                    "landscape-right", "landscape-down", "landscape-left", "landscape-top"]:
        train_ranks.append(Train_Card())
        train_ranks[i].name = Rank
        image = load_image_bw(path, "/test/diff/{}.jpg".format(Rank))
        w = image.shape[1]
        h = image.shape[0]
        image_norm = []
        if (w > h):
            image_norm = cv2.resize(image, (CARD_WRAP_LONG_MAX, CARD_WRAP_SHORT_MAX))
        else:
            image_norm = cv2.resize(image, (CARD_WRAP_SHORT_MAX, CARD_WRAP_LONG_MAX))
        train_ranks[i].img = image_norm
        (thresh, image_threshold) = cv2.threshold(image_norm, THRESHOLD_MIN, 255, cv2.THRESH_BINARY)
        train_ranks[i].image_threshold = image_threshold
        train_ranks[i].portrait = (Rank.startswith("landscape") == False)
        i = i + 1

    return train_ranks


def load_image(filepath, name):
    """Loads rank images from directory specified by filepath. Stores
    them in a list of Train_ranks objects."""
    filename = filepath + name
    return cv2.imread(filename, cv2.IMREAD_COLOR)
def load_image_bw(filepath, name):
    """Loads rank images from directory specified by filepath. Stores
    them in a list of Train_ranks objects."""
    filename = filepath + name
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

def save_image(name, image):
    filepath = os.path.dirname(os.path.abspath(__file__))
    filename = filepath + name
    ##print ("save file named {}".format(filename))
    return cv2.imwrite(filename, image)

def show_thumb(name, image, x_index, y_index):
    """show tumbnail on screen to debug image pipeline"""

    MAX_WIDTH = CARD_WRAP_SHORT_MAX
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    r = MAX_WIDTH / image.shape[1]
    dim = (MAX_WIDTH, int(image.shape[0] * r))
    
    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("Card Detector-"+name, resized);
    cv2.moveWindow("Card Detector-"+name, x_index * (dim[0] + 20), y_index * (dim[1] + 20));


def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(15,15),0)

    # The best threshold level depends on the ambient lighting conditions.
    # For bright lighting, a high threshold must be used to isolate the cards
    # from the background. For dim lighting, a low threshold must be used.
    # To make the card detector independent of lighting conditions, the
    # following adaptive threshold method is used.
    #
    # A background pixel in the center top of the image is sampled to determine
    # its intensity. The adaptive threshold is set at 50 (THRESH_ADDER) higher
    # than that. This allows the threshold to adapt to the lighting conditions.
    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_TRIANGLE)
    
    return thresh

def find_cards(thresh_image, train_cards, image):
    """Finds all card-sized contours in a thresholded camera image.
    Returns the number of cards, and a list of card contours sorted
    from largest to smallest."""

    # Find contours and sort their indices by contour size
    dummy,cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)

    # If there are no contours, do nothing
    if len(cnts) == 0:
        return [], []
    
    # Otherwise, initialize empty sorted contour and hierarchy lists
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)

    # Fill empty lists with sorted contour and sorted hierarchy. Now,
    # the indices of the contour list still correspond with those of
    # the hierarchy list. The hierarchy array can be used to check if
    # the contours have parents or not.
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    # Determine which of the contours are cards by applying the
    # following criteria: 1) Smaller area than the maximum card size,
    # 2), bigger area than the minimum card size, 3) have no parents,
    # and 4) have four corners

    # Initialize a new "cards" list to assign the card objects.
    # k indexes the newly made array of cards.
    cards = []
    k = 0

    for i in range(len(cnts_sort)):
        ##cv2.drawContours(image, cnts_sort[i], -1, (255,0,0), 3)
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.1*peri,True)
        
        cnt = cnts_sort[i]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        if ((hier_sort[i][3] == -1) and (len(approx) == 4)):
            cv2.drawContours(image,[box],-1,(255,0,0),3)
            ##print ("** size={}".format(size))
            if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)):
                cnt_is_card[i] = 1

                # Create a card object from the contour and append it to the list of cards.
                # preprocess_card function takes the card contour and contour and
                # determines the cards properties (corner points, etc). It generates a
                # flattened 200x300 image of the card, and isolates the card's
                # suit and rank from the image.
                # Initialize new Query_card object
                qCard = Query_card()
                qCard.index = k
                qCard.contour = cnt
                pts = np.float32(approx)
                qCard.corner_pts = pts
                qCard.box = box
                ##print ("box {} - {}", k, box)

                cards.append(preprocess_card(qCard, train_cards, image))

                # Find the best rank and suit match for the card.
                ##cards[k].best_rank_match,cards[k].best_suit_match,cards[k].rank_diff,cards[k].suit_diff = Cards.match_card(cards[k],train_ranks,train_suits)

                # Draw center point and match result on the image.
                image = draw_results(image, qCard)
                k = k + 1

    return cards

def match_card_orientation(qCard, train_cards, image):
    test_w = qCard.warp.shape[1]
    test_h = qCard.warp.shape[0]
    (thresh, test_image_binary) = cv2.threshold(qCard.warp, THRESHOLD_MIN, 255, cv2.THRESH_BINARY)
    portrait = test_w < test_h

    best_match = 0.0
    best_match_index = -1

    for i in range(len(train_cards)):
        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        if (((i % 2) == 0) == portrait):
            compareImage_binary = train_cards[i].image_threshold
            ##print ("size={} size={}".format(compareImage_binary.shape, test_image_binary.shape))
            (score, diff) = compare_ssim(compareImage_binary, test_image_binary, full=True)
            diff = (diff * 255).astype("uint8")
            if (score > best_match):
                best_match = score
                best_match_index = i
                ##show_thumb("diff-{}".format(i), diff, 2, 0)
                print("image {}-{} SSIM: {}".format(i, train_cards[i].name, score))
                ##qCard.match = train_cards[i]
                qCard.best_number_match = train_cards[i].name
    return qCard


def preprocess_card(qCard, train_cards, image):
    """Uses contour to find information about the query card. Isolates rank
    and suit images from the card."""

    contour = qCard.contour
    pts = qCard.corner_pts

    # Find width and height of card's bounding rectangle
    x,y,w,h = cv2.boundingRect(contour)
    qCard.width, qCard.height = w, h

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened image using perspective transform
    qCard.warp = flattener2(image, pts, w, h)
    ##show_thumb("warp", qCard.warp, 2, 0)
    ##save_image("/images/pos/warp-{}.jpg".format(qCard.index), qCard.warp)
    trainCard = match_card_orientation(qCard, train_cards, image)

    return qCard

'''
    # Grab corner of warped card image and do a 4x zoom
    Qcorner = qCard.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    Qcorner_zoom = cv2.resize(Qcorner, (0,0), fx=4, fy=4)

    # Sample known white pixel intensity to determine good threshold level
    white_level = Qcorner_zoom[15,int((CORNER_WIDTH*4)/2)]
    thresh_level = white_level - CARD_THRESH
    if (thresh_level <= 0):
        thresh_level = 1
    retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2. THRESH_BINARY_INV)
    
    # Split in to top and bottom half (top shows rank, bottom shows suit)
    Qrank = query_thresh[20:185, 0:128]
    Qsuit = query_thresh[186:336, 0:128]

    # Find rank contour and bounding rectangle, isolate and find largest contour
    dummy, Qrank_cnts, hier = cv2.findContours(Qrank, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea,reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query rank
    # image to match dimensions of the train rank image
    if len(Qrank_cnts) != 0:
        x1,y1,w1,h1 = cv2.boundingRect(Qrank_cnts[0])
        Qrank_roi = Qrank[y1:y1+h1, x1:x1+w1]
        Qrank_sized = cv2.resize(Qrank_roi, (RANK_WIDTH,RANK_HEIGHT), 0, 0)
        qCard.rank_img = Qrank_sized

    # Find suit contour and bounding rectangle, isolate and find largest contour
    dummy, Qsuit_cnts, hier = cv2.findContours(Qsuit, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    Qsuit_cnts = sorted(Qsuit_cnts, key=cv2.contourArea,reverse=True)
    
    # Find bounding rectangle for largest contour, use it to resize query suit
    # image to match dimensions of the train suit image
    if len(Qsuit_cnts) != 0:
        x2,y2,w2,h2 = cv2.boundingRect(Qsuit_cnts[0])
        Qsuit_roi = Qsuit[y2:y2+h2, x2:x2+w2]
        Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        qCard.suit_img = Qsuit_sized
'''

    
def draw_results(image, qCard):
    """Draw the card name, center point, and contour on the camera image."""

    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image,(x,y),5,(255,0,0),-1)

    number_name = qCard.best_number_match

    # Draw card name twice, so letters have black outline
    cv2.putText(image,(number_name+' of'),(x-60,y-10),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,(number_name+' of'),(x-60,y-10),font,1,(50,200,200),2,cv2.LINE_AA)

    # Can draw difference value for troubleshooting purposes
    # (commented out during normal operation)
    #name_diff = str(qCard.number_diff)
    #cv2.putText(image,name_diff,(x+20,y+30),font,0.5,(0,0,255),1,cv2.LINE_AA)

    return image

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    pts2 = np.zeros((4, 2), dtype = "float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    pts2[0] = pts[0][0]
    pts2[1] = pts[1][0]
    pts2[2] = pts[2][0]
    pts2[3] = pts[3][0]
    s = pts2.sum(axis=1)
    ##print ("points 1= {}, 2={}, 3={}, 4={}".format(pts2[0], pts2[1], pts2[2], pts2[3]))
    rect[0] = pts2[np.argmin(s)]
    rect[2] = pts2[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts2, axis = 1)
    rect[1] = pts2[np.argmin(diff)]
    rect[3] = pts2[np.argmax(diff)]
 
    # return the ordered coordinates
    ##print ("rect 1= {}, 2={}, 3={}, 4={}".format(rect[0], rect[1], rect[2], rect[3]))
    return rect

def flattener2(image, pts, w, h):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
 
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    if (maxWidth > maxHeight):
        maxHeight = CARD_WRAP_SHORT_MAX
        maxWidth = CARD_WRAP_LONG_MAX
    else:
        maxWidth = CARD_WRAP_SHORT_MAX
        maxHeight = CARD_WRAP_LONG_MAX

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)

    # return the warped image
    return warped


def detectPaniniCorner(image):
    """ detect number corner in panini and distiguish rotation"""

def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
        
    maxWidth = 300
    maxHeight = 200

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

    return warp
