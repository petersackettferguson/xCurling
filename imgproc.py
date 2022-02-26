import constants
import gen
import vis

import math
import cv2
import numpy as np
from glob import glob

BLU = np.array([150,80,50])
YEL = np.array([35,80,50])
RED = np.array([5, 60, 40])

BLU0 = np.array([140,50,50])
BLU1 = np.array([160,255,255])
YEL0 = np.array([25,50,50])
YEL1 = np.array([45,255,255])
RED0 = np.array([0,50,50])
RED1 = np.array([15,255,255])

#image must be in HSV format
def find_centers(img):
    mask_blue = cv2.inRange(img, BLU0, BLU1)
    mask_yellow = cv2.inRange(img, YEL0, YEL1)
    mask_red = cv2.inRange(img, RED0, RED1)

    res_blue = cv2.bitwise_and(img,img, mask=mask_blue)
    res_yellow = cv2.bitwise_and(img,img, mask=mask_yellow)
    res_red = cv2.bitwise_and(img,img, mask=mask_red)

    gray_blue = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)
    gray_yellow = cv2.cvtColor(res_yellow, cv2.COLOR_BGR2GRAY)
    gray_red = cv2.cvtColor(res_red, cv2.COLOR_BGR2GRAY)

    _,thresh_blue = cv2.threshold(gray_blue,10,255,cv2.THRESH_BINARY)
    _,thresh_yellow = cv2.threshold(gray_yellow,10,255,cv2.THRESH_BINARY)
    _,thresh_red = cv2.threshold(gray_red,10,255,cv2.THRESH_BINARY)

    cnt_b, hierarchy1 = cv2.findContours(thresh_blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt_y, hierarchy3 = cv2.findContours(thresh_yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt_r, hierarchy2 = cv2.findContours(thresh_red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    ycs = list()
    rcs = list()
    r12 = 0.0
    tee = None

    fthrs = 1.5
    frat = 5*math.pi/9
    fmin = frat / 1.5
    fmax = frat * 1.5

    cl = img.shape[1]/4
    cr = img.shape[1]*3/4
    cb = img.shape[0]/4
    ct = img.shape[1]*3/4
    for c in cnt_b:
        area = cv2.contourArea(c)
        center, radius = cv2.minEnclosingCircle(c)
        #a_to_r2 = float(area) / pow(radius, 2)
        if area > 3000 and center[0] > cl and center[0] < cr and center[1] > cb and center[1] < ct:
            r12 = radius
            tee = center

    scale = constants.TWELVE / r12
    rthrs = 1.5
    rmin = (constants.R_ROCK / scale) / rthrs
    rmax = (constants.R_ROCK / scale) * rthrs

    if tee is None:
        return None

    for c in cnt_y:
        center, radius = cv2.minEnclosingCircle(c)
        if radius > rmin and radius < rmax:
            ycs.append(np.subtract(tee, center) * scale)
    
    for c in cnt_r:
        center, radius = cv2.minEnclosingCircle(c)
        if radius > rmin and radius < rmax:
            rcs.append(np.subtract(tee, center) * scale)

    return ycs, rcs

#imgs = [cv2.imread(url) for url in glob('imgs/*.png')]
#sheets = list()
#for img in imgs:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycs, rcs = find_centers(img)
    s = img.shape
    sheets.append(ycs + rcs)

#data = gen.sheet_to_data(sheets[0])
#vis.plot_data(gen.sheet_to_data(sheets[0]))
#print(sheets)

