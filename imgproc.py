import constants
import gen
import vis

import math
import cv2
import numpy as np
from glob import glob

DEBUG = False
# makes output match image, but internal processing is mirrored
FLIP = True
RES = 1000

BLU0 = np.array([85,30,30])
BLU1 = np.array([105,255,255])
YEL0 = np.array([20,80,130])
YEL1 = np.array([30,255,220])
RED1_0 = np.array([0,90,100])
RED1_1 = np.array([20,255,190])
RED2_0 = np.array([170,90,100])
RED2_1 = np.array([180,255,190])

#image must be in HSV format
def find_centers(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # SHEET #
    _,thresh_gray = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    cnt_g,_ = cv2.findContours(thresh_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)

    ldbg = cv2.cvtColor(thresh_gray, cv2.COLOR_GRAY2BGR)

    sc = None
    ma = 0.0
    for c in cnt_g:
        a = cv2.contourArea(c)
        if a > ma:
            ma = a
            sc = c
    #e = 0.1*cv2.arcLength(sc, True)
    #asc = cv2.approxPolyDP(sc, e, True)
    bx,by,bw,bh = cv2.boundingRect(sc)
    lb = bx
    rb = bx+bw
    rect = cv2.minAreaRect(sc)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    if DEBUG:
        cv2.line(ldbg,(lb,0),(lb,RES),(255,0,0))
        cv2.line(ldbg,(rb,0),(rb,RES),(255,0,0))
        #cv2.drawContours(ldbg,[sc],0,(255,0,0),2)
        cv2.rectangle(ldbg, (bx,by), (bx+bw, by+bh), (0,0,255),2)
        cv2.drawContours(ldbg,[box],0,(0,255,0),2)
        cv2.imshow('ldbg {}'.format(id(img)), ldbg)

    # BLUE / SCALE #
    mask_blue = cv2.inRange(hsv, BLU0, BLU1)
    res_blue = cv2.bitwise_and(hsv,hsv, mask=mask_blue)
    gray_blue = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)
    _,thresh_blue = cv2.threshold(gray_blue,10,255,cv2.THRESH_BINARY)
    bdbg = cv2.cvtColor(thresh_blue, cv2.COLOR_GRAY2BGR)

    bcircs = cv2.HoughCircles(thresh_blue,cv2.HOUGH_GRADIENT,2.05,RES/4)[0]
    bcircs = [list([c for c in h]) for h in bcircs]

    r12 = 0.0
    tee = None
    w = float(hsv.shape[1])
    for c in bcircs:
        x, y, radius = c
        center = (x, y)
        if x > w/3 and x < 2*w/3 and x-radius > lb and x+radius < rb:
            if radius > r12:
                r12 = radius
                tee = center

        if DEBUG:
            c = [int(f) for f in center]
            r = int(radius)
            cv2.circle(bdbg, c, r, (240, 240, 0), 2)

    scale = constants.TWELVE / r12
    if tee is None:
        return None

    if DEBUG:
        c = [int(f) for f in tee]
        r = int(r12)
        cv2.circle(bdbg, c, r, (255, 0, 0), 6)
        cv2.imshow('bdbg {}'.format(id(img)), bdbg)

    # ROCKS #
    mask_yellow = cv2.inRange(hsv, YEL0, YEL1)
    mask_red1 = cv2.inRange(hsv, RED1_0, RED1_1)
    mask_red2 = cv2.inRange(hsv, RED2_0, RED2_1)
    mask_red = mask_red1 + mask_red2

    res_yellow = cv2.bitwise_and(hsv,hsv, mask=mask_yellow)
    res_red = cv2.bitwise_and(hsv,hsv, mask=mask_red)

    gray_yellow = cv2.cvtColor(res_yellow, cv2.COLOR_BGR2GRAY)
    gray_red = cv2.cvtColor(res_red, cv2.COLOR_BGR2GRAY)

    blur_yellow = cv2.blur(gray_yellow, (3,3))
    blur_red = cv2.blur(gray_red, (3,3))

    _,thresh_yellow = cv2.threshold(blur_yellow,10,255,cv2.THRESH_BINARY)
    _,thresh_red = cv2.threshold(blur_red,10,255,cv2.THRESH_BINARY)

    cdbg = thresh_red + thresh_yellow
    cdbg = cv2.cvtColor(cdbg, cv2.COLOR_GRAY2BGR)

    cnt_y,_ = cv2.findContours(thresh_yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
    cnt_r,_ = cv2.findContours(thresh_red,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)

    ycs = list()
    rcs = list()

    THRESH = 1.18
    MINFILL = 0.25
    ADJUST = 0.8 # color does not extend to the edge of the rock
    rmin = (constants.R_ROCK / scale) * ADJUST / THRESH
    rmax = (constants.R_ROCK / scale) * ADJUST * THRESH
    for c in cnt_y:
        center, radius = cv2.minEnclosingCircle(c)
        (x, y) = center
        if radius > rmin and radius < rmax and x > lb and x < rb:
            area = cv2.contourArea(c)
            if area > MINFILL * math.pi * pow(radius, 2):
                ycs.append(np.subtract(tee, center) * scale)
                if DEBUG:
                    ic = [int(f) for f in center]
                    r = int(radius)
                    cv2.circle(cdbg, ic, r, (0, 240, 240), 6)
    
    for c in cnt_r:
        center, radius = cv2.minEnclosingCircle(c)
        (x, y) = center
        if radius > rmin and radius < rmax and x > lb and x < rb:
            area = cv2.contourArea(c)
            if area > MINFILL * math.pi * pow(radius, 2):
                rcs.append(np.subtract(tee, center) * scale)
                if DEBUG:
                    ic = [int(f) for f in center]
                    r = int(radius)
                    cv2.circle(cdbg, ic, r, (0, 0, 255), 6)

    if DEBUG:
        print("Y: {}, R: {}".format(len(ycs), len(rcs)))
        cv2.imshow('cdbg {}'.format(id(img)), cdbg)

    return ycs, rcs

def get_sheets():
    imgs = [cv2.imread(url) for url in glob('imgs/*.png')]

    sheets = list()
    for img in imgs:
        y, x, c = img.shape
        # convert to RESxRES square, center cropped with full height
        if x > y:
            mgn = (x-y)//2
            img = img[:,mgn:mgn+y,:]
            img = cv2.resize(img, (RES, RES))
            #cv2.imshow('resized', img)
        if FLIP:
            img = cv2.flip(img, 1)

        ycs, rcs = find_centers(img)
        sheets.append(ycs + rcs)

    data = list()
    for sheet in sheets:
        # add throw and success?
        throw = sheet
        data.append(throw)

    return data

