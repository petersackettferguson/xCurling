import constants
import gen
import vis

import math
import cv2
import numpy as np
from glob import glob

DEBUG = True
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # SHEET #
    _,thresh_gray = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    blur_gray = cv2.blur(thresh_gray, (3,3))
    canny = cv2.Canny(blur_gray, 100, 200)
    canny_blur = cv2.blur(canny, (5, 5))

    ldbg = cv2.cvtColor(thresh_gray, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLinesP(canny_blur,1,0.01,10,minLineLength=RES/2)
    lines = [l[0] for l in lines]
    MAXSKEW=RES/20
    sidelines = list()
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line
            if abs(x2-x1) < MAXSKEW:
                sidelines.append(line)
    llines = list()
    rlines = list()

    warped = img.copy()
    lb = 0
    rb = RES
    for line in sidelines:
        x1,y1,x2,y2=line
        if (x1+x2)/2 < RES /2:
            llines.append(line)
        else:
            rlines.append(line)
    if len(llines) > 0 and len(rlines) > 0:
        lbound = np.mean(llines,axis=0)
        rbound = np.mean(rlines,axis=0)
        l1 = [lbound[0], lbound[1]]
        l2 = [lbound[2], lbound[3]]
        r1 = [rbound[0], rbound[1]]
        r2 = [rbound[2], rbound[3]]
        if l1[1] > l2[1]:
            l1, l2 = l2, l1
        if r1[1] > r2[1]:
            r1, r2 = r2, r1
        l2d = [l1[0], l2[1]]
        r2d = [r1[0], r2[1]]

        ini = np.float32([l1, l2, r1, r2])
        fin = np.float32([l1, l2d, r1, r2d])

        warp_matrix = cv2.getPerspectiveTransform(ini, fin)
        warped = cv2.warpPerspective(img,warp_matrix,(RES,RES))

        lb = int(l1[0])
        rb = int(r1[0])

        if DEBUG:
            lx1,ly1,lx2,ly2=[int(v) for v in lbound]
            rx1,ry1,rx2,ry2=[int(v) for v in rbound]
            cv2.line(ldbg, (lx1,ly1), (lx2,ly2), (0,0,255),2)
            cv2.line(ldbg, (rx1,ry1), (rx2,ry2), (0,0,255),2)

    if DEBUG:
        cv2.line(ldbg,(lb,0),(lb,RES),(0,255,0))
        cv2.line(ldbg,(rb,0),(rb,RES),(0,255,0))
        cv2.line(ldbg,(lb,0),(lb,RES),(0,255,0))
        cv2.line(ldbg,(rb,0),(rb,RES),(0,255,0))

    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    gdbg = cv2.cvtColor(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

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
        cv2.circle(bdbg, c, r, (255, 0, 0), 5)
        cv2.circle(gdbg, c, r, (255, 0, 0), 5)

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
                    cv2.circle(cdbg, ic, r, (0, 240, 240), 5)
                    cv2.circle(gdbg, ic, r, (0, 240, 240), 5)
    
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
                    cv2.circle(cdbg, ic, r, (0, 0, 255), 5)
                    cv2.circle(gdbg, ic, r, (0, 0, 255), 5)

    if DEBUG:
        #cv2.imshow('ldbg {}'.format(id(img)), ldbg)
        #cv2.imshow('bdbg {}'.format(id(img)), bdbg)
        #cv2.imshow('cdbg {}'.format(id(img)), cdbg)
        cv2.imshow('debug {}'.format(id(img)), gdbg)
        cv2.waitKey(0)

    return ycs, rcs

def get_sheets():
    imgs = [cv2.imread(url) for url in glob('imgs/*.png')]

    sheets = list()
    for img in imgs:
        y, x, c = img.shape
        # convert to RESxRES square, center cropped with full height
        if x > y:
            mgn = (x-y)//2
            img = img[:,mgn:-mgn,:]
            img = cv2.resize(img, (RES, RES), interpolation=cv2.INTER_AREA)
            #cv2.imshow('resized', img)
            #cv2.waitKey(0)
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

