import constants
import gen
import vis

import math
import cv2
from cv2 import xphoto
import numpy as np
from glob import glob

DEBUG = True
# makes output match image, but internal processing is mirrored
FLIP = True
RES = 1000

# Options: None, ROTATE_TEE
WARP_METHOD = "ROTATE_TEE"

R_ADJ = 0.8 # color does not extend to the edge of the rock
R_THR = 1.15 # rock radius permissibility
R_FILL = 0.50 # minimum proportion of color filling rock radius

# target mark color
TGT0 = np.array([50,220,20])
TGT1 = np.array([60,255,50])

BLU0 = np.array([85,30,30])
BLU1 = np.array([105,255,255])
YEL0 = np.array([20,70,130])
YEL1 = np.array([45,255,250])
RED1_0 = np.array([0,90,100])
RED1_1 = np.array([20,255,220])
RED2_0 = np.array([175,90,100])
RED2_1 = np.array([180,255,220])

def find_tee(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # FIND TEE
    mask_blue = cv2.inRange(hsv, BLU0, BLU1)
    res_blue = cv2.bitwise_and(hsv,hsv, mask=mask_blue)
    gray_blue = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)
    _,thresh_blue = cv2.threshold(gray_blue,10,255,cv2.THRESH_BINARY)
    bdbg = cv2.cvtColor(thresh_blue, cv2.COLOR_GRAY2BGR)

    bcircs = cv2.HoughCircles(thresh_blue,cv2.HOUGH_GRADIENT,2.05,RES/4)[0]
    bcircs = [list([c for c in h]) for h in bcircs]

    r12 = 0.0
    tee = None
    for c in bcircs:
        x, y, radius = c
        center = (x, y)
        if x > RES/3 and x < 2*RES/3 and radius < RES/2 and y > RES/2:
            if radius > r12:
                r12 = radius
                tee = center

        if DEBUG:
            c = [int(f) for f in center]
            r = int(radius)
            cv2.circle(bdbg, c, r, (240, 240, 0), 2)

    if tee is None:
        return None

    if DEBUG:
        c = [int(f) for f in tee]
        r = int(r12)
        cv2.circle(bdbg, c, r, (255, 0, 0), 5)

    return bdbg, c, r


# WARP IMAGE SO SHEET SIDES ARE VERTICAL
# 1: filter image to highlight sheet vs non-sheet
# 2: detect lines indicating sides of sheet. Average left and right groups.
# 3: find the bisecting angle of these lines to find the required rotation
# 4: calculate warp matrix to rotate + fix perspective
def warp(img, tee=None, method=None):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,thresh_gray = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    blur_gray = cv2.blur(thresh_gray, (3,3))
    canny = cv2.Canny(blur_gray, 100, 200)
    canny_blur = cv2.blur(canny, (5, 5))

    warped = img.copy()
    ldbg = cv2.cvtColor(thresh_gray, cv2.COLOR_GRAY2BGR)
    lb = 0
    rb = RES

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

    for line in sidelines:
        x1,y1,x2,y2=line
        if (x1+x2)/2 < RES /2:
            llines.append(line)
        else:
            rlines.append(line)
    if len(llines) > 0 and len(rlines) > 0:
        lbound = np.mean(llines,axis=0)
        rbound = np.mean(rlines,axis=0)

        
        lm = (lbound[2]-lbound[0])/(lbound[3]-lbound[1])
        rm = (rbound[2]-rbound[0])/(rbound[3]-rbound[1])
        la = math.atan(lm)
        ra = math.atan(rm)
        theta = la + ra
        def lbfy(y):
            b = lbound[0] - (lm*lbound[1])
            return (lm*y)+b
        def rbfy(y):
            b = rbound[0] - (rm*rbound[1])
            return (rm*y)+b
        def rotate(pt, center):
            x, y = pt
            cx, cy = center
            sx, sy = x - cx, y - cy
            rx = sx*math.cos(theta) - sy*math.sin(theta) + cx
            ry = sx*math.sin(theta) + sy*math.cos(theta) + cy
            return [rx, ry]

        ini = None
        fin = None
        if method is None or method == "NONE":
            lb = min((lbfy(0), lbfy(RES)))
            rb = max((rbfy(0), rbfy(RES)))
        elif method == "ROTATE_BASE":
            l0 = [lbfy(0), 0]
            r0 = [rbfy(0), 0]
            c = [(l0[0] + r0[0])/2, 0]
            l1 = [lbfy(RES), RES]
            r1 = [rbfy(RES), RES]
            l0d = rotate(l0, c)
            r0d = rotate(r0, c)
            l1d = rotate(l1, c)
            r1d = rotate(r1, c)

            ini = np.float32([l0, l1, r0, r1])
            fin = np.float32([l0d, l1d, r0d, r1d])
        elif method == "ROTATE_TEE":
            if tee is None:
                raise Exception("tee required if using method ROTATE_TEE")
            l0 = [lbfy(tee[1]), tee[1]]
            r0 = [rbfy(tee[1]), tee[1]]
            l1 = [lbfy(RES), RES]
            r1 = [rbfy(RES), RES]
            l0d = rotate(l0, tee)
            r0d = rotate(r0, tee)
            l1d = rotate(l1, tee)
            r1d = rotate(r1, tee)

            ini = np.float32([l0, l1, r0, r1])
            fin = np.float32([l0d, l1d, r0d, r1d])
        elif method == "BASE":
            l0 = [lbfy(0), 0]
            r0 = [rbfy(0), 0]
            c = [(l0[0] + r0[0])/2, 0]
            l1 = [lbfy(RES), RES]
            r1 = [rbfy(RES), RES]
            l0d = rotate(l0, c)
            r0d = rotate(r0, c)
            l1d = [l0d[0], RES]
            r1d = [r0d[0], RES]

            ini = np.float32([l0, l1, r0, r1])
            fin = np.float32([l0d, l1d, r0d, r1d])
        elif method == "TEE":
            if tee is None:
                raise Exception("tee required if using method TEE")
            l0 = [lbfy(tee[1]), tee[1]]
            r0 = [rbfy(tee[1]), tee[1]]
            l1 = [lbfy(RES), RES]
            r1 = [rbfy(RES), RES]
            l0d = rotate(l0, tee)
            r0d = rotate(r0, tee)
            l1d = [l0d[0], RES]
            r1d = [r0d[0], RES]

            ini = np.float32([l0, l1, r0, r1])
            fin = np.float32([l0d, l1d, r0d, r1d])

        if method is not None or method != "NONE":
            warp_matrix = cv2.getPerspectiveTransform(ini, fin)
            warped = cv2.warpPerspective(img,warp_matrix,(RES,RES))

        lb = int(l0d[0])
        rb = int(r0d[0])

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

    return warped, ldbg, lb, rb

def find_target(img, tee, scale, lb, rb):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(img, TGT0, TGT1)
    res_green = cv2.bitwise_and(img,img, mask=mask_green)
    gray_green = cv2.cvtColor(res_green, cv2.COLOR_BGR2GRAY)
    _,thresh_green = cv2.threshold(gray_green,10,255,cv2.THRESH_BINARY)

    tdbg = cv2.cvtColor(thresh_green, cv2.COLOR_GRAY2BGR)

    cnt_g,_ = cv2.findContours(thresh_green,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
    gcs = list()
    for c in cnt_g:
        center, radius = cv2.minEnclosingCircle(c)
        (x, y) = center
        if radius < RES/50 and x > lb and x < rb:
            area = cv2.contourArea(c)
            if area > .8 * math.pi * pow(radius, 2):
                gcs.append(np.subtract(tee, center) * scale)
                if DEBUG:
                    ic = [int(f) for f in center]
                    r = int(radius)
                    cv2.circle(tdbg, ic, r, (0, 240, 240), 1)
    if DEBUG:
        print(len(gcs), "targets")

    if len(gcs) >= 1:
        target = gcs[-1]
    else:
        target = None

    return tdbg, target

def find_rocks(img, tee, scale, lb, rb):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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

    rdbg = cv2.cvtColor(thresh_yellow + thresh_red, cv2.COLOR_GRAY2BGR)

    cnt_y,_ = cv2.findContours(thresh_yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
    cnt_r,_ = cv2.findContours(thresh_red,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)

    ycs = list()
    rcs = list()

    rmin = (constants.R_ROCK / scale) * R_ADJ / R_THR
    rmax = (constants.R_ROCK / scale) * R_ADJ * R_THR
    for c in cnt_y:
        center, radius = cv2.minEnclosingCircle(c)
        (x, y) = center
        if radius > rmin and radius < rmax and x > lb and x < rb:
            area = cv2.contourArea(c)
            if area > R_FILL * math.pi * pow(radius, 2):
                ycs.append(np.subtract(tee, center) * scale)
                if DEBUG:
                    ic = [int(f) for f in center]
                    r = int(radius)
                    cv2.circle(rdbg, ic, r, (0, 240, 240), 5)
    
    for c in cnt_r:
        center, radius = cv2.minEnclosingCircle(c)
        (x, y) = center
        if radius > rmin and radius < rmax and x > lb and x < rb:
            area = cv2.contourArea(c)
            if area > R_FILL * math.pi * pow(radius, 2):
                rcs.append(np.subtract(tee, center) * scale)
                if DEBUG:
                    ic = [int(f) for f in center]
                    r = int(radius)
                    cv2.circle(rdbg, ic, r, (0, 0, 255), 5)

    return rdbg, ycs, rcs

def process_sheet(img):
    # FIND TEE
    bdbg, tee, r12 = find_tee(img)
    scale = constants.TWELVE / r12

    # WARP SHEET
    warped,ldbg,lb,rb = warp(img, tee=tee, method=WARP_METHOD)
    gdbg = cv2.cvtColor(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    # ADJUST TEE
    #bdbg, tee, r12 = find_tee(warped)
    #scale = constants.TWELVE / r12

    # COLOR CORRECT
    wb = cv2.xphoto.createSimpleWB()
    cc = wb.balanceWhite(warped)

    # LOCATE TARGET MARKING
    tdbg, target = find_target(cc, tee, scale, lb, rb)

    # LOCATE ROCKS
    rdbg, ycs, rcs = find_rocks(cc, tee, scale, lb, rb)

    if DEBUG:
        c = [int(f) for f in tee]
        r = int(r12)
        cv2.circle(gdbg, c, r, (255, 0, 0), 5)

        cv2.line(gdbg,(lb,0),(lb,RES),(0,255,0), 2)
        cv2.line(gdbg,(rb,0),(rb,RES),(0,255,0), 2)
        cv2.line(gdbg,(lb,0),(lb,RES),(0,255,0), 2)
        cv2.line(gdbg,(rb,0),(rb,RES),(0,255,0), 2)

        if target is not None:
            tc = [int(f) for f in (-target/scale + tee)]
            cv2.circle(gdbg, tc, int(5), (0, 255, 0), 5)
        for c in ycs:
            c = [int(f) for f in (-c/scale + tee)]
            cv2.circle(gdbg, c, int(constants.R_ROCK/scale), (0, 240, 240), 5)
        for c in rcs:
            c = [int(f) for f in (-c/scale + tee)]
            cv2.circle(gdbg, c, int(constants.R_ROCK/scale), (0, 0, 255), 5)

        #cv2.imshow('ldbg {}'.format(id(img)), ldbg)
        #cv2.imshow('bdbg {}'.format(id(img)), bdbg)
        cv2.imshow('tdbg {}'.format(id(img)), tdbg)
        #cv2.imshow('rdbg {}'.format(id(img)), rdbg)
        cv2.imshow('debug {}'.format(id(img)), gdbg)
        cv2.waitKey(0)

    return ycs, rcs, target

def get_sheets():
    urls = glob('imgs/*.png')
    imgs = [cv2.imread(url) for url in urls]

    sheets = list()
    for img, url in zip(imgs, urls):
        print(url)
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

        hit = None
        if url[-5] == 'H':
            hit = 1
        elif url[-5] == 'M':
            hit = 0
        if hit is not None:
            ycs, rcs, target = process_sheet(img)
            sheets.append((ycs + rcs, target, hit))
        else:
            print("ERROR: no result specified for", url)


    data = list()
    for sheet, target, hit in sheets:
        throw = gen.sheet_to_data(sheet)
        throw.update({"x": target[0], "y": target[1], "hit": hit})
        data.append(throw)

    return data

