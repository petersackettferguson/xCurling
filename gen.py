import constants

import math
import numpy as np
import matplotlib as plt

# coordinates measured from center of rock, with tee at the origin
# perspective is a birds-eye view with the back line at the bottom

def collide(a, b):
    if a is None or b is None:
        return None

    if np.linalg.norm(a - b) < (2 * constants.R_ROCK):
        return True
    else:
        return False

# determines the distance of a feature f from the line drawn between the origin point o and the target t
def dist(f, t, o=[0, constants.THROW]):
    if collide(f, t):
        return None
    if t[1] - f[1] > 0:
        return None
    
    dist = np.cross(f-o, t-o) / np.linalg.norm(t-o)
    #dist = f[0] - t[0] + (0.0000001 * (np.random.rand() - 0.5))
    return abs(dist)

# difficulty is inversely proportional to the distance between the throw and object
def exdiff(a, b):
    if a is None or b is None:
        return None

    d = dist(a, b)
    if d is None:
        return 0

    diff = 20.0 * np.log(3.0/abs(d))

    if diff < 0:
        #print("DEBUG", diff, 0.0)
        return 0
    elif diff > 100:
        #print("DEBUG", diff, 100.0)
        return 100
    else:
        #print("DEBUG", diff, diff)
        return diff

def chance(d, BASE=0.99, S=3.0):
    p = BASE - (d / (100*S))
    if p < 0:
        return 0
    elif p > 1:
        return 1
    else:
        return p
    

def rand_positions(n, scale=1.0, y_scale=2.0):
    positions = np.random.normal(scale=scale, size=(n, 2))
    positions = [np.asarray((x, y_scale * y)) for (x, y) in positions]

    return positions


def new_sheet(n=constants.N_ROCKS, scale=1.0, y_scale=2.0):
    sheet = list()
    rocks = rand_positions(constants.N_ROCKS, scale=scale, y_scale=y_scale)

    for i in range(n):
        j = 0
        while j < len(sheet):
            if collide(sheet[j], rocks[i]):
                #print("SHEET", sheet, j)
                #print("ROCK", sheet[j])
                #print("Popping", sheet[j][0], "for", rocks[i])
                #print()
                sheet.pop(j)
            else:
                j += 1
        sheet.append(rocks[i])

    while len(sheet) < n:
        sheet.append(None)

    return np.asarray(sheet, dtype=object)

def sheet_to_data(sheet):
    data = {}
    while len(sheet) < constants.N_ROCKS:
        sheet.append(None)
    for (i, rock) in enumerate(sheet):
        lx = "r{}x".format(i)
        ly = "r{}y".format(i)
        if rock is None:
            data.update([(lx, 0.0), (ly, -constants.HOG)])
        else:
            data.update([(lx, rock[0]), (ly, rock[1])])
    return data

def scale_data(data, throw=True):
    xkeys = ["r{}x".format(i) for i in range(constants.N_ROCKS)]
    ykeys = ["r{}y".format(i) for i in range(constants.N_ROCKS)]
    for xk in xkeys:
        data[xk] /= constants.SIDE
    for yk in ykeys:
        data[yk] /= constants.HOG
    if throw:
        data["x"] /= constants.SIDE
        data["y"] /= constants.HOG

def inv_scale(data, throw=True):
    xkeys = ["r{}x".format(i) for i in range(constants.N_ROCKS)]
    ykeys = ["r{}y".format(i) for i in range(constants.N_ROCKS)]
    for xk in xkeys:
        data[xk] *= constants.SIDE
    for yk in ykeys:
        data[yk] *= constants.HOG
    if throw:
        data["x"] *= constants.SIDE
        data["y"] *= constants.HOG

#unused—harmed performance
def to_rel(data):
    xkeys = ["r{}x".format(i) for i in range(constants.N_ROCKS)]
    ykeys = ["r{}y".format(i) for i in range(constants.N_ROCKS)]
    for xk in xkeys:
        data[xk] -= data["x"]
    for yk in ykeys:
        data[yk] -= data["y"]

def throw_chance(sheet, throw):
    return math.prod([chance(exdiff(d, throw)) for d in filter(lambda r: r is not None, sheet)])


def roc_throws(N=constants.N_ENDS):
    throws = list()
    hits = 0

    for sheet in [new_sheet() for i in range(N)]:
        for rock in sheet:
            if rock is None:
                break

            #diff = max([exdiff(d, rock) for d in filter(lambda r: r is not None, sheet)])
            chance = throw_chance(sheet, rock)
            hit = np.random.rand() < chance #chance(diff)
            
            data = sheet_to_data(sheet)
            data.update([("x", rock[0]), ("y", rock[1]), ("hit", int(hit))])
            throws.append(data)
            if hit:
                hits += 1

    return np.asarray(throws)

def loc_throws(N=constants.N_ENDS):
    throws = list()
    hits = 0

    for sheet in [new_sheet() for i in range(N)]:
        targets = rand_positions(len(sheet))
        for target in targets:
            #diff = max([exdiff(d, target) for d in filter(lambda r: r is not None, sheet)])
            chance = throw_chance(sheet, target)
            hit = np.random.rand() < chance #chance(diff)
            
            throws.append((sheet, target, hit))
            if hit:
                hits += 1

    return np.asarray(throws)

