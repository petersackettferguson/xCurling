import constants
import gen

import matplotlib.pyplot as plt
import numpy as np


def plot_data(data, scaled=False, throw=False, text=None):
    if scaled:
        gen.inv_scale(data)
    xlabels = ["r{}x".format(i) for i in range(constants.N_ROCKS)]
    ylabels = ["r{}y".format(i) for i in range(constants.N_ROCKS)]
    xpts = [data[xl] for xl in xlabels]
    ypts = [data[yl] for yl in ylabels]


    c1 = plt.Circle((0, 0), constants.BUTTON, color='w')
    c4 = plt.Circle((0, 0), constants.FOUR, color='r')
    c8 = plt.Circle((0, 0), constants.EIGHT, color='w')
    c12 = plt.Circle((0, 0), constants.TWELVE, color='b')

    pts = [plt.Circle((x, y), constants.R_ROCK, color="grey") for (x, y) in zip(xpts, ypts)]


    plt.xlim((-constants.SIDE, constants.SIDE))
    plt.ylim((-constants.HACK, constants.HOG))

    ax = plt.gca()

    ax.add_patch(c12)
    ax.add_patch(c8)
    ax.add_patch(c4)
    ax.add_patch(c1)
    
    for pt in pts:
        ax.add_patch(pt)
    if throw:
        plt.plot(data['x'], data['y'], 'x', c='g')
    if text is not None:
        plt.text(-6, 18, text)
    
    plt.show()
