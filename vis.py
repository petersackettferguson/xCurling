import constants
import gen

import matplotlib.pyplot as plt
import numpy as np
from sklearn import feature_extraction

def plot_sheet(data, ax=None, scaled=False, fill=True, a=1.0):
    if scaled:
        gen.inv_scale(data)
    xlabels = ["r{}x".format(i) for i in range(constants.N_ROCKS)]
    ylabels = ["r{}y".format(i) for i in range(constants.N_ROCKS)]
    xpts = [data[xl] for xl in xlabels]
    ypts = [data[yl] for yl in ylabels]


    c1 = plt.Circle((0, 0), constants.BUTTON, color='w', fill=fill, alpha=a)
    c4 = plt.Circle((0, 0), constants.FOUR, color='r', fill=fill, alpha=a)
    c8 = plt.Circle((0, 0), constants.EIGHT, color='w', fill=fill, alpha=a)
    c12 = plt.Circle((0, 0), constants.TWELVE, color='b', fill=fill, alpha=a)

    pts = [plt.Circle((x, y), constants.R_ROCK, color="grey") for (x, y) in zip(xpts, ypts)]

    if ax is None:
        ax = plt.gca()

    ax.set_xlim(-constants.SIDE, constants.SIDE)
    ax.set_ylim(-constants.HACK, constants.HOG)

    ax.add_patch(c12)
    ax.add_patch(c8)
    ax.add_patch(c4)
    ax.add_patch(c1)
    
    for pt in pts:
        ax.add_patch(pt)

def plot_data(data, scaled=False, throw=False, text=None):
    plot_sheet(data, scaled=scaled)

    if throw:
        plt.plot(data['x'], data['y'], 'x', c='g')
    if text is not None:
        plt.text(-6, 18, text)
    
    plt.show()

def plot_map(data, xr, yr, mps, axs=None, labels=None, scaled=False):
    
    mx = np.arange(-9.0, 9.0, 1.0)
    my = np.arange(-12.0, 21.0, 1.0)
    Z = list()

    if axs is None:
        ax = plt.gca()
        ax.pcolormesh(xr, yr, mps, shading='nearest')
        plot_sheet(data, scaled=scaled, fill=False)
    else:
        for i, ax in enumerate(axs):
            ax.pcolormesh(xr, yr, mps[i], shading='nearest')
            plot_sheet(data, ax=ax, scaled=scaled, fill=False)
            if labels is not None:
                ax.text(-6, 18, labels[i])


    plt.show()
