import core
import gen
import imgproc
import vis

import numpy as np
import matplotlib.pyplot as plt
from sklearn import feature_extraction

models, labels, throws = core.create_models(method='img')

vis_sheet = throws[2]
mx = np.arange(-8.5, 8.5, 1.0)
my = np.arange(-12.5, 21.5, 1.0)
mps = list()
calprobs: list()

v = feature_extraction.DictVectorizer(sparse=False)
for model in models:
    Z = list()
    for y in my:
        Zr = list()
        for x in mx:
            pt = (x, y)
            xythrow = vis_sheet
            xythrow.update([("x", pt[0]), ("y", pt[1]), ("hit", None)])
            X = v.fit_transform(xythrow)
            if len(X[0]) == 35:
                X = [t[1:] for t in X]
            p = model.predict_proba(X)[0][1]
            Zr.append(p)
        Z.append(Zr)
    mps.append(Z)

fig, axs = plt.subplots(1, len(models))
vis.plot_map(vis_sheet, mx, my, mps, labels=labels, axs=axs)
