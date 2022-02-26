import core
import gen
import imgproc
import vis

import numpy as np
import matplotlib.pyplot as plt
from sklearn import feature_extraction

models, labels = core.create_models()

v = feature_extraction.DictVectorizer(sparse=False)
vis_sheet = gen.sheet_to_data(gen.new_sheet())
mx = np.arange(-7.5, 7.5, 1.0)
my = np.arange(-11.5, 20.5, 1.0)
mps = list()
for model in models:
    Z = list()
    for y in my:
        Zr = list()
        for x in mx:
            pt = (x, y)
            xythrow = vis_sheet
            xythrow.update([("x", pt[0]), ("y", pt[1])])
            X = v.fit_transform(xythrow)
            p = model.predict_proba(X)[0][1] #generalize
            Zr.append(p)
        Z.append(Zr)
    mps.append(Z)

fig, axs = plt.subplots(1, 3)
vis.plot_map(vis_sheet, mx, my, mps, labels=labels, axs=axs)