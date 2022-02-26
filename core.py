import constants
import gen
import vis

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import feature_extraction
from sklearn import neural_network
from sklearn import svm
from sklearn import model_selection

import matplotlib.pyplot as plt

N=160
def create_models(method='random'):
    if method == 'random':
        throws_data = gen.roc_throws(N=N)
    if method == 'pict':
        throws_data = imgproc.get_image_data()

    hits = [t["hit"] for t in throws_data].count(True)
    print("N:", N)

    df = pd.DataFrame.from_records(throws_data)
    v = feature_extraction.DictVectorizer(sparse=False)
    X = v.fit_transform(throws_data)
    X = [t[1:] for t in X]
    y = df["hit"]

    lr = linear_model.LogisticRegression(max_iter=1000, C=100.0)
    lr.fit(X, y)
    svc = svm.SVC(probability=True, C=1000.0)
    svc.fit(X, y)
    nn = neural_network.MLPClassifier(max_iter=10000)
    nn.fit(X, y)

    lr_scores = model_selection.cross_val_score(lr, X, y)
    print("LR: {:.2f} accuracy with stdev {:.4f}".format(lr_scores.mean(), lr_scores.std()))
    svc_scores = model_selection.cross_val_score(svc, X, y)
    print("SVC: {:.2f} accuracy with stdev {:.4f}".format(svc_scores.mean(), svc_scores.std()))
    nn_scores = model_selection.cross_val_score(nn, X, y)
    print("NN: {:.4f} accuracy with stdev {:.4f}".format(nn_scores.mean(), nn_scores.std()))

    models = [lr, svc, nn]
    labels = ["Logistic Regression", "Support Vector Classification", "MLPC Classifier"]
    return models, labels


#vis_sheet = gen.new_sheet()
#vis_throw = gen.sheet_to_data(vis_sheet)
#vis_target = gen.rand_positions(1)[0]
#vis_throw.update([("x", vis_target[0]), ("y", vis_target[1])])
#gen.scale_data(vis_throw)

#v = feature_extraction.DictVectorizer(sparse=False)
#X = v.fit_transform(vis_throw)
#lr_vis_prediction = lr.predict_proba(X)[0][1]
#svc_vis_prediction = svc.predict_proba(X)[0][1]
#nn_vis_prediction = nn.predict_proba(X)[0][1]
#cal_prediction = gen.throw_chance(vis_sheet, vis_target)

#s = "lr predicted accuracy: {:.4f}\nsvc predicted accuracy: {:.4f}\nnn predicted accuracy: {:.4f}\ncalculated accuracy: {:.4f}".format(lr_vis_prediction, svc_vis_prediction, nn_vis_prediction, cal_prediction)
#vis.plot_data(vis_throw, scaled=True, throw=True, text=s)
