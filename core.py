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

N=1000
throws_data = gen.roc_throws(N=N)
hits = [t["hit"] for t in throws_data].count(True)
print("N:", N)
#print("sample accuracy:", float(hits)/len(throws_data))

df = pd.DataFrame.from_records(throws_data)
v = feature_extraction.DictVectorizer(sparse=False)
X = v.fit_transform(throws_data)
X = [t[1:] for t in X]
y = df["hit"]

lr = linear_model.LogisticRegression(max_iter=10000)
lr.fit(X, y)
svc = svm.SVC(probability=True)
svc.fit(X, y)
nn = neural_network.MLPClassifier(max_iter=10000)
nn.fit(X, y)

lr_scores = model_selection.cross_val_score(lr, X, y)
print("LR: {:.2f} accuracy with stdev {:.4f}".format(lr_scores.mean(), lr_scores.std()))
svc_scores = model_selection.cross_val_score(svc, X, y)
print("SVC: {:.2f} accuracy with stdev {:.4f}".format(svc_scores.mean(), svc_scores.std()))
nn_scores = model_selection.cross_val_score(nn, X, y)
print("NN: {:.4f} accuracy with stdev {:.4f}".format(nn_scores.mean(), nn_scores.std()))

#predicted_hits = 0
#errors = 0
#successes = 0
#N = 1000
#deltas = []
#for _ in range(N):
#    test_sheet = gen.new_sheet()
#    test_throw = gen.sheet_to_data(test_sheet)
#    test_target = gen.rand_positions(1)[0]
#    test_throw.update([("x", test_target[0]), ("y", test_target[1])])
#    gen.scale_data(test_throw)
#    tv = feature_extraction.DictVectorizer(sparse=False)
#    tX = tv.fit_transform(test_throw)
#    #test_prediction = regr.predict_proba(tX)[0][1]
#    test_prediction = nn.predict_proba(tX)[0][1]
#
#    #print(test_prediction)
#    #if test_prediction < 0 or test_prediction > 1:
#    #    errors += 1
#    #else:
#    deltas.append(abs(test_prediction - gen.throw_chance(test_sheet, test_target)))
#    successes += 1
#    if np.random.rand() < test_prediction:
#        predicted_hits += 1
#
#
#print("ABSOLUTE")
##print("RELATIVE")
#print("predicted accuracy:", float(predicted_hits)/float(successes))
##print("error rate:", float(errors)/float(N))
#print("mean delta:", np.mean(deltas))
#print("median delta:", np.median(deltas))

vis_sheet = gen.new_sheet()
vis_throw = gen.sheet_to_data(vis_sheet)
vis_target = gen.rand_positions(1)[0]
vis_throw.update([("x", vis_target[0]), ("y", vis_target[1])])
gen.scale_data(vis_throw)

v = feature_extraction.DictVectorizer(sparse=False)
X = v.fit_transform(vis_throw)
lr_vis_prediction = lr.predict_proba(X)[0][1]
svc_vis_prediction = svc.predict_proba(X)[0][1]
nn_vis_prediction = nn.predict_proba(X)[0][1]
cal_prediction = gen.throw_chance(vis_sheet, vis_target)

s = "lr predicted accuracy: {:.4f}\nsvc predicted accuracy: {:.4f}\nnn predicted accuracy: {:.4f}\ncalculated accuracy: {:.4f}".format(lr_vis_prediction, svc_vis_prediction, nn_vis_prediction, cal_prediction)
vis.plot_data(vis_throw, scaled=True, throw=True, text=s)
