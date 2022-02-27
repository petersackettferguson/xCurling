import constants
import gen
import vis

import numpy as np
import pandas as pd
from sklearn import linear_model, ensemble, neural_network, svm
from sklearn import feature_extraction, model_selection
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt

N=1000
def create_models(models=['rfc', 'svc', 'mplc'], method='random'):
    if method == 'random':
        throws_data = gen.roc_throws(N=N)
    if method == 'pict':
        throws_data = imgproc.get_image_data()

    #hits = [t["hit"] for t in throws_data].count(True)
    print("N:", N)

    df = pd.DataFrame.from_records(throws_data)
    v = feature_extraction.DictVectorizer(sparse=False)
    X = v.fit_transform(throws_data)
    X = [t[1:] for t in X]
    y = df["hit"]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    #lr = linear_model.LogisticRegression(max_iter=1000, C=100.0)
    #lr.fit(X_train, y_train)

    #rfc = ensemble.RandomForestClassifier()
    #rfc_c = CalibratedClassifierCV(base_estimator=rfc, n_jobs=-1)
    #rfc.fit(X_train, y_train)
    #rfc_c.fit(X_train, y_train)

    #svc = svm.SVC(probability=True, C=100.0)
    #svc_c = CalibratedClassifierCV(base_estimator=svc, n_jobs=-1)
    #svc.fit(X_train, y_train)
    #svc_c.fit(X_train, y_train)

    mlpc_relu = neural_network.MLPClassifier(activation='relu', max_iter=10000)
    mlpc_relu_c = CalibratedClassifierCV(base_estimator=mlpc_relu, n_jobs=-1)
    mlpc_relu_c.fit(X_train, y_train)
    mlpc_relu_score = mlpc_relu_c.score(X_test, y_test)
    print("MLPC (relu) |", mlpc_relu_score)

    mlpc_tanh = neural_network.MLPClassifier(activation='tanh', max_iter=10000)
    mlpc_tanh_c = CalibratedClassifierCV(base_estimator=mlpc_tanh, n_jobs=-1)
    mlpc_tanh_c.fit(X_train, y_train)
    mlpc_tanh_score = mlpc_tanh_c.score(X_test, y_test)
    print("MLPC (tanh) |", mlpc_tanh_score)

    mlpc_logi = neural_network.MLPClassifier(activation='logistic', max_iter=10000)
    mlpc_logi_c = CalibratedClassifierCV(base_estimator=mlpc_logi, n_jobs=-1)
    mlpc_logi_c.fit(X_train, y_train)
    mlpc_logi_score = mlpc_logi_c.score(X_test, y_test)
    print("MLPC (logi) |", mlpc_logi_score)

#    mlpc_lbfgs = neural_network.MLPClassifier(solver='lbfgs', max_iter=1000)
#    mlpc_lbfgs_c = CalibratedClassifierCV(base_estimator=mlpc_lbfgs, n_jobs=-1)
#    mlpc_lbfgs_c.fit(X_train, y_train)
#    mlpc_lbfgs_score = mlpc_lbfgs_c.score(X_test, y_test)
#    print("MLPC (lbfgs) |", mlpc_lbfgs_score)

#    mlpc_sgd = neural_network.MLPClassifier(solver='sgd', max_iter=1000)
#    mlpc_sgd_c = CalibratedClassifierCV(base_estimator=mlpc_sgd, n_jobs=-1)
#    mlpc_sgd_c.fit(X_train, y_train)
#    mlpc_sgd_score = mlpc_sgd_c.score(X_test, y_test)
#    print("MLPC (sgd) |", mlpc_sgd_score)

    #lr_scores = model_selection.cross_val_score(lr, X, y)
    #print("LR: {:.2f} accuracy with stdev {:.4f}".format(lr_scores.mean(), lr_scores.std()))
#    rfc_scores = model_selection.cross_val_score(rfc, X, y)
#    print("RFC: {:.2f} accuracy with stdev {:.4f}".format(rfc_scores.mean(), rfc_scores.std()))
#    svc_scores = model_selection.cross_val_score(svc, X, y)
#    print("SVC: {:.2f} accuracy with stdev {:.4f}".format(svc_scores.mean(), svc_scores.std()))
#    mlpc_scores = model_selection.cross_val_score(mlpc, X, y)
#    print("MLPC: {:.4f} accuracy with stdev {:.4f}".format(mlpc_scores.mean(), mlpc_scores.std()))
#    rfc_c_scores = model_selection.cross_val_score(rfc_c, X, y)
#    print("RFC (C): {:.2f} accuracy with stdev {:.4f}".format(rfc_c_scores.mean(), rfc_c_scores.std()))
#    svc_c_scores = model_selection.cross_val_score(svc_c, X, y)
#    print("SVC (C): {:.2f} accuracy with stdev {:.4f}".format(svc_c_scores.mean(), svc_c_scores.std()))
#    mlpc_c_scores = mlpc_c.score(X, y)
#    print("MLPC (C): {:.2f} accuracy with stdev {:.4f}".format(mlpc_c_scores.mean(), mlpc_c_scores.std()))
#    mlp_fgs_c_scores = model_selection.cross_val_score(mlp_fgs_c, X, y)
#    print("MLPC LBFGS (C): {:.2f} accuracy with stdev {:.4f}".format(mlp_fgs_c_scores.mean(), mlp_fgs_c_scores.std()))

    models = [mlpc_relu_c, mlpc_tanh_c, mlpc_logi_c]
    labels = ["MLP Classifier (relu)", "MLP Classifier (tanh)", "MLP Classifier (logistic)"]
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
#mlpc_vis_prediction = mlpc.predict_proba(X)[0][1]
#cal_prediction = gen.throw_chance(vis_sheet, vis_target)

#s = "lr predicted accuracy: {:.4f}\nsvc predicted accuracy: {:.4f}\nmlpc predicted accuracy: {:.4f}\ncalculated accuracy: {:.4f}".format(lr_vis_prediction, svc_vis_prediction, mlpc_vis_prediction, cal_prediction)
#vis.plot_data(vis_throw, scaled=True, throw=True, text=s)
