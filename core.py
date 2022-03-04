import constants
import gen
import imgproc
import vis

import numpy as np
import pandas as pd
from sklearn import linear_model, ensemble, gaussian_process, neural_network, svm
from sklearn import feature_extraction, metrics, model_selection
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt

N=1000
def create_models(models=['rfc', 'svc', 'mplc'], method='rand', n=N):
    throws_data = None
    if method == 'rand':
        throws_data = gen.roc_throws(N=n)
    if method == 'img':
        throws_data = imgproc.get_sheets()
        n = len(throws_data)

    #hits = [t["hit"] for t in throws_data].count(True)
    print("N:", n)

    df = pd.DataFrame.from_records(throws_data)
    v = feature_extraction.DictVectorizer(sparse=False)
    X = v.fit_transform(throws_data)
    X = [t[1:] for t in X]
    y = df["hit"]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    mlpc_pg = {
        'hidden_layer_sizes': [(80), (160), (80, 40), (160, 80)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam'],
        #'alpha': [0.01, 0.01, 0.1, 1],
        'alpha': [0.1],
        'learning_rate': ['constant','adaptive'],
    }

    #mlpc = neural_network.MLPClassifier(max_iter=1000)
    #grid = model_selection.GridSearchCV(mlpc, mlpc_pg, scoring='balanced_accuracy', n_jobs=-1)
    #grid.fit(X_train, y_train)
    #print(grid.best_params_)
    #mlpc_c = CalibratedClassifierCV(grid.best_estimator_, n_jobs=-1)
    #mlpc_c.fit(X_train, y_train)
    #grid_predictions = mlpc_c.predict(X_test)
    #print("Accuracy: {:.4f}".format(metrics.balanced_accuracy_score(y_test, grid_predictions)))

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

    #mlpco = neural_network.MLPClassifier(alpha=1, max_iter=1000)
    #mlpco_c = CalibratedClassifierCV(base_estimator=mlpco, n_jobs=-1)
    #mlpco_c.fit(X_train, y_train)
    #mlpco_predictions = mlpco_c.predict(X_test)
    #print("MLP Classifier (1) |", metrics.balanced_accuracy_score(y_test, mlpco_predictions))

    mlpca = neural_network.MLPClassifier(alpha=.1, learning_rate='adaptive', max_iter=1000)
    mlpca_c = CalibratedClassifierCV(base_estimator=mlpca, n_jobs=-1)
    mlpca_c.fit(X_train, y_train)
    mlpca_predictions = mlpca_c.predict(X_test)
    print("MLP Classifier (Adaptive) |", metrics.balanced_accuracy_score(y_test, mlpca_predictions))

    mlpc1 = neural_network.MLPClassifier(alpha=.1, max_iter=1000)
    mlpc1_c = CalibratedClassifierCV(base_estimator=mlpc1, n_jobs=-1)
    mlpc1_c.fit(X_train, y_train)
    mlpc1_predictions = mlpc1_c.predict(X_test)
    print("MLP Classifier (.1) |", metrics.balanced_accuracy_score(y_test, mlpc1_predictions))
    
    mlpc01 = neural_network.MLPClassifier(alpha=.01, max_iter=1000)
    mlpc01_c = CalibratedClassifierCV(base_estimator=mlpc01, n_jobs=-1)
    mlpc01_c.fit(X_train, y_train)
    mlpc01_predictions = mlpc01_c.predict(X_test)
    print("MLP Classifier (.01) |", metrics.balanced_accuracy_score(y_test, mlpc01_predictions))

#    mlpc001 = neural_network.MLPClassifier(alpha=.001, max_iter=1000)
#    mlpc001_c = CalibratedClassifierCV(base_estimator=mlpc001, n_jobs=-1)
#    mlpc001_c.fit(X_train, y_train)
#    mlpc001_predictions = mlpc001_c.predict(X_test)
#    print("MLP Classifier (.001) |", metrics.balanced_accuracy_score(y_test, mlpc001_predictions))

#    mlpc0001 = neural_network.MLPClassifier(alpha=.0001, max_iter=1000)
#    mlpc0001_c = CalibratedClassifierCV(base_estimator=mlpc0001, n_jobs=-1)
#    mlpc0001_c.fit(X_train, y_train)
#    mlpc0001_predictions = mlpc0001_c.predict(X_test)
#    print("MLP Classifier (.0001) |", metrics.balanced_accuracy_score(y_test, mlpc0001_predictions))

#    gpc = gaussian_process.GaussianProcessClassifier(n_jobs=-1)
#    gpc_c = CalibratedClassifierCV(base_estimator=gpc, n_jobs=-1)
#    gpc_c.fit(X_train, y_train)
#    gpc_predictions = gpc_c.predict(X_test)
#    print("GP Classifier |", metrics.balanced_accuracy_score(y_test, gpc_predictions))

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

    models = [mlpca_c, mlpc1_c, mlpc01_c]
    labels = ["MLP Classifier (Adaptive)", "MLP Classifier (.1)", "MLP Classifier (.01)", "MLP Classifier (GridSearch)"]
    return models, labels, throws_data


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
