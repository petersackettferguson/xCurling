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

    #mlpc = neural_network.mlpclassifier(max_iter=1000)
    #grid = model_selection.gridsearchcv(mlpc, mlpc_pg, scoring='balanced_accuracy', n_jobs=-1)
    #grid.fit(x_train, y_train)
    #print(grid.best_params_)
    #mlpc_c = calibratedclassifiercv(grid.best_estimator_, n_jobs=-1)
    #mlpc_c.fit(x_train, y_train)
    #grid_predictions = mlpc_c.predict(x_test)
    #print("accuracy: {:.4f}".format(metrics.balanced_accuracy_score(y_test, grid_predictions)))

    #rfc = ensemble.RandomForestClassifier()
    #rfc_c = CalibratedClassifierCV(base_estimator=rfc, n_jobs=-1)
    #rfc.fit(X_train, y_train)
    #rfc_c.fit(X_train, y_train)

    #svc = svm.SVC(probability=True, C=100.0)
    #svc_c = CalibratedClassifierCV(base_estimator=svc, n_jobs=-1)
    #svc.fit(X_train, y_train)
    #svc_c.fit(X_train, y_train)

    lr = linear_model.LogisticRegression(max_iter=1000, C=1000.0)
    lr_c = CalibratedClassifierCV(base_estimator=lr, n_jobs=-1)
    lr_c.fit(X_train, y_train)
    lr_predictions = lr_c.predict(X_test)
    print("Logistic Regression (C=1000) |", metrics.balanced_accuracy_score(y_test, lr_predictions))

    mlpcc = neural_network.MLPClassifier(alpha=.1, learning_rate='constant', max_iter=1000)
    mlpcc_c = CalibratedClassifierCV(base_estimator=mlpcc, n_jobs=-1)
    mlpcc_c.fit(X_train, y_train)
    mlpcc_predictions = mlpcc_c.predict(X_test)
    print("MLP Classifier (Constant) |", metrics.balanced_accuracy_score(y_test, mlpcc_predictions))
    
    mlpca = neural_network.MLPClassifier(alpha=.1, learning_rate='adaptive', max_iter=1000)
    mlpca_c = CalibratedClassifierCV(base_estimator=mlpca, n_jobs=-1)
    mlpca_c.fit(X_train, y_train)
    mlpca_predictions = mlpca_c.predict(X_test)
    print("MLP Classifier (Adaptive) |", metrics.balanced_accuracy_score(y_test, mlpca_predictions))

#    gpc = gaussian_process.GaussianProcessClassifier(n_jobs=-1)
#    gpc_c = CalibratedClassifierCV(base_estimator=gpc, n_jobs=-1)
#    gpc_c.fit(X_train, y_train)
#    gpc_predictions = gpc_c.predict(X_test)
#    print("GP Classifier |", metrics.balanced_accuracy_score(y_test, gpc_predictions))

    models = [lr_c, mlpcc_c, mlpca_c]
    labels = ["Logistic Regression (C=1000)", "MLP Classifier (Constant)", "MLP Classifier (Adaptive)"]
    return models, labels, throws_data

