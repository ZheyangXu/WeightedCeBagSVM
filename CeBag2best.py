# coding=utf-8
import numpy as np
import pandas as pd

import time
 os

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from WeightedCeBagSVM import WeightCeBagSVM
from CeBagSVM import CeBagSVM
from BaggingSVM import BaggingSVM

def Cebag2best(X, Y, name, debug=False):

    C = 20  #惩罚因子
    C_step = 1 #步长

    kf = KFold(n_splits=5)

    m = X.shape[0]

    train_score = []
    dev_score = []

    model = []
    for i in np.arange(0, 3.01, 0.1):
        times1 = time.time()
        bootstrap_loop_num = int(10 ** i)

        if int(10 ** i) == int(10 ** (i + 0.1)):
            continue
        if debug:
            if int(10 ** i) > 3:
                break
        for j in np.arange(0.2, 1.01, 0.1):

            bootstrap_samples_num = int(j * m)
            score0 = []
            score1 = []
            for train, dev in kf.split(X, Y):
                X_train, Y_train, X_dev, Y_dev = X[train, :], Y[train], X[dev, :], Y[dev]

                CeBag = CeBagSVM(bootstrap_samples_num, bootstrap_loop_num, C, C_step)
                CeBag.fit(X_train, Y_train)
                score0.append(CeBag.score(X_train, Y_train))
                score1.append(CeBag.score(X_dev, Y_dev))
            model.append(CeBag)
            train_score.append([int(10 ** i), bootstrap_samples_num, np.average(score0)])
            dev_score.append([int(10 ** i), bootstrap_samples_num, np.average(score1)])

            print(name + ": This %s loop, bootstrap loops number is %s, bootstrap samples number is %s, the cost is : %s"\
              % (i, int(10 ** i), bootstrap_samples_num, time.time()-times1))
            print("---------------------------------------------------------------------------------------------")

        pd.DataFrame(train_score).to_csv(name + ' train_score.csv', header=None)
        pd.DataFrame(dev_score).to_csv(name + ' dev_score.csv', header=None)



debug = False

breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target
Cebag2best(X, Y, 'B.Cancer', debug)

German = pd.read_csv('datasets/German.csv', header=None).values
X = German[:, :-1]
Y = np.array([1 if i == 1 else 0 for i in German[:, -1]])
Cebag2best(X, Y, 'German', debug)

Heart = pd.read_csv('datasets/Heart.csv', header=None).values
X = Heart[:, :-1]
Y = np.array([1 if i == 1 else 0 for i in Heart[:, -1]])
Cebag2best(X, Y, 'Heart', debug)

Twonorm = pd.read_csv('datasets/twonorm.csv', header=None).values
X = Twonorm[:, :-1]
Y = np.array([1 if i == 1 else 0 for i in Twonorm[:, -1]])
Cebag2best(X, Y, 'Twonorm', debug)

Banana = pd.read_csv('datasets/banana_data.csv', header=None).values
X = Banana[:, 1:]
Y = np.array([1 if i == 1 else 0 for i in Banana[:, 0]])
Cebag2best(X, Y, 'Banana', debug)
