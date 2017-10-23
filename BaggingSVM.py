# coding=utf-8
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class BaggingSVM(object):


    def __init__(self, bootstrap_samples_num, bootstrap_loop_num, C):


        self.bootstrap_samples_num = bootstrap_samples_num
        self.bootstrap_loop_num = bootstrap_loop_num
        self.C = C


    def Bootstrap(self, X, Y, ):
        """
                Do Bootsrap sampling for data X and Y.

                Parameters
                ----------
                 X:  array of shape [n_samples, n_features]
                    The samples of data.

                Y:  array of shape [n_samples, ]
                    The labels of data.


                Returns
                -------
                X_in_bag: dict
                    The samples of different bootstrap loop in bag.

                Y_in_bag: dict
                    The lables of different bootstrap loop in bag.

                X_out_bag: dict
                    The samples of different bootstrap loop out bag.

                Y_out_of_bag: dict
                    The lables of different bootstrap loop in bag.

                """

        # dataset size or the number of samples
        m = X.shape[0]
        # the number of variables or features
        n = X.shape[1]

        X_in_bag = defaultdict()
        Y_in_bag = defaultdict()
        X_out_bag = defaultdict()
        Y_out_bag = defaultdict()

        ll_indx = np.random.randint(0, m, size=(self.bootstrap_loop_num, m))
        for loop in range(self.bootstrap_loop_num):
            X_in_bag[str(loop)] = X[ll_indx[loop]]
            Y_in_bag[str(loop)] = Y[ll_indx[loop]]

            X_out_bag[str(loop)] = np.delete(X, list(set(ll_indx[loop])), 0)
            Y_out_bag[str(loop)] = np.delete(Y, list(set(ll_indx[loop])))

        return X_in_bag, Y_in_bag, X_out_bag, Y_out_bag

    def fit(self, X, Y):
        """
        Train the WeightedCeBagSVM for dataset.

        Parameters
        ----------
        X: array of shape [n_samples, n_features]
            The samples of dataset.

        Y: array of shape [n_samples, ]
            The lables of dataset.
        """

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        # print('fit:',X.shape)
        self.X_in_bag, self.Y_in_bag, self.X_out_bag, self.Y_out_bag = self.Bootstrap(X, Y)
        self.clf = defaultdict()
        for loop in range(self.bootstrap_loop_num):
            self.clf[str(loop)] = SVC(C=self.C).fit(self.X_in_bag[str(loop)],self.Y_in_bag[str(loop)])

    def predict(self, X):
        """
        Predict the lables of given samples.

        Parameters
        ----------
        X: array of shape [n_samples, n_features]

        Returns
        -------
        Predictions: array of [n_samples, ]
            The predicted lables of given samples.
        """


        X = self.scaler.transform(X)
        P = []
        # print("Predict:", X.shape)
        for loop in range(self.bootstrap_loop_num):
            P.append(self.clf[str(loop)].predict(X))

        Predictions = [0 if np.sum(i) < self.bootstrap_loop_num / 2.0 else 1 for i in  np.matrix(P).T]

        return np.array(Predictions)


    def score(self,X, Y):
        """
        Compute the accuracy of classifier on given samples.

        Parameters
        ----------

        X: array of shape [n_samples, n_features]
            Given samples.
        Y: array of shape [n_samples, n_features]
            Given lables.

        Returns
        -------
        acc: float
            The accuracy of classifier on given samples.
        """

        Y_pred = self.predict(X)
        return np.sum(np.array(Y) == np.array(Y_pred)) / len(Y)
