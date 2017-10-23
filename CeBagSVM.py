# coding=utf-8
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class CeBagSVM(object):

    '''
    bootstrap_samples_num 样本数量
    # bootstrap_loop_num 抽样次数
    # C 惩罚因子  C_step 步长
    '''
    def __init__(self, bootstrap_samples_num, bootstrap_loop_num, C, C_step):


        self.bootstrap_samples_num = bootstrap_samples_num    # the number of samples in one bootstrap sample
        self.bootstrap_loop_num = bootstrap_loop_num          # the number of bootstrap samples
        self.C = C                                          # the regularization parameter
        self.C_step = C_step                                # the step of regularization parameter to search



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

        # Generate the index that are in bag
        ll_indx = np.random.randint(0, m, size=(self.bootstrap_loop_num, m))
        # Do bootstrap sampling
        for loop in range(self.bootstrap_loop_num):

            # Selecte the samples into the in-bag
            X_in_bag[str(loop)] = X[ll_indx[loop]]
            Y_in_bag[str(loop)] = Y[ll_indx[loop]]

            # Selected the samples into the out-of-bag
            X_out_bag[str(loop)] = np.delete(X, list(set(ll_indx[loop])), 0)
            Y_out_bag[str(loop)] = np.delete(Y, list(set(ll_indx[loop])))

        return X_in_bag, Y_in_bag, X_out_bag, Y_out_bag

    def Get_Accuracy(self, X, Y, clf):
        """
                Compute the accuracy of data X in a classifier.

                Parameters
                ----------
                X: array of shape [n_samples, n_features]
                    The samples of data.

                Y: array of shape [n_samples, ]
                    The lables of data.

                clf: callable
                    The classifiers that can be callable

                Returns
                -------
                acc: float
                    The accuracy of data X in a classifier.
                """

        Y_predtions = clf.predict(X)
        acc = self.Accuracy(Y, Y_predtions)

        return acc

    def Accuracy(self, Y, Y_predictions):

        int_frade_nums = 0
        int_detected_nums = 0
        int_labelled_nums = 0

        for index, y in enumerate(Y):
            if y == 1:
                int_frade_nums += 1
                if Y_predictions[index] == 1:
                    int_detected_nums += 1

        for y_pre in Y_predictions:
            if y_pre == 1:
                int_labelled_nums += 1

        accPositive = int_detected_nums / int_frade_nums
        accNegative = 1 - (int_labelled_nums - int_detected_nums) / (len(Y) - int_frade_nums)

        return accPositive, accNegative

    def Get_PE_SVM(self, X_train, Y_train, X_test, Y_test, C, C_step, AccPositive, AccNegative):

        """
        Train the positive class-wise expert(PE) for a bootstrap sample.

        Parameters
        ----------
        X_train: array of shape [n_samples, n_features]
            The samples of train set for training the PE-SVM.

        Y_train: array of shape [n_samples, ]
            The lables of train set for traing the PE-SVM.

        X_test: array of shape [n_samples, n_features]
            The samples for test set for evaluating the PE-SVM.

        Y_test: array of shape [n_samples, n_features]
            The lables of test set for evaluating the PE-SVM.

        C: float
            The regularization parameter of PE-SVM.

        C_step: float
            The step of C changing for searching a suitable PE-SVM.

        AccPositive: float
            The accuracy of individual SVM on out-of-bag of positive class respectively.

        AccNegative: float
            The accuracy of individual SVM on out-of-bag of negative class respectively.

        Returns:
        --------
        clf: object
            The classifier that positive class-wise expert PE.
        """


        while C > 0:

            # clf: the classifier
            # accPositive: pacc+
            # accNegative: pacc-

            # AccPositive: acc+
            # accNegative: acc-

            # Train a positive class-wise exoert
            clf, accPositive, accNegative = self.PE_SVM(X_train, Y_train, X_test, Y_test, C)

            # if pacc+ <= acc+ or pacc <= -1/2 acc- decrease the C or goto next step
            if accPositive > AccPositive and accNegative > 0.5 * AccNegative:
                break
            C = C - C_step

        return clf

    def PE_SVM(self, X_train, Y_train, X_test, Y_test, C):
        """
        Train a SVM for positive class-wise expert and return pacc+ and pacc-.

        Parameters
        ----------
        X_train: array of shape [n_samples, n_features]
            The samples of train set for training the PE-SVM.

        Y_train: array of shape [n_samples, ]
            The lables of train set for traing the PE-SVM.

        X_test: array of shape [n_samples, n_features]
            The samples for test set for evaluating the PE-SVM.

        Y_test: array of shape [n_samples, n_features]
            The lables of test set for evaluating the PE-SVM.

        C: float
            The regularization parameter of PE-SVM.

        Returns
        -------
        clf: object
            The classifier for a particular C.

        accPositive: float
            The test accuracy of this classifier on out-of-bag.

        accNegative: float
            The test accuracy of this classifier on out-of-bag.
        """


        clf = SVC(class_weight={1: C}).fit(X_train, Y_train)
        accPositive, accNegative = self.Get_Accuracy(X_test, Y_test, clf)

        return clf, accPositive, accNegative

    def Get_NE_SVM(self, X_train, Y_train, X_test, Y_test, C, C_step, AccPositive, AccNegative):
        """
        Train the negative class-wise expert(NE) for a bootstrap sample.

        Parameters
        ----------
        X_train: array of shape [n_samples, n_features]
            The samples of train set for training the NE-SVM.

        Y_train: array of shape [n_samples, ]
            The lables of train set for traing the NE-SVM.

        X_test: array of shape [n_samples, n_features]
            The samples for test set for evaluating the NE-SVM.

        Y_test: array of shape [n_samples, n_features]
            The lables of test set for evaluating the NE-SVM.

        C: float
            The regularization parameter of NE-SVM.

        C_step: float
            The step of C changing for searching a suitable NE-SVM.

        AccPositive: float
            The accuracy of individual SVM on out-of-bag of positive class respectively.

        AccNegative: float
            The accuracy of individual SVM on out-of-bag of negative class respectively.

        Returns:
        --------
        clf: object
            The classifier that negative class-wise expert NE.
        """

        while C > 0:

            # clf: the classifier
            # accPositive: pacc+
            # accNegative: pacc-

            # AccPositive: acc+
            # accNegative: acc-

            # Train a negative class-wise exoert
            clf, accPositive, accNegative = self.NE_SVM(X_train, Y_train, X_test, Y_test, C)

            # if pacc+ <= acc+ or pacc <= -1/2 acc- decrease the C or goto next step
            if AccNegative < accNegative and accPositive > 0.5 * AccPositive:
                break
            C = C - C_step

        return clf

    def NE_SVM(self, X_train, Y_train, X_test, Y_test, C):
        """
        Train a SVM for negative class-wise expert.


        Parameters
        ----------
        X_train: array of shape [n_samples, n_features]
            The samples of train set for training the NE-SVM.

        Y_train: array of shape [n_samples, ]
            The lables of train set for traing the NE-SVM.

        X_test: array of shape [n_samples, n_features]
            The samples for test set for evaluating the NE-SVM.

        Y_test: array of shape [n_samples, n_features]
            The lables of test set for evaluating the NE-SVM.

        C: float
            The regularization parameter of NE-SVM.

        Returns
        -------
        clf: object
            The classifier for a particular C.

        accPositive: float
            The test accuracy of this classifier on out-of-bag.

        accNegative: float
            The test accuracy of this classifier on out-of-bag.
        """


        clf = SVC(class_weight={0: C}).fit(X_train, Y_train)
        accPositive, accNegative = self.Get_Accuracy(X_test, Y_test, clf)

        return clf, accPositive, accNegative

    def Get_MT_SVM(self, X_train, Y_train):

        return SVC().fit(X_train, Y_train)

    def SVM_of_One_Bag(self, X_train, Y_train, X_test, Y_test, C, C_step):
        """
        Train the positive class-wise expert(PE), negative class-wise expert(NE) and mediator classifier(MT) for
        a bootrstrap sample.

        Parameters
        ----------
        X_train: array of shape [n_samples, n_features]
            The in-bag-samples of a bootstrap samples for training PE, NE and MT.

        Y_train: array of shape [n_samples, ]
            The in-bag-lables of a bootstrap sample for training PE, NE and MT.

        X_test: array of shape [n_samples, n_features]
            The out-bag-samples of a bootstrap sample for training PE, NE and MT.

        Y_test: array of shape [n_samples, ]
            The out-bag-lables of a bootstrap sample for training PE, NE and MT.

        C: float
            The regularization parameter of PE-SVM.

        C_step: float
            The step of C changing for searching a suitable PE-SVM.

        Returns
        -------
        clfPE: object
            The positive class-wise expert.

        clfNE: object
            The negative class-wise expert.

        clfMT: object
            The mediator classifier.
        """

        # Train an individual SVM
        clf = self.Get_MT_SVM(X_train, Y_train)
        AccPositive, AccNegative = self.Get_Accuracy(X_test, Y_test, clf)

        # Train a positive class-wise expert
        clfPE = self.Get_PE_SVM(X_train, Y_train, X_test, Y_test, C, C_step, AccPositive, AccNegative)

        # Train a negative class-wise expert
        clfNE = self.Get_NE_SVM(X_train, Y_train, X_test, Y_test, C, C_step, AccPositive, AccNegative)

        # Train a mediator classifier
        # Get the index of train set for training mediator classifier
        MT_index = clfPE.predict(X_test) != clfNE.predict(X_test)

        # Make sure the train set for training mediator classifier is not empty set,
        # if it is a empty set,
        # replace with X_train and Y_train
        if True in MT_index:
            X_of_MT = X_test[MT_index]
            Y_of_MT = Y_test[MT_index]
            #print(Y_of_MT)
            if 0. not in Y_of_MT or 1. not in Y_of_MT:
                X_of_MT = X_train
                Y_of_MT = Y_train
        else:
            X_of_MT = X_train
            Y_of_MT = Y_train

        clfMT = self.Get_MT_SVM(X_of_MT, Y_of_MT)
        return clfPE, clfNE, clfMT

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

        # data preprocessing
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # Bootstrap
        self.X_in_bag, self.Y_in_bag, self.X_out_bag, self.Y_out_bag = self.Bootstrap(X, Y)

        # The set of PE, NE and MT
        self.clfPE, self.clfNE, self.clfMT = defaultdict(), defaultdict(), defaultdict()

        # fit
        for loop in range(self.bootstrap_loop_num):
            self.clfPE[str(loop)], self.clfNE[str(loop)], self.clfMT[str(loop)] = self.SVM_of_One_Bag(
                self.X_in_bag[str(loop)],
                self.Y_in_bag[str(loop)],
                self.X_out_bag[str(loop)],
                self.Y_out_bag[str(loop)],
                self.C,
                self.C_step)

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

        # data preprocessing
        X = self.scaler.transform(X)
        P = []

        # Predict
        for loop in range(self.bootstrap_loop_num):
            P.extend([self.clfPE[str(loop)].predict(X),
                     self.clfNE[str(loop)].predict(X),
                     self.clfMT[str(loop)].predict(X)])

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
