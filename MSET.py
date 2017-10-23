#coding=utf-8
import numpy as np
from numpy.linalg import norm

class MSET(object):

    def __init__(self, T, memory_matrix_size):
        self.T = T
        m = self.T.shape[1]
        if m < memory_matrix_size:
            self.memory_matrix_size = m
        else:
            self.memory_matrix_size = memory_matrix_size


    def generate_memory_matrix(self):
        """
        Generate memory matrix.
        """

        # data size
        m = self.T.shape[1]

        # 求出每个变量的最大值和最小值， 并取出对应的列
        max_index = []
        min_index = []

        for variable in self.T:
            max_index.append(np.argmax(variable))
            min_index.append(np.argmin(variable))

        D1_index = np.unique(np.append(max_index, min_index))

        if len(D1_index) < self.memory_matrix_size:

            # 每个采样时刻的 2 范数， 并按照升序排列
            norm2_of_T = np.zeros((m, 2))

            for index, sample in enumerate(self.T.T):
                norm2_of_T[index, :] = [norm(sample, ord=2), index]

            # 求出相应的采样间隔时间
            Ts = int(m / (self.memory_matrix_size - len(D1_index)))

            # 按照一定的间隔， 取出 T 中对应的采样点
            D2_index = []
            for i in range(1, m, Ts):
                D2_index.append(
                    np.int64(np.sort(norm2_of_T, axis=0)[i, 1])
                )
            D_index = np.unique(np.append(D1_index, D2_index))
        else:
            D_index = D1_index

        if len(D_index) < self.memory_matrix_size:

            add_size = self.memory_matrix_size - len(D_index)
            add_index = np.random.randint(0, self.memory_matrix_size, add_size)
            D_index = np.append(D_index, add_index)

        D_index = D_index[:self.memory_matrix_size]#.astype(dtype='int64')



        self.D = self.T[:, D_index]
        self.inv_of_D_DT = np.linalg.pinv(self.kernel(self.D.T, self.D))

    def kernel(self, X, Y):
        """
        Do kernel operation between in matrix X anf matrix B.

        Parameters
        ----------

        X: array of shape [n, m]
            A matrix.
        Y: array of shape [m, p]

        Returns
        -------
        kernel matrix: array of shape [n, p]
            A matrix.
        """

        ker_matrix = np.zeros_like(X @ Y)

        for i in range(X.shape[0]):
            for j in range(Y.shape[1]):
                ker_matrix[i, j] = norm(X[i, :] - Y[:, j])

        return np.matrix(ker_matrix)

    def estimater(self, X_obs):
        """
        Estimate the similarity of given samples.

        Parameters
        ----------

        X_obs: array of shape [n_features, n_samples]
            Given samples.

        Returns
        -------
        similarity: array of shape [n_samples, ]
            The similarity of given samples.
        """


        X_est = self.D @ (self.inv_of_D_DT @ self.kernel(self.D.T, X_obs.T))
        distances = []
        for vectorObs, vectorEst in zip(X_obs, X_est.T):
            distances.append(norm(vectorObs-vectorEst, ord=2))
        distances = np.array(distances)
        similarity = 1 / (1 + distances)

        return X_est, similarity

