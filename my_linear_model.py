import numpy as np


class LogisticRegression:

    def __init__(self, max_iter=100, lr=0.01):
        """
        Initialize the Learner
        :param max_iter: int, Maximum number of iterations
        :param lr: float, Learning rate
        """
        self.coef_ = 0.0
        self.intercept_ = 0.0
        self.max_iter = max_iter
        self.error_ = list()
        self.lr = lr

    @staticmethod
    def loss(a, y):
        """
        This is log loss or binary cross entropy function
        :param a: ndArray, Contains predicated values
        :param y: ndArray, Contains true values
        :return:
        """
        eps = 1e-7
        return (-y*np.log(a + eps) - (1-y)*np.log(1 - a + eps)).mean()

    @staticmethod
    def sigmoid(z):
        """
        Activation function
        :param z: ndArray, Contains linear equation's values
        :return: ndArray, Computed sigmoid values
        """
        # return 1 / (1 + np.exp(-z))
        return np.exp(np.fmin(z, 0)) / (1 + np.exp(-np.abs(z)))

    def bgd(self, x, y):
        """
        Batch Gradient Descent
        :param x: ndArray, [n_instance, n_features]
        :param y: ndArray, [n_instance, ]
        """
        z = np.dot(x, self.coef_.T) + self.intercept_
        a = self.sigmoid(z)
        self.error_.append(self.loss(a, y))

        dz = a - y.reshape(-1, 1)
        dw = (1/x.shape[0])*np.dot(dz.T, x)
        db = (1/x.shape[0]) * np.sum(dz)

        self.coef_ -= self.lr * dw
        self.intercept_ -= self.lr * db

    def fit(self, x, y):
        """
        Learner learns the data
        :param x: ndArray, [n_instance, n_features]
        :param y: ndArray, [n_instance, ]
        """
        x = np.array(x)
        y = np.array(y)

        self.coef_ = np.random.randn(1, x.shape[1]) * 0.01
        self.intercept_ = 0.0

        for epoch_number in range(self.max_iter):
            self.bgd(x, y)

    def predict(self, x):
        """
        Classifier predicts the data
        :param x: ndArray, [n_instance, n_features]
        :return: ndArray, Activation values computed over values of linear equation.
        """
        return np.array(self.sigmoid(np.dot(x, self.coef_.T) + self.intercept_), dtype=np.int32)
