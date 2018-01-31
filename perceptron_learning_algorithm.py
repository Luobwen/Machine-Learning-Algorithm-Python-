import numpy as np
import matplotlib.pyplot as plt


class PLA(object):
    """
    Perceptron Learning Algorithm for two classes in two dimensions


    """

    def __init__(self, eta, iteration):
        self.eta = eta
        self.iteration = iteration

    def initialize_w(self):
        """
        initialize parameter w=[w0, w1, w2]
        w2 stands for b
        """
        w = np.random.rand(1, 3)  # w is a row vector
        return w

    def stack_x(self, X1, X2):
        """
        use method np.vstack to stack two different classes' input matrix together
        """
        X = np.vstack((X1, X2))
        return X

    def initialize_x(self, X):
        """
        initialize X: add a new column which equals to 1
        """
        row, col = np.shape(X)
        Matrix_one = np.ones((row, 1))
        X = np.hstack((X, Matrix_one))
        return X

    def algorithm_pla(self, X1, X2, Y):
        """
        execute perceptron learning algorithm to update w until finish the iteration
        --------------
        X : each row is a input, each column stands for a feature
        """
        w = self.initialize_w()
        xx = np.linspace(-5, 5, 100)
        yy = (-1 * w[:, 1] / w[:, 0]) * xx - w[:, 2] / w[:, 0]
        plt.plot(xx, yy, 'g')
        X = self.stack_x(X1, X2)
        X = self.initialize_x(X)
        row, col = np.shape(X)
        for i in range(self.iteration):

            j = np.random.randint(0, row - 1)
            if X[j].dot(w.T) * Y[j] < 0:
                w += self.eta * X[j] * Y[j]
        xx = np.linspace(-5, 5, 100)
        yy = (-1 * w[:, 1] / w[:, 0]) * xx - w[:, 2] / w[:, 0]
        plt.scatter(X1[:, 0], X1[:, 1])
        plt.scatter(X2[:, 0], X2[:, 1])
        plt.plot(xx, yy, 'r')
        plt.show()


if __name__ == "__main__":
    X1 = 1 * np.random.randn(1000, 2) + (0, -1)
    X2 = 1 * np.random.randn(1000, 2) + (3, 3)
    Y = np.vstack((np.ones((1000, 1)), -1 * np.ones((1000, 1))))
    eta = 0.1
    iteration = 1000
    PLA(eta, iteration).algorithm_pla(X1, X2, Y)