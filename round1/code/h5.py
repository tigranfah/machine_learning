import numpy as np
from scipy.stats import multivariate_normal
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

class NormalNaiveBayes:
    """
    Normal naive bayes implementation
    """
    def __init__(self):
        self.mean = []
        self.cov = []
        self.p_c = []

    def train(self, data):
        num_points = sum([len(datum) for datum in data])
        for datum in data:
            self.cov.append(self.calc_cov(datum.T))
            self.mean.append(np.mean(datum, axis=0))
            self.p_c.append(len(datum) / num_points)

    @staticmethod
    def multivariate_normal(x, mean, cov):
        # todo, implement multivariate normal probability density
        return multivariate_normal.pdf(x, mean=mean, cov=cov)

    @staticmethod
    def calc_cov(datum):
        # todo, implement covariance (bonus, do research how to do it)
        return np.cov(datum)

    def predict(self, x):
        return [self.multivariate_normal(x, self.mean[i], self.cov[i])
                          for i in range(len(self.mean))]

    def plot_class_areas(self):
        # after train you know maximum and minimum for each feature
        # create plot where each color corresponds to each class using
        # feature1 and feature 2
        # todo, (bonus)
        pass

if __name__ == "__main__":
	nnb = NormalNaiveBayes()
	nnb.train(X_train)
	# for datum, klass in zip(X_test, y_test):
	print(nnb.predict(X_test[0]))
