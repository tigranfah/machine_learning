import numpy as np

from dataclasses import dataclass, field


class DistMeasures:

	@staticmethod
	def euclidean(v1, v2):
		return np.linalg.norm(v1 - v2)

	@staticmethod
	def manhattan(v1, v2):
		return np.sum(np.abs(v1 - v2))

	@staticmethod
	def minkowski(v1, v2, degree):
		return np.sum((v1 - v2)**degree) ** (1/degree)


@dataclass(init=True)
class ClusteringBase:

	cluster_centers_ : list = field(default_factory=list)
	c_size : int = None
	dist_measure : object = None
	scores : list = field(default_factory=list)

	def reset(self):
		self.cluster_centers_ = []
		self.c_size = None
		self.dist_measure = None
		self.scores = []

	def __init_alg(self):
		pass

	def fit(self):
		pass


class KMeans(ClusteringBase):

	def __init__(self):
		super().__init__()
	
	def __init_alg(self, X, c_size, dist_m):

		self.c_size = c_size
		self.dist_measure = dist_m

		for j in range(c_size):
			for i in range(X.shape[1]):
				self.cluster_centers_.append(np.random.uniform(X[:, i].min(), X[:, i].max(), 1))
		self.cluster_centers_ = np.array(self.cluster_centers_).reshape(c_size, -1)

	def fit(self, X, c_size, n_iters=100, dist_measure=DistMeasures.euclidean):
		self.reset()
		self.__init_alg(X, c_size, dist_measure)

		for i in range(n_iters):
			dists = self.predict_dists(X)
			self.scores.append(np.sum(dists.min(axis=0)))
			clusters = dists.argmin(axis=0)
			#print(clusters)
			unique_labels = np.unique(clusters)
			for l in unique_labels:
				inds = (clusters == l)
				self.cluster_centers_[l] = X[inds].mean(axis=0)

	def predict_dists(self, X):
		dists = []
		for center in self.cluster_centers_:
			dists.append([self.dist_measure(center, x) for x in X])
		return np.array(dists)	

	def predict(self, X):
		dists = []
		for center in self.cluster_centers_:
			dists.append([self.dist_measure(center, x) for x in X])
		dists = np.array(dists)
		return dists.argmin(axis=0)


class KMeansPlusPlus(KMeans):

	def __init_alg(self, X, c_size, dist_m):

		self.c_size = c_size
		self.dist_measure = dist_m

		for i in range(X.shape[1]):
			self.cluster_centers_.append(np.random.uniform(X[:, i].min(), X[:, i].max(), 1))

		self.cluster_centers_ = np.array(self.cluster_centers_).reshape(1, -1)

		for i in range(1, c_size):
			dists = self.predict_dists(X)
			centers = X[np.sum(dists, axis=0).argmax()]
			self.cluster_centers_ = np.vstack((self.cluster_centers_, centers))
