# clustering models' implementations

import numpy as np
from scipy.stats import multivariate_normal

from dataclasses import dataclass, field


def silhouette(clust_alg, X_train, X_test):
	clusters = clust_alg.predict(X_train)

	disimilarity_from_clusters = []

	for j in np.unique(clusters):
		inds = (clusters == j)
		test_dis = []
		for x_te in X_test:
			cluster_dist = np.sum([clust_alg.dist_measure(x_te, x_tr) for x_tr in X_train[inds]])
			test_dis.append(cluster_dist * (1/len(inds)))
		disimilarity_from_clusters.append(test_dis)

	disimilarity_from_clusters = np.array(disimilarity_from_clusters)
	# take first second closest cluster values
	disimilarity_from_clusters.sort(axis=0)
	closest_clusters = disimilarity_from_clusters[:2]
	if len(closest_clusters) == 1:
		return np.zeros(len(closest_clusters[0]))
	silhouette_score = (closest_clusters[1] - closest_clusters[0]) / closest_clusters[1]
	return silhouette_score


def softmax(X):
	return np.exp(X) / np.sum(np.exp(X))


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
		dists = self.predict_dists(X)
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


# ------------

# this algorithm is not tested
class KMedoid(ClusteringBase):

	def __init_alg(self, X, c_size):
		self.c_size = c_size
		self.cluster_centers_ = np.random.choice(X, c_size, replace=False)

	def fit(self, X, c_size, n_iters=100):
		self.reset()
		self.__init_alg(X, c_size)

		for i in range(n_iters):
			dists = self.predict_dists(X)
			clusters = dists.argmin(axis=0)
			unique_labels = np.unique(clusters)
			for l in unique_labels:
				inds = (clusters == l)
				closest_centroid_id = np.argmin(np.abs(X[inds] - X[inds].mean()))
				self.cluster_centers_ = X[closest_centroid_id]

	def predict_dists(self, X):
		dists = []
		for center in self.cluster_centers_:
			dists.append(np.abs(center - X))
		return np.array(dists)

	def predict(self, X):
		dists = self.predict_dists(X)
		return dists.argmin(axis=0)

# ------------


class GuassianMixturesModel(ClusteringBase):

	def __init__(self, use_k_means_to_init=True):
		super().__init__()
		"""
			The following model is using EM algoritm to do clustering.
			I use k_means to initialize the centroids for GMM as it is common,
			other options are not implemented.

			Additional sources that helped with implementation.
			https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf
			https://towardsdatascience.com/understanding-the-covariance-matrix-92076554ea44
		"""
		self.use_k_means_to_init = use_k_means_to_init

		# shows covariance between features of each cluster
		self.cov_matrices = None

		# is the probability distribution of how probable is a new unknown point to be from each cluster
		self.pi_prob = None

		# prob dist of how likely a point is from each cluster
		self.responsibilities = None

	def reset(self):
		super().reset()
		self.pi_prob = None
		self.cov_matrices = None
		self.responsibilities = None

	def __init_alg(self, X, c_size, n_iters, dist_m):

		self.c_size = c_size
		self.distance_measure = dist_m

		k_means = KMeansPlusPlus()
		k_means.fit(X, c_size, n_iters, dist_m)
		self.cluster_centers_ = k_means.cluster_centers_

		unique_y, counts = np.unique(k_means.predict(X), return_counts=True)
		self.pi_prob = np.ones((c_size, )) * 1e-5
		for l, v in zip(unique_y, softmax(counts)):
			self.pi_prob[l] = v

		self.cov_matrices = [1*np.eye(X.shape[1]) for _ in range(self.c_size)]

		self.responsibilities = np.zeros((len(X), c_size))

	def fit(self, X, c_size, n_iters=100, dist_measure=DistMeasures.euclidean, k_means_n_iters=100):

		# here are implemented the formulas in lecture slides

		if not self.use_k_means_to_init:
			return NotImplemented

		self.reset()
		self.__init_alg(X, c_size, k_means_n_iters, dist_measure)

		for n in range(n_iters):

			## updare responsibilities

			for j in range(self.c_size):
				# multivariate_normal.pdf function estimates the density of gaussian
				summed_de = np.sum(self.pi_prob[j] * multivariate_normal.pdf(X, self.cluster_centers_[j], self.cov_matrices[j], allow_singular=True))
				#self.responsibilities[:, j] = self.pi_prob[j] * multivariate_normal.pdf(X, self.cluster_centers_[j], self.cov_matrices[j]) / summed_de

				for i in range(len(self.responsibilities)):
					pre_res = self.pi_prob[j] * multivariate_normal.pdf(X[i], self.cluster_centers_[j], self.cov_matrices[j], allow_singular=True)
					pre_res /= summed_de
					self.responsibilities[i][j] = pre_res

				self.responsibilities[:, j] = self.responsibilities[:, j] / self.responsibilities[:, j].max()

			self.responsibilities = np.array([softmax(x) for x in self.responsibilities])
			#print(self.responsibilities)
	
			y_pred = self.predict(X)
			unique_labels = np.unique(y_pred)

			## update cluster centers

			for j, l in zip(range(self.c_size), unique_labels):
				inds = (y_pred == l)
				mul_res = np.zeros((X.shape[1], ))
				for i, x in enumerate(X[inds]):
					mul_res = mul_res + self.responsibilities[inds][i][j] * x
				self.cluster_centers_[j] = mul_res / np.sum(self.responsibilities[inds][:, j])

			## update covariance matrices

			for j, l in zip(range(self.c_size), unique_labels):
				inds = (y_pred == l)
				cov_mat = np.zeros((X.shape[1], X.shape[1]))
				for i, x in enumerate(X[inds]):
					mat = np.dot((x - self.cluster_centers_[j]).reshape(-1, 1), (x - self.cluster_centers_[j]).reshape(1, -1))
					cov_mat = cov_mat + self.responsibilities[inds][i][j] * mat
				self.cov_matrices[j] = cov_mat / np.sum(self.responsibilities[:, j])
			#print("Cov matrix")
			#print(self.cov_matrices)

			## updatre prior probailities

			for j in range(self.c_size):
				self.pi_prob[j] = (1 / len(X)) * np.sum(self.responsibilities[:, j])

			#print("centers", self.cluster_centers_)
			#print("pi", self.pi_prob)

	def predict_prob(self, X):
		density_est = np.array([multivariate_normal.pdf(X, self.cluster_centers_[j], self.cov_matrices[j], allow_singular=True) for j in range(self.c_size)]).T
										# devide by its max value to normalize and get more legitimate prob.
		return np.array([softmax(x) for x in (density_est / density_est.max())])

	def predict(self, X):
		return np.argmax(self.predict_prob(X), axis=1)
