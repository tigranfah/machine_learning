import numpy as np

from dataclasses import dataclass


class MeasureFunctions:

	@staticmethod
	def distances_measure(x, x_):
		dist = (x - x_) ** 2
		return np.sqrt(np.abs(np.sum(dist, axis=1)))


@dataclass(init=True, repr=True)
class k_means:

	"""k-means model for unsupervised learning."""

	n_clusters: int
	# similarity measure function
	simil_meas_func: object

	def train(self, data, sample_nks=None, sample_nk_limits=None):

		self.loss = []
		self.r = np.zeros((len(data), self.n_clusters))

		if sample_nk_limits:
			nk_limits = sample_nk_limints
		else: nk_limits = data.min(), data.max()

		self.nk = (sample_nks if sample_nks 
				else np.random.uniform(0, 30, 
									size=(self.n_clusters, 
									data.shape[1])))
		self.EM(data)

		return self

	def EM(self, data):
		for _ in range(5):
			self.__expectation(data)
			self.__minimization(data)

	def predict(self, test_data):
		pass

	def loss(self, data):
		for n in data

	def __expectation(self, data):
		rns = np.array([self.simil_meas_func(x, self.nk) for x in data])
		minim_dis_indices = np.argmin(rns, axis=1)
		self.r[:] = 0
		for i in range(len(minim_dis_indices)):
			self.r[i, minim_dis_indices[i]] = 1

	def __minimization(self, data):
		binary_columns = self.r.T
		k_cluster_xs = [np.where(column == 1) 
						for column in binary_columns]
		for i in range(len(self.nk)):
			self.nk[i] = data[k_cluster_xs[i]].mean(axis=0)
