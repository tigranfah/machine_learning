import numpy as np
from scipy.stats import multivariate_normal

from dataclasses import dataclass


class MeasureFunctions:

	@staticmethod
	def distances_measure(x, x_):
		dist = (x - x_) ** 2
		return np.sqrt(np.abs(np.sum(dist, axis=0)))

	@staticmethod
	def similar_color(c, c_):
		return np.sum(np.abs(c - c_) ** 2)


@dataclass(init=True, repr=True)
class k_means:

	"""k-means model for unsupervised learning."""

	n_clusters: int
	# similarity measure function
	simil_meas_func: object
	similar_loss_value: float = 1000
	n_similar_loss: int = 3
	sample_nks: np.ndarray = None
	sample_nk_limits: tuple = ()
	mean_func: object = lambda x: x.mean(axis=0)

	def train(self, data):

		self.loss_history = []
		self.r = np.zeros((len(data), self.n_clusters))

		if self.sample_nk_limits:
			nk_limits = self.sample_nk_limints
		else: nk_limits = data.min(), data.max()

		self.nk = (self.sample_nks if not self.sample_nks is None
				else np.random.uniform(*nk_limits, 
									size=(self.n_clusters, 
									data.shape[1])))
		self.EM(data)

		return self

	def EM(self, data):

		while True:
			self.r[:] = self.__expectation(data)
			self.nk[:] = self.__minimization(data)

			current_loss = self.loss(data)
			last_n_losses = self.loss_history[-self.n_similar_loss:]
			self.loss_history.append(current_loss)

			if len(self.loss_history) < self.n_similar_loss:
				continue

			if np.all(last_n_losses 
					  - current_loss <= self.similar_loss_value):
				break

	def predict(self, data):
		return self.__expectation(data)

	def loss(self, data):
		loss = 0
		for n in range(len(data)):
			for k in range(len(self.nk)):
				loss += self.r[n][k] * self.simil_meas_func(data[n], 
															self.nk[k])
		return loss

	def __expectation(self, data):
		rns = np.array([
				[self.simil_meas_func(x, k) for k in self.nk]
				for x in data
		])
		minim_dis_indices = np.argmin(rns, axis=1)

		r = np.zeros(self.r.shape)
		r[:] = 0

		for i in range(len(minim_dis_indices)):
			r[i, minim_dis_indices[i]] = 1
		return r

	def __minimization(self, data):
		binary_columns = self.r.T
		k_cluster_xs = [
				np.where(column == 1) 
				for column in binary_columns 
		]

		nk = np.zeros(self.nk.shape)

		for i in range(len(nk)):
			# some clusters might not have any points
			if data[k_cluster_xs[i]].size == 0:
				continue

			nk[i] = self.mean_func(data[k_cluster_xs[i]])

		return nk


@dataclass(init=True, repr=True)
class GaussiansMixtures:

	n_clusters: int
	# similarity measure function
	simil_meas_func: object
	similar_loss_value: float = 1000
	n_similar_loss: int = 3
	sample_nks: np.ndarray = None
	sample_nk_limits: tuple = ()

	def train(self, data):

		if self.sample_nk_limits:
			nk_limits = self.sample_nk_limints
		else: nk_limits = data.min(), data.max()

		self.nk = (self.sample_nks if not self.sample_nks is None
				else np.random.uniform(*nk_limits, 
									size=(self.n_clusters, 
									data.shape[1])))

		self.cov_matrix_2 = [np.eye(data.shape[1])] * self.n_clusters

		self.w = np.zeros((len(data), self.n_clusters))

		self.pi = np.array([1 / self.n_clusters] * self.n_clusters)

		self.log_likelihoods = []

		self.EM(data)

		return self

	def predict(self, data):
		return self.__expectation(data)

	def EM(self, data):

		for i in range(5):
			print('iter')

			self.w[:] = self.__expectation(data)
			self.__maximization(data)


	def __expectation(self, data):
		w = np.empty((len(data), self.n_clusters))
		for k in range(self.n_clusters):
			w[:, k] = self.pi[k] * multivariate_normal.pdf(data, 
													self.nk[k],
													self.cov_matrix_2[k])
		return w

	def __maximization(self, data):
		for k in range(self.n_clusters):

			w_sum = self.w[:, k].sum()
			self.pi[k] = w_sum / len(data)
			self.nk[k] = (self.w[:, k] * data.T).sum(axis=1) / w_sum

			data_nk_diff = np.array([x - self.nk[k]
									for x in data]) ** 2
			print((self.w[:, k] * data_nk_diff))
			self.cov_matrix_2[k] = (self.w[:, k] * data_nk_diff).sum(0) / w_sum
