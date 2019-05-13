import numpy as np

from typing import List
from dataclasses import dataclass

from sklearn.model_selection import train_test_split


@dataclass(init=True, repr=True)
class Blender:

	models: List
	blender_estimator: object
	blend_size: float = 0.4

	def train(self, X, y, random_state=None):

		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=self.blend_size,
			random_state=random_state
		)

		model_predictions = []
		for i in range(len(self.models)):
			self.models[i] = self.models[i].fit(X_train, y_train)
			model_predictions.append(
					self.models[i].predict_proba(X_test)[:, 1])

		# predictions that are 
		# treated as features for blender
		predictions = np.hstack([
				pred.reshape(-1, 1)
				for pred in model_predictions])

		self.blender_estimator.fit(predictions, y_test)

	def predict(self, X_test):
		prob = [model.predict_proba(X_test)[:, 1]
				for model in self.models]

		predictions = np.hstack([
				pred.reshape(-1, 1)
				for pred in prob
		])
		return self.blender_estimator.predict(predictions)

	def score(self, X_test, y_test):
		return np.sum(self.predict(X_test) == y_test) / len(X_test)


@dataclass(init=True, repr=True)
class Stacker:
	"""Stacker class."""
