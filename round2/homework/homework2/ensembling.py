import numpy as np

from typing import List
from dataclasses import dataclass

from sklearn.model_selection import train_test_split


@dataclass(init=True, repr=True)
class Blender:
    """Blender class."""
    
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
    
    models: List
    stack_estimator: object
    stack_size: float = 0.4
        
    # the code in train function 
    # definitely need to be refactored
    def train(self, X, y, folds=5, random_state=None):
        
        fold_size = len(X) // folds
        
        features = []
        
        for fold in range(folds):
            mask = np.ones(len(X), dtype=bool)
            mask[fold * fold_size : (fold+1) * fold_size] = False


            x_train, y_train = X[mask], y[mask]
            x_test, y_test = X[~mask], y[~mask]
            
            models = [
                model.fit(x_train, y_train)
                for model in self.models
            ]
            
            # features for stacker
            features.append([
                np.hstack([(
                    model.predict_proba(x_test)[:, 1].reshape(-1, 1))
                    for model in models
                ]), 
                y_test
            ])

        self.features, self.targets = zip(*features)

        self.features = np.vstack(self.features)
        self.targets = np.concatenate(self.targets)
            
        self.stack_estimator.fit(self.features, self.targets)

    def predict(self, X_test):
        features = np.hstack([(model.predict_proba(X_test)[:, 1].reshape(-1, 1))
                    for model in self.models])

        return self.stack_estimator.predict(features)

    def score(self, X_test, y_test):
        return np.sum(self.predict(X_test) == y_test) / len(X_test)