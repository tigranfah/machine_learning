import numpy as np


def grid_search(estimator, entropy, params, X, y, folds=5):
    fold_size = len(X) // folds
    
    for fold in range(folds):
        mask = np.ones(len(X), dtype=bool)
        mask[fold * fold_size : ((fold + 1) * fold_size)] = False
        
        x_train, train_y = X[mask], X[~mask] 
        x_test, test_y = y[mask], y[~mask]
        
        models = [
            estimator(entropy=entropy, **param).train(x_train, x_test)
            for param in params
        ]
        
        models_and_scores = [
            (model, model.score(train_y, test_y)) 
            for model in models
        ]
        
    models_and_scores.sort(key=lambda x:x[1], reverse=True)
    return models_and_scores[0]


class FunctionSelection:
    
    @staticmethod
    def cross_entropy(x, y):
        # SUM p(x) * log2(x)
        classes, counts = np.unique(y, return_counts=True)
        p = counts / len(x)
        l_p = np.array([np.log(x) if not x is 0 else 0.0 for x in p])
        return - np.sum(p * (l_p / np.log(2)))
    
    @staticmethod
    def gini(x):
        return