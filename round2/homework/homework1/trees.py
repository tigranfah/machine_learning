from dataclasses import dataclass

import numpy as np

@dataclass(init=True, repr=True)
class DecisionNode:
    
    X: np.ndarray
    y: np.ndarray
    split_value: float = None
    index: int = None
    left_node = None
    right_node = None
    is_leaf: bool = False


@dataclass(init=True, repr=True)
class DecisionTree:
    
    entropy: object
    leaf_count: int
    max_depth: int
    sepr: int = 50
    
    def train(self, X, y):
        self.__max_depth_count = 0
        self.__leaf_count = 0
        self.h = []
        
        self.tree = self.add_node(DecisionNode(X, y, is_leaf=False))
        
        return self
    
    def add_node(self, node):
        
        self.__max_depth_count += 1
        
        if (self.max_depth <= self.__max_depth_count 
            or self.leaf_count <= self.__leaf_count):
            self.__leaf_count += 1
            node.is_leaf = True
            return node
        
        node.split_value, node.index = self.__find_best_split(node.X, node.y)
        
        split_mask = (node.X >= node.split_value)[:, node.index]
        
        if (set([len(node.X[~split_mask]), 
                 len(node.X[split_mask])])
            & set([0, 1])):
            self.__leaf_count += 1
            node.is_leaf = True
            return node
        
        left_node = DecisionNode(node.X[~split_mask], node.y[~split_mask], 
                        is_leaf=False)
        right_node = DecisionNode(node.X[split_mask], node.y[split_mask], 
                        is_leaf=False)
        
        node.left_node = self.add_node(left_node)
        node.right_node = self.add_node(right_node)
        
        return node
    
    def score(self, y_train, y_test):
        return np.sum(self.predict(y_train) == y_test) / len(y_train)
    
    def predict(self, y_train):
        return np.array([self.__predict_with_node(x, self.tree) for x in y_train])
        
    def __find_best_split(self, X, y):
        best_h = float('inf')
        best_split = None
        best_index = None
        
        for i in range(len(X.T)):
            for split in np.linspace(X.T[i].min(), X.T[i].max(), self.sepr)[1:-1]:
                mask = (X >= split)[:, i]
                h1 = self.entropy(X[mask], y[mask])
                h2 = self.entropy(X[~mask], y[~mask])
                if h1 + h2 < best_h:
                    best_h, best_split, best_index = h1 + h2, split, i

        self.h.append(best_h)
        return best_split, best_index
    
    def __predict_with_node(self, x, node):
        if node.is_leaf:
            classes, counts = np.unique(node.y, return_counts=True)
            return classes[np.argmax(counts)]
        
        pred_f = self.__predict_with_node
        return (pred_f(x, node.right_node) 
                if x[node.index] >= node.split_value
               else pred_f(x, node.left_node))