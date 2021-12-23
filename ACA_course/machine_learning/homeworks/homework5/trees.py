import numpy as np

from dataclasses import dataclass



def RSS(y_true, y_pred):
        return np.sum((y_true - y_pred)**2)

    
class Metrics:
    
    @staticmethod
    def RSS_metrics(y_true):
        y_pred = y_true.mean()
        return RSS(y_true, y_pred)
    
    @staticmethod
    def gini(y_true):
        _, counts = np.unique(y_true, return_counts=True)
        return np.sum([(c / len(y_true)) * (1 - (c / len(y_true))) for c in counts])
    
    @staticmethod
    def cross_entropy(y_true):
        _, counts = np.unique(y_true, return_counts=True)
        return - np.sum([(c / len(y_true)) * np.log(c / len(y_true)) for c in counts])
    
    @staticmethod
    def get(name):
        return {
            "RSS" : Metrics.RSS_metrics,
            "gini" : Metrics.gini,
            "cross-entropy" : Metrics.cross_entropy
        }[name]


@dataclass(init=True)
class DecisionNode:
    
    x_data : np.ndarray
    y_data : np.ndarray
    metrics : object
        
    split_value : float = None
    split_label : str = str()
    split_index : int = None
    is_leaf : bool = True
        
    split_range : int = 20
    metrics_value : float = None
        
    sub_node1 : object = None
    sub_node2 : object = None
        
    def __find_the_best_split(self, x_data, y_data, col_names):
#         print("split value", self.split_value)
#         print("split index", self.split_index)

        for i in range(x_data.shape[1]):
                                                                    # don't include min and max
            for th in np.linspace(x_data[:, i].min(), x_data[:, i].max(), self.split_range+2)[1:-1]:
                # get datapoints in threshold
                ind = (x_data[:, i] > th)
#                 if np.all(ind) or not np.all(ind):
#                     continue
                
                # make prediction compute metrics value
                met_score1 = self.metrics(y_data[ind])
                met_score2 = self.metrics(y_data[~ind])
#                 print("metrics values,", met_score1, met_score2)
                
                # update split value and index
                if self.metrics_value is None or self.metrics_value > met_score1 + met_score2:
                    self.metrics_value = met_score1 + met_score2
                    self.split_label = col_names[i]
                    self.split_value = th
                    self.split_index = i
                    
#         print("after")
#         print("split value", self.split_value)
#         print("split index", self.split_index)
            
    def split(self, col_names):
#         print()
#         print("Original", self.x_data.shape, self.y_data.shape)
        
        self.__find_the_best_split(self.x_data, self.y_data, col_names)
        
        ind = (self.x_data[:, self.split_index] > self.split_value)
#         print()
#         print(np.all(ind))
#         print(np.all(~ind))
        if np.all(ind) or np.all(~ind):
            return (self, )
        
#         print(self.x_data[ind].shape, self.y_data[ind].shape)
#         print(self.x_data[~ind].shape, self.y_data[~ind].shape)
        self.sub_node1 = type(self)(x_data=self.x_data[ind], y_data=self.y_data[ind], metrics=self.metrics)
        self.sub_node2 = type(self)(x_data=self.x_data[~ind], y_data=self.y_data[~ind], metrics=self.metrics)
        
        self.is_leaf = False
        
        return self.sub_node1, self.sub_node2
        
    def predict(self, x):
        if self.is_leaf: return self.prediction
        
        if x[self.split_index] > self.split_value:
            return self.sub_node1.predict(x)
        return self.sub_node2.predict(x)
    
    def __str__(self):
        if self.is_leaf:
            return f"DecisionLeaf (prediction : {self.prediction})"
        return f"DecisionNode (split_label : {self.split_label}, split_index : {self.split_index},  split_value : {self.split_value}, metrics_value : {self.metrics_value}), sub_nodes ({self.sub_node1}, {self.sub_node2})"
    
    def __repr__(self):
        return str(self)
    
    
class DecisionNodeRegressor(DecisionNode):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # take mean of all values
        self.prediction = self.y_data.mean()
        
        
class DecisionNodeClassifier(DecisionNode):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # take the dominante class
        data, count = np.unique(self.y_data, return_counts=True)
        self.prediction = data[count.argmax()]

    
class DecisionTree:
    
    def __init__(self, max_depth):
        self.root_node = None
        self.max_depth = max_depth
        self.leaf_list = []
        self.node_type = None
        
    def fit(self, X_train, y_train, col_names=None):
        if not col_names: col_names = [f"feature{i+1}" for i in range(X_train.shape[1])]
            
        self.root_node = self.node_type(np.array(X_train), np.array(y_train), metrics=self.metrics)
        
        self.leaf_list.append(self.root_node)
        
        for i in range(self.max_depth + 1):
#             print(self.leaf_list)
            new_leaf_list = []
            for leaf in self.leaf_list:
                
                r_leafs = leaf.split(col_names)
                new_leaf_list.extend(r_leafs)
                
            self.leaf_list.clear()
            self.leaf_list = new_leaf_list[:]
            
    def predict_one(self, x):
        return self.root_node.predict(x)
    
    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])
    
    def __str__(self):
        return str(self.root_node)
    
    def __repr__(self):
        return str(self)
    

class ImpDecisionTreeClassifier(DecisionTree):
    
    def __init__(self, max_depth=10, metrics="gini"):
        super().__init__(max_depth)
        self.node_type = DecisionNodeClassifier
        self.metrics = Metrics.get(metrics)
        
        
class ImpDecisionTreeRegressor(DecisionTree):
    
    def __init__(self, max_depth=10):
        super().__init__(max_depth)
        self.node_type = DecisionNodeRegressor
        self.metrics = Metrics.get("RSS")
            