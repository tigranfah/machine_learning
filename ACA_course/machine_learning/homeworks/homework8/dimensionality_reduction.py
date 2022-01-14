import numpy as np

from scipy.spatial import distance
from IPython.display import clear_output


class ImpPCA:
    
    """
        PCA
    """
    
    def __init__(self, n_components):
        self.n_components = n_components
        
    def fit(self, X):
        X_meaned = X - np.mean(X, axis=0)

        cov_mat = np.cov(X_meaned, rowvar=False)

        eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)

        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]

        self.eigenvector_subset = sorted_eigenvectors[:,0:self.n_components]
        
    def transform(self, X):
        X_meaned = X - np.mean(X, axis=0)
        return np.dot(self.eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
    

"""t-SNE implementation"""

def cross_entropy(p):
    return - np.sum(p * np.log(p, where=(p != 0)))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def distance_euclidean(x, X):
    if len(X.shape) == 2:
        ax=1
    else: ax=0
    return np.linalg.norm(x - X, axis=ax)

def KL_divergence(P, Q):
    oper_ind = (P != 0) & (Q != 0)
    return np.sum(P * np.log(P / Q, where=oper_ind))


class ImpTSNE:
    
    """
        t-SNE
    """
    
    def __init__(self, n_dim, dist_measure=distance_euclidean):
        self.n_dim = n_dim
        self.dist_measure = dist_measure
        # calculated std for each datapoint
        self.std_data_points = None
        self.perplexity = None
        self.losses = []
        
    def __init_alg(self, X, perp, std_init=1e-1):
        self.X = X
        self.std_data_points = np.ones(len(self.X)) * std_init
        self.perplexity = perp
        self.losses = []
        
    def reset(self):
        self.__init_alg(self.X, self.n_dim, self.perplexity)
    
    def fit(self, X, perp, learning_rate=0.1, max_iter=100, std_init=1e-1, display_func=None):
        self.__init_alg(X, perp, std_init)
        
        self.__binary_search()
        
        self.__symmetric_P()
        
        self.__gradient_descent(max_iter, learning_rate, display_func)
    
        print("Finished.")
        
    def __binary_search(self):
        # find std for every datapoint with binary_search
        for i in range(len(self.X)):
            print()
            if not self.binary_search_std(i, self.perplexity):
                raise Exception(f"Could not find std for point {self.X[i]} at index {i}.")
            else:
                clear_output(wait=True)
                print(f"Found std for {i+1}.", end="\r")
                
    def __symmetric_P(self):
        """
            make P symmetrical, 
            that is gonna be the probabilities that we want to achieve in the lower dimention
        """
        self.P = []
    
        for i, x in enumerate(self.X):
            self.P.append([])
            for j, y in enumerate(self.X):
                factor = 1/(2)
                self.P[-1].append(
                    factor * (
                    self.choose_probability_guassian(x, y, self.std_data_points[i])
                    +
                    self.choose_probability_guassian(y, x, self.std_data_points[j])
                    )
                )
                
        self.P = np.array(self.P)
        self.P = np.array([softmax(p/p.max()) for p in self.P])

    def __gradient_descent(self, max_iter, learning_rate, display_func):
        
        # use gradient descent to find y_ in n dimention
        ## note: instead of guassian we use t-dist for Y_
        
        self.Y_ = np.random.normal(0, 1e-4, (len(self.X), self.n_dim))
        
        for n in range(max_iter):
            Q = ImpTSNE.choose_probability_t_dist(self.Y_, self.dist_measure)
            Q = np.array([softmax(q/q.max()) for q in Q])
            self.losses.append(
                np.sum([KL_divergence(self.P[k], Q[k]) for k in range(len(Q))])
            )
            print(f"Iter {n+1}, loss {self.losses[-1]}", end="\r")
            
            # gradient descent
            for i in range(len(self.Y_)):
                partial_der_i = np.sum([
                    (self.P[i][j] - Q[i][j]) * (self.Y_[i] - self.Y_[j]) * ((1 + self.dist_measure(self.Y_[i], self.Y_[j])) ** -1) 
                     for j in range(len(self.Y_))
                ], axis=0)
#                 print(partial_der_i)
                self.Y_[i] -= learning_rate * partial_der_i
    
            if display is not None:
                clear_output(wait=True)
                display_func(self.Y_)
        
    def choose_probability_guassian(self, x, y, std):
        """
            the probability that i will choose j as its neighbour
        """
        if np.all(x == y): return 0
        num = - self.dist_measure(x, y) / (2 * (std ** 2))
        denum = - self.dist_measure(x, self.X) / (2 * (std ** 2))
        return np.exp(num) / np.sum(np.exp(denum))
    
    def calc_perplexity(self, i):
        p_dist = []
        for j in range(len(self.X)):
            if i != j:
                p_dist.append(self.choose_probability_guassian(self.X[i], self.X[j], self.std_data_points[i]))
            else: p_dist.append(0)
        return 2 ** cross_entropy(np.array(p_dist))
    
    def binary_search_std(self, i, perp, error=0.1):
        lower_bound = 0
        upper_bound = np.inf
        
        factor = 0.5
        
        while True:
            curr_perp = self.calc_perplexity(i)
            if abs(curr_perp - perp) <= 0 + error:
                return True
            
            if curr_perp > perp:
                upper_bound = self.std_data_points[i]
                self.std_data_points[i] -= (upper_bound - lower_bound) * factor
            if upper_bound == np.inf:
                self.std_data_points[i] *= 2
            elif curr_perp < perp:
                lower_bound = self.std_data_points[i]
                self.std_data_points[i] += (upper_bound - lower_bound) * factor
                
        return False
    
    @staticmethod
    def choose_probability_t_dist(Y_, dist=distance_euclidean):
        
        denum = []
        for x in Y_:
            denum.append(np.sum( [(1 + dist(x, k)) ** -1 for k in Y_ if np.all(k != x)] ))
            
        denum = np.sum(denum)
        
        output = []
        
        for x in Y_:
            output.append([])
            for y in Y_:
                num = (1 + dist(x, y)) ** -1
                output[-1].append(num / denum)
            
        return np.array(output)
        