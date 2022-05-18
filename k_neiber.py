from turtle import distance
import numpy as np
from collections import Counter 

def ED(x_1, x_2):  
    return np.sqrt(np.sum((int(float(x_1 )) - (int(float(x_2))) **2)))   


class KNN():
    def __init__(self,k) -> None:
        self.k = k
        
    def fit(self,X,y):
        self.X_träin = X 
        self.y_träin = y


    def predict(self,X):
        y_pred = [self.predict_most_common(x) for x in X ]
        return np.array(y_pred)
    

    def predict_most_common(self,x):
        
        distance = [ED(x, x_tr) for x_tr in self.X_träin]
        k_index = np.argsort(distance)[:self.k]

        k_neigbers_labels = [self.y_träin[c] for c in k_index]

        most_commo = Counter(k_neigbers_labels).most_common(1)

        return most_commo[0][0]       










        ####################################################


from scipy import stats

class BruteForceKNN:
    """
    Methods:
    -------
    fit: Calculate distances and ranks based on given data
    predict: Predict the K nearest self.neighbors based on problem type
    """ 
  
    

    def __init__(self, k, problem: int=0, metric: int=0):
        """
            Parameters
            ----------
            k: Number of nearest self.neighbors
            problem: Type of learning
            0 = Regression, 1 = Classification
            metric: Distance metric to be used. 
            0 = Euclidean, 1 = Manhattan
        """
        self.k = k
        self.problem = problem
        self.metric = metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


    # def accuracy (self,y_true,y_pred):
    #     return np.sum(y_true == y_pred)/ len(y_true)
    

    def predict(self, X_test):
        import numpy as np
        from scipy import stats

        m = self.X_train.shape[0]
        n = X_test.shape[0]
        y_pred = []

        # Calculating distances  
        for i in range(n):  # for every sample in X_test
            distance = []  # To store the distances
            for j in range(m):  # for every sample in X_train
                if self.metric == 0:
                    d = (np.sqrt(np.sum(np.square(X_test.iloc[i,:] - self.X_train.iloc[j,:]))))  # Euclidean distance
                else:
                    d = (np.absolute(X_test.iloc[i, :] - self.X_train.iloc[j,:]))  # Manhattan distance
                distance.append((d, y_train[j]))    
            distance = sorted(distance) # sorting distances in ascending order

            # Getting k nearest neighbors
            neighbors = []
            for item in range(self.k):
                neighbors.append(distance[item][1])  # appending K nearest neighbors

            # Making predictions
            if self.problem == 0:
                y_pred.append(np.mean(neighbors))  # For Regression
            else:
                y_pred.append(stats.mode(neighbors)[0][0])  # For Classification
        return y_pred
##########################################################
cls = BruteForceKNN(k = 1)
trän=cls.fit(x_train,y_train)
preds= cls.predict(x_test)


