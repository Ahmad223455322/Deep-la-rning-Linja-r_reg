import numpy as np
from collections import Counter 

def ED(x_1, x_2):  
    return np.sqrt(np.sum((int(x_1 ) - (int(x_2)) **2)))   


class KNN():
    def __init__(self,k) -> None:
        self.k = k
        
    def fit(self,X,y):
        self.X_träin = X 
        self.y_träin = y


    def predict(self,X):
        y_pred = [self.predict_most_common(x) for x in X ]
        return np.array(y_pred)
    
    

    def accuracy (y_true,y_pred):
        return np.sum(y_true == y_pred)/ len(y_true)


    def predict_most_common(self,x):
        
        avstånd = [ED(x, x_tr) for x_tr in self.X_träin]
        k_index = np.argsort(avstånd)[:self.k]

        k_neigbers_labels = [self.y_träin[c] for c in k_index]

        most_commo = Counter(k_neigbers_labels).most_common(1)

        return most_commo[0][0]       