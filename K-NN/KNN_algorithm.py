#__author__=Luo Bowen

import numpy as np
import pandas as pd
import math
import operator

class KNN():
    def __init__(self, k):
        self.k = k
        
    def distance(self, x1, x2):
        return math.sqrt(sum((x1 - x2) ** 2))
    
    def knearestneighbor(self, X, T):
        row, col = np.shape(X)
        X1 = X[:,0:col-1]
        Y = X[:, col-1:]
        temp = []
        for i in range(row):
            temp.append(self.distance(X1[i], T))
        sort_temp = np.argsort(temp)  #argsort
        k_team = []
        for j in range(self.k):
            k_team.append(sort_temp[j])
        classdic = {}
        for k in k_team:
            classdic[Y[k][0]] = classdic.get(Y[k][0], 0) + 1
        sort_classdic = sorted(classdic.items(), key=operator.itemgetter(1), reverse=True)
        return sort_classdic[0][0]
    
if __name__ == "__main__":
    X = pd.read_csv('iris.data.txt',header = None)
    X = np.array(X)
    T = np.array([5.8, 2.7, 5.1, 1.9])
    c = KNN(5).knearestneighbor(X, T)
    print(c)
        
            
        