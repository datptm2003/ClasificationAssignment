import numpy as np
from itertools import combinations
from sklearn.model_selection import KFold
import random

from classifier.BaseClassifier import BaseClassifier


class ThreeClassesClassifier(BaseClassifier):
    def __init__(self, data, target):
        super().__init__(data,target)
    
    def __str__(self):
        return f"ThreeClassesClassifier(numClasses = {len(np.unique(self.target))})"
    
    def fit(self, train_data, train_target):
        
        idx = np.where((train_target >= 0))
        # print(idx)
        tempX = train_data[idx]
        # tempT = train_target[idx]
        tempT = np.zeros((len(train_target[idx]), 3))
        tempT[np.arange(len(train_target[idx])), train_target[idx]] = 1
        
        ones_column = np.ones((tempX.shape[0], 1))
        tempX = np.hstack((tempX, ones_column))
        # print("===",tempT)
        # tempT[tempT == first] = -1
        # tempT[tempT != -1] = 1
        
        self._w = np.dot(np.linalg.pinv(tempX),tempT)
        # print("***",self._w)
        return self._w

    def predict(self, test, pair_id):
        res = np.dot(np.transpose(self._w),np.array(test)).tolist()
        # print(res)
        return res.index(max(res))
    
    def test(self, k):
        kf = KFold(n_splits=k, shuffle=True, random_state=random.randint(1, 100))
        acc_by_folds = []
        
        for train_index, test_index in kf.split(self.data):
            train_data, test_data = self.data[train_index], self.data[test_index]
            train_target, test_target = self.target[train_index], self.target[test_index]
            
            # class_pairs = list(combinations(np.unique(test_target), 2))
            self.fit(train_data,train_target)
            # print("W:",self._w)
            acc = 0
            # print(test_target)
            for i, test in enumerate(test_data):
                # class_rank = [0,0,0]
                
                # idx = np.where((test_target == first) | (test_target != first))
                # tempT = test_target[idx]
                # tempCond = (tempT != first)
                # tempT[tempT == first] = 0
                # tempT[tempCond] = 1
                # print("TempT",test_target)
                    
                # class_rank[self.predict(np.append(test,1), None)] += 1
                # print(class_rank)
                pred = self.predict(np.append(test,1), None)
                # print("Rank:",class_rank)
                # print(test_target)
                # print("Test:",pred,i,test_target[i])
                
                if pred == test_target[i]:
                    acc += 1
                
            acc_by_folds.append(acc / len(test_data))

        return acc_by_folds
    