import numpy as np
from itertools import combinations
from sklearn.model_selection import KFold
import random

from classifier.BaseClassifier import BaseClassifier


class TwoClassesOneOneClassifier(BaseClassifier):
    def __init__(self, data, target):
        super().__init__(data,target)
    
    def __str__(self):
        return f"TwoClassesOneOneClassifier(numClasses = {len(np.unique(self.target))})"

    def fit(self, train_data, train_target):
        class_pairs = list(combinations(np.unique(train_target), 2))
        
        for (first, second) in class_pairs:
            idx = np.where((train_target == first) | (train_target == second))
            # print("Original:",train_target)
            tempX = train_data[idx]
            # tempT = train_target[idx]
            tempT = np.zeros((len(train_target[idx]), 2))
            tempT[np.arange(len(train_target[idx])), [1 if t == second else 0 for t in train_target[idx]]] = 1
            # print(tempT)
            ones_column = np.ones((tempX.shape[0], 1))
            tempX = np.hstack((tempX, ones_column))

            self._w[(first, second)] = np.dot(np.linalg.pinv(tempX),tempT)
        return self._w

    def predict(self, test, pair_id):
        # print(pair_id)
        # print(np.dot(np.transpose(self._w[pair_id]),np.array(test)))
        res = np.dot(np.transpose(self._w[pair_id]),np.array(test)).tolist()
        # print(res)
        return res.index(max(res))
    
    def test(self, k):
        kf = KFold(n_splits=k, shuffle=True, random_state=random.randint(1, 100))
        acc_by_folds = []

        for train_index, test_index in kf.split(self.data):
            train_data, test_data = self.data[train_index], self.data[test_index]
            train_target, test_target = self.target[train_index], self.target[test_index]
            
            class_pairs = list(combinations(np.unique(test_target), 2))
            self.fit(train_data,train_target)
            # print("W:",self._w)
            acc = 0

            for i, test in enumerate(test_data):
                class_rank = [0,0,0]
                for (first, second) in class_pairs:
                    # idx = np.where((test_target == first) | (test_target == second))
                    # # tempX = test_data[idx]
                    # tempT = test_target[idx]
                    # tempT[tempT == first] = 0
                    # tempT[tempT == second] = 1
                        
                    if self.predict(np.append(test,1), (first, second)) == 0:
                        class_rank[first] += 1
                    else:
                        class_rank[second] += 1
                    
                pred = class_rank.index(max(class_rank))
                # print("Test:",pred,test_target[i])
                
                if pred == test_target[i]:
                    acc += 1
                
            acc_by_folds.append(acc / len(test_data))

        return acc_by_folds
