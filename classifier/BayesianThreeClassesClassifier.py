import numpy as np
from itertools import combinations
from sklearn.model_selection import KFold
import random

from classifier.BaseClassifier import BaseClassifier

class BayesianThreeClassesClassifier(BaseClassifier):
    def __init__(self, data, target):
        super().__init__(data,target)

    def __str__(self):
        return f"BayesianThreeClassesClassifier(numClasses = {len(np.unique(self.target))})"
    
    def mean(self, veclist):
        return sum(veclist) / len(veclist)
    
    def softmax(self, pred_list):
        explist = np.array([np.exp(pred) for pred in pred_list])
        return explist / sum(explist)


    def withinClassCovMat(self, train_data, train_target):
        res = np.zeros((len(train_data[0]),len(train_data[0])))
        for classId in range(3):
            idx = np.where(train_target == classId)
            tempX = train_data[idx]
            # tempT = train_target[idx]
            m = self.mean(tempX)
            class_res = np.zeros((len(train_data[0]),len(train_data[0])))
            for i in range(len(tempX)):
                class_res += np.dot((tempX[i] - m),np.transpose(tempX[i] - m))
            res += class_res
        return res
    
    def fit(self,train_data,train_target):
        covMat = self.withinClassCovMat(train_data,train_target) / len(train_data)
        for i in range(3):
            idx = np.where(train_target == i)
            mean = self.mean(train_data[idx])
            self._w[i] = np.dot(np.linalg.pinv(covMat), mean)
            temp = np.dot(np.dot(np.transpose(mean),np.linalg.pinv(covMat)),mean)
            self._w[i] = np.append(self._w[i],-0.5*temp + np.log(len(train_data[idx]) / len(train_data)))
        return self._w
    
    def predict(self, test, pair_id):
        res = []
        for i in range(3):
            res.append(np.dot(np.transpose(self._w[i]),np.array(test)).tolist())
        expres = self.softmax(np.array(res)).tolist()
        return expres.index(max(expres))
        

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
                pred = self.predict(np.append(test,1), None)
                # print("Rank:",class_rank)
                # print(test_target)
                # print("Test:",pred,i,test_target[i])
                
                if pred == test_target[i]:
                    acc += 1
                
            acc_by_folds.append(acc / len(test_data))

        return acc_by_folds