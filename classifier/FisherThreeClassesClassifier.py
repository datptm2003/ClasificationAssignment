import numpy as np
from itertools import combinations
from sklearn.model_selection import KFold
import random

from classifier.BaseClassifier import BaseClassifier
from classifier.ThreeClassesClassifier import ThreeClassesClassifier


class FisherThreeClassesClassifier(BaseClassifier):
    def __init__(self, data, target):
        super().__init__(data,target)
        self.proj_w = None
    
    def __str__(self):
        return f"FisherThreeClassesClassifier(numClasses = {len(np.unique(self.target))})"
    
    def mean(self, veclist):
        return sum(veclist) / len(veclist)
    
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
    
    def totalCovMat(self, train_data):
        m = self.mean(train_data)
        res = np.zeros((len(train_data[0]),len(train_data[0])))
        for i in range(len(train_data)):
            res += np.dot((train_data[i] - m),np.transpose(train_data[i] - m))
        return res
    
    def fit(self, train_data, train_target):
        sW = self.withinClassCovMat(train_data, train_target)
        sB = self.totalCovMat(train_data) - sW
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(sW).dot(sB))

        # Sort eigenvalues and corresponding eigenvectors in descending order
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

        # print("===",eig_pairs)
        # Select the top k eigenvectors (k = n_classes - 1)
        self.proj_w = np.hstack([eig_pairs[i][1].reshape(4, 1) for i in range(2)]).real
        new_train_data = np.dot(train_data,self.proj_w)

        threeClassModel = ThreeClassesClassifier(new_train_data,train_target)
        # print("New train data:",len(new_train_data))
        self._w = threeClassModel.fit(new_train_data,train_target)
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
            
            self.fit(train_data,train_target)
            # print("W:",self._w)
            acc = 0
            # print(test_target)
            for i, test in enumerate(test_data):
                pred = self.predict(np.append(np.dot(test,self.proj_w),1), None)
                # print("W:")
                # print("Test:",pred,test_target[i])
                if pred == test_target[i]:
                    acc += 1
            acc_by_folds.append(acc / len(test_data))
        return acc_by_folds
    