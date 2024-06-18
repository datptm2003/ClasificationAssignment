from sklearn import datasets
import Classifier as clf

DATA_PER_CLASS = 50

iris = datasets.load_iris()
data = iris.data
target = iris.target

base = clf.FisherThreeClassesClassifier(data,target)

print(base.test(5))
