from sklearn import datasets
from classifier.TwoClassesOneOneClassifier import TwoClassesOneOneClassifier
from classifier.TwoClassesOneRestClassifier import TwoClassesOneRestClassifier
from classifier.ThreeClassesClassifier import ThreeClassesClassifier
from classifier.FisherThreeClassesClassifier import FisherThreeClassesClassifier
from classifier.BayesianThreeClassesClassifier import BayesianThreeClassesClassifier

iris = datasets.load_iris()
data = iris.data
target = iris.target

### Comment the models which you do not need to view the results ###

clf = TwoClassesOneOneClassifier(data,target)
# clf = TwoClassesOneRestClassifier(data,target)
# clf = ThreeClassesClassifier(data,target)
# clf = FisherThreeClassesClassifier(data,target)
# clf = BayesianThreeClassesClassifier(data,target)


print(clf.test(5))
