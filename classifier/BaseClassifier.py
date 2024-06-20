import numpy as np

class BaseClassifier:
    def __init__(self, data: np.ndarray, target: np.ndarray):
        self.data = data
        self.target = target

        self._w = {}

    def __str__(self):
        pass
    
    def __call__(self, data, target):
        return self.__init__(data, target)

    def fit(self) -> list:
        pass

    def predict(self, test: list, pair_id) -> float:
        pass

    def test(self, k) -> list:
        pass
