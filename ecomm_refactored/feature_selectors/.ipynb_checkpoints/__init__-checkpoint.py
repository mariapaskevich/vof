from abc import ABCMeta, abstractmethod


class AbstractFeatureSelectors(metaclass=ABCMeta):

    def __init__(self):
        pass
    
    @abstractmethod
    def fit(self, features, labels, **kwargs):
        pass

    @abstractmethod
    def fit_transform(self, features, labels, **kwargs):
        pass
    
    @abstractmethod
    def transform(self, features, **kwargs):
        pass