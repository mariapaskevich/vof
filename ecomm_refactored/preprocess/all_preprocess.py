from mltask.preprocess import AbstractMLPreprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


class FeaturesScaler(AbstractMLPreprocessing):
    #Features scaler. Works on the entire array of features provided
    
    def __init__(self):
        super(FeaturesScaler, self).__init__()
        self.scaler = StandardScaler()        
        
    def fit(self, features, **kwargs):
        self.scaler.fit(features)
        return
    
    def fit_transform(self, features, **kwargs):
        return self.scaler.fit_transform(features)    
    
    def transform(self, features, **kwargs):
        return self.scaler.transform(features)
    
    
class FillMissingVals(AbstractMLPreprocessing):
    #Fill missing values with mean for the feature
    
    def __init__(self):
        super(FillMissingVals, self).__init__()
        self.data = None
        
    def fit(self, features, **kwargs):
        self.data_to_fit = pd.DataFrame(features)
        return       

    def transform(self, features, **kwargs):
        self.data_to_transform = pd.DataFrame(features)
        
        for col in self.data_to_transform.columns:
            self.data_to_transform[col] = self.data_to_transform[col].fillna(self.data_to_fit[col].mean())
            
        return self.data_to_transform.values
    
    def fit_transform(self, features, **kwargs):
        self.data_to_fit = pd.DataFrame(features)
        self.data_to_transform = pd.DataFrame(features)
        
        for col in self.data_to_transform.columns:
            self.data_to_transform[col] = self.data_to_transform[col].fillna(self.data_to_fit[col].mean())
            
        return self.data_to_transform.values

        
class TrainTestSplit():
    #Split data into training and test set
    
    def __init__(self):
        pass
        
    @staticmethod 
    def train_test_splitter(features, labels, size, seed, **kwargs):
        return train_test_split(features, labels, test_size=size, random_state=seed, **kwargs)
