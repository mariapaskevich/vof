from mltask.feature_selectors import AbstractFeatureSelectors
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


class FeatureSelectorPearson(AbstractFeatureSelectors):
    #Feature selection is performed using Pearson correlation coefficient. Returns indexes of features with corr >= min_corr_lvl
    
    def __init__(self, min_corr_lvl = .2):        
        super(FeatureSelectorPearson, self).__init__()
        self.min_corr_lvl = min_corr_lvl
              
    def fit(self, features, labels):        
        self.corr_w_labels = abs(np.corrcoef(features, labels.reshape(-1,1),rowvar=False)[-1,:-1])
        return 
      
    def fit_transform(self, features, labels, min_corr_lvl = .2):
        self.corr_w_labels = abs(np.corrcoef(features, labels.reshape(-1,1),rowvar=False)[-1,:-1])
        return features[:,np.argwhere(self.corr_w_labels >= self.min_corr_lvl).reshape(-1)]
        
    def transform(self, features):
        return features[:,np.argwhere(self.corr_w_labels >= self.min_corr_lvl).reshape(-1)]
    

class FeatureSelectorANOVA(AbstractFeatureSelectors):
    #Feature selection is performed using ANOVA F measure via the f_classif() function
    
    def __init__(self,k_best, **kwargs):
        super(FeatureSelectorANOVA, self).__init__()
        self.feature_selector = SelectKBest(score_func=f_classif, k=k_best, **kwargs)     
        
    def fit(self, features, labels):
        self.feature_selector.fit(features,labels)
        return
    
    def fit_transform(self, features, labels):
        return self.feature_selector.fit_transform(features,labels)
    
    def transform(self, features):
        return self.feature_selector.transform(features)
