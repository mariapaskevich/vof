import sys
from ecomm_refactored.utils import *
#from mltask.preprocess.all_preprocess import *
#from mltask.feature_selectors.all_feature_selectors import *
#from mltask.models.all_models import *
#from mltask.evaluators.all_evaluators import *

def main(config):
    # downloads the datasets
    data = get_dataset(startdt = '2015-01-01', finishdt = '2016-01-01')
    
    top_items = get_top_items(data, top_items = 5)
    
    
    
    #Split into test and train
    splitter = TrainTestSplit()
    X_train, X_test, y_train, y_test = splitter.train_test_splitter(features, labels, config['preprocessing']['test_split'], config['seed'])

    
    #Select relevant features

    
    #Standardize features by removing the mean and scaling to unit variance
    if config['preprocessing']['scale_features']:       
        
        
    #Fill in missing values using mean for the feature
    #if config['preprocessing']['fill_missing']:  
        
    #Train a model
    #print(config['model']['class'])
    

    #Cross validation prediction
    
    #Prediction on validation set
    
    #Evaluation metrics
    
    # final step should be archiving your experiment
    #archive_experiment(model, metrics, config)
    print('The output has been successfully saved')
    
if __name__ == '__main__':
    main(read_config(sys.argv[1]))