import pandas as pd


class RecursiveForecaster(): 
    
    def __init__(self, data, target):
        self.x = data.drop(target, axis=1)
        self.y = data[target]

    def add_lag_features(self,lags):
        self.lags=lags
        for i in lags:
            self.x['lag_'+str(i)] = self.y.shift(i)    
            
    
    #def optimize(self, pred):
    #    self.cost_function
    
    #def fit(self):
        #we run our daily loops for valdation and call the optimizer
    
    def create_prediction(self, model_name,
                          start_day='2019-01-02', 
                          last_known_day='2019-12-01', 
                          prediction_day='2019-12-02',
                          frequency='D',
                          horizon=10):     
        
        print(model_name)
        
        X_train = self.x.loc[start_day:last_known_day]
        y_train = self.y.loc[start_day:last_known_day]
        
        
        model_name.fit(X_train,y_train)

        pred_index = pd.date_range(start=prediction_day, 
                                   periods=horizon, 
                                   freq=frequency)
        
        #print(pred_index)
        pred_df = pd.DataFrame(index=pred_index, 
                               columns=[self.y.name])
        
        #print(pred_df)
        #Create prediction for each time step in pred_index
        for step in pred_index:
            indx = str(step)#[:-9]
            #print(indx)
            X_pred = pd.DataFrame(self.x.loc[indx]).T
            #print(X_pred)
            x_index = self.x.loc[indx].name
            #print(pred_df.loc[x_index])
            pred_df.loc[x_index] = model_name.predict(X_pred)
            
            #update features with lags containing lags of predicted value 
            for i in self.lags:
                self.x.loc[indx,'lag_'+str(i)] = pred_df.shift(i)

        return pred_df