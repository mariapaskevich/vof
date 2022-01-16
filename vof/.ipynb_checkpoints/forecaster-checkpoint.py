import pandas as pd
import datetime
from pycaret.internal.pycaret_experiment import TimeSeriesExperiment

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
    
    def create_single_prediction(self, model_name,
                          start_day='2019-01-02', 
                          last_known_day='2019-12-01', 
                          prediction_day='2019-12-02',
                          frequency='D',
                          horizon=10):     
        
        print(model_name)
        
        x_modified = self.x.copy()
        x_modified.loc[prediction_day:] = x_modified.loc[prediction_day:] # check to avoid leakage
        
        X_train = x_modified.loc[start_day:last_known_day]
        y_train = self.y.loc[start_day:last_known_day]
        
        
        model_name.fit(X_train,y_train)
        
        try:
            self.model_coefs = model_name.coef_
        except:
            pass
        

        pred_index = pd.date_range(start=prediction_day, 
                                   periods=horizon, 
                                   freq=frequency)
        
        pred_df = pd.DataFrame(index=pred_index, 
                               columns=[self.y.name])
        
        #Create prediction for each time step in pred_index
        for step in pred_index:
            indx = str(step)#[:-9]
            
             
            if step != pred_index[0]: #starting from the second step, update x_modified with lags containing lags of predicted value 
                for i in self.lags:
                    inx_to_replace = pred_df.shift(i).dropna().index
                    x_modified.loc[inx_to_replace,'lag_'+str(i)] = pred_df.shift(i).loc[inx_to_replace].values

            X_pred = pd.DataFrame(x_modified.loc[indx]).T
            x_index = x_modified.loc[indx].name
            pred_df.loc[x_index] = model_name.predict(X_pred)            
            
            
        return pred_df
        
    
    def create_naive_prediction(self, model_name='daily naive',
                      start_day='2019-01-02', 
                      last_known_day='2019-12-01', 
                      prediction_day='2019-12-02',
                      terminal_day='2019-12-17',
                      frequency='D',
                      horizon=10):     

        #print('Im in create_naive_prediction!')
        pred_days = pd.date_range(start=prediction_day, end=terminal_day, freq='D')
        last_known_days = pd.date_range(start=last_known_day, periods=len(pred_days), freq='D')

        if frequency=='D':
            end_of_prediction = str(datetime.datetime.strptime(terminal_day, '%Y-%m-%d') + datetime.timedelta(days=horizon))

        elif frequency=='H':
            end_of_prediction = str(datetime.datetime.strptime(terminal_day, '%Y-%m-%d') + datetime.timedelta(hours=horizon))

                
        forecast_for_days = pd.DataFrame(index=pd.date_range(start=prediction_day, end=end_of_prediction, freq=frequency))
        
        for (pd_day,lk_day) in zip(pred_days,last_known_days):
            
            print(model_name)
            if frequency=='D':
                pred_values = horizon*list(self.y.loc[lk_day].values)

            elif frequency=='H':
                #lk_day_24h is a new index which include each hour in the last known day
                lk_day_24h = pd.date_range(start=lk_day, periods=24, freq='H')
                pred_values = int(horizon/24)*list(self.y.loc[lk_day_24h].values)

            else:
                print('exception!') 

            
            pred_index = pd.date_range(start=pd_day, 
                                   periods=horizon, 
                                   freq=frequency)
            
            pred_df = pd.DataFrame(index=pred_index, 
                               columns=[self.y.name],
                               data=pred_values)
            
            forecast_for_days[pd_day] = pred_df
            
        return forecast_for_days

     
    def create_recursive_prediction(self, model_name,
                                      start_day='2019-01-02', 
                                      last_known_day='2019-12-01', 
                                      prediction_day='2019-12-02',
                                      terminal_day='2019-12-17',
                                      freq='D',
                                      h=10):
        
        pred_days = pd.date_range(start=prediction_day, end=terminal_day, freq='D')

        start_days = pd.date_range(start=start_day, periods=len(pred_days), freq='D')

        last_known_days = pd.date_range(start=last_known_day, periods=len(pred_days), freq='D')
        
        if freq == 'D':
            end_of_prediction = str(datetime.datetime.strptime(terminal_day, '%Y-%m-%d') + datetime.timedelta(days=h))
        elif freq == 'H':
            end_of_prediction = str(datetime.datetime.strptime(terminal_day, '%Y-%m-%d') + datetime.timedelta(hours=h))
        
        forecast_for_days = pd.DataFrame(index=pd.date_range(start=prediction_day, end=end_of_prediction, freq=freq))

        for (pd_day,st_day,lk_day) in zip(pred_days,start_days,last_known_days):
            
            res = self.create_single_prediction(model_name,
                              start_day=st_day,
                              last_known_day=lk_day,
                              prediction_day=pd_day,
                              frequency=freq,
                              horizon=h)

            forecast_for_days[pd_day] = res

        return forecast_for_days
        
    def create_pycaret_prediction(self,start_day='2019-01-02', 
                                      last_known_day='2019-12-01', 
                                      prediction_day='2019-12-02',
                                      terminal_day='2019-12-17',
                                      freq='D',
                                      h=10):
    
        exp = TimeSeriesExperiment()
        exp.setup(data, fh = 7, fold = 3, session_id = 123)
        best = exp.compare_models()
        
        
    
        return