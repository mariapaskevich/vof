import random
import pandas as pd 
import numpy as np
import nevergrad as ng



class OptimalDecisionMakers(): 
    
    def __init__(self, data, target):
        self.x = data#.drop([target], axis=1)
        self.y = data[target]
    
    def select_value_optimal_model(self,
                                   optimizer,
                                   cost_function,
                                   decision_time_step,
                                   optimization_horizon,
                                   time_unit = 'H',
                                   random_seed=42,
                                   return_predictions=False):
                
        random.seed(random_seed)
        self.cost_function = cost_function
        self.optimal_decisions_df = pd.DataFrame(index=(self.x.index),columns=self.x.columns).iloc[:-optimization_horizon]

        for col in range(len(self.optimal_decisions_df.columns)):

            total_steps = range(0,len(self.x)-optimization_horizon,decision_time_step)
            recommendation = np.array([])

            for step in total_steps:
                
                par = ng.p.Array(shape=(optimization_horizon,)).set_bounds(lower=-80, upper=80)
                optimizer = ng.optimizers.CMA(parametrization=par, budget=5000)
                
                try:
                    self.candidate = self.x.iloc[step:step+optimization_horizon,col]
                except:
                    print(self.optimal_decisions_df.shape)
                    print(self.x.iloc[step:step+optimization_horizon,col])
                    
                step_recommendation = optimizer.minimize(cost_function)
                recommendation = np.append(recommendation,step_recommendation.value[0:decision_time_step])
                

            self.optimal_decisions_df.iloc[:,col] = recommendation
            

        return self.optimal_decisions_df
    
    def evaluate(self):
        
        for col in range(len(self.optimal_decisions_df.columns)):
            
            print(self.cost_function(self.optimal_decisions_df.iloc[:,col]))
        
    def select_value_optimal_hyperparameters():
        return