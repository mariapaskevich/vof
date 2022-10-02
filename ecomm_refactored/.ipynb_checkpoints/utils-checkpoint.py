"""
You can add any arguments you need in the functions below
apart from get_dataset
"""
import yaml
import logging
import numpy as np
import pandas as pd
import altair as alt
import dill as pickle
import os
from datetime import datetime
import math


def read_config(path):
    with open(path, 'r') as stream:
        conf = yaml.safe_load(stream)
    logging.info(f'Config params: {conf}')
    return conf


def archive_experiment(model, metrics, config):
    
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')
        
    folder_id = config['model']['class']+str(datetime.now())
    path = f'./outputs/{folder_id}'

    if not os.path.exists(path):
        os.makedirs(path)
        
    metrics.to_csv(os.path.join(path,'evaluation_metrics.csv'))
    
    with open(os.path.join(path,'model_pkl'), 'wb') as pkl_file:
        pickle.dump(model, pkl_file)
    
    with open(os.path.join(path,'config.yaml'), 'w') as yaml_file:
        yaml.dump(config, yaml_file)
        
    return



def get_dataset(startdt = '2015-01-01', finishdt = '2016-01-01'):
    
    data = pd.read_csv('data/ecomm/sales_train_data_merged_CA_3_2014.csv',index_col='date').fillna(0)[:-1]
    data.index = data.index.astype('datetime64[ns]')
    data = data.loc[startdt:finishdt]

    data.reset_index(inplace=True)
    
    #modify wday into sin and cos to get rid if hierarchy in the cat numbers

    data['sin_wday'] = data.wday.apply(math.sin)
    data['cos_wday'] = data.wday.apply(math.cos)

    data = data.join(pd.get_dummies(data.event_name_1, prefix='event_name_1'))
    data = data.join(pd.get_dummies(data.event_name_2, prefix='event_name_2'))
    data = data.join(pd.get_dummies(data.event_type_1, prefix='event_type_1'))
    data = data.join(pd.get_dummies(data.event_type_2, prefix='event_type_2'))

    # getting rid of irrelevant data

    data = data.drop(['d','store_id','weekday','wday',
                              'event_name_1', 'event_name_2',
                              'event_type_1','event_type_2',
                              'month','year','snap_TX','snap_WI'], axis=1)

    return data

def get_top_items(data, top_items = 5):
    top_items = data.groupby('item_id').sum().sort_values(by='sales',ascending=False).index[:top_items]
    return top_items

def plot_items(data):
    alt.data_transformers.disable_max_rows()

    chart = alt.Chart(data.loc[data.item_id.isin(top_items)]).mark_line().encode(
                            x='date:T',
                            y=alt.Y('sales:Q'),
                            color='item_id',
                            tooltip=['item_id','date:T','sales:Q']
                        ).properties(width=500, height=400)
    return chart


def get_input_item(data, item_id='HOUSEHOLD_1_110'):
    item = data.loc[data.item_id.isin([item_id])]
    item.loc[item.sales<=0,'sales'] = item.sales.median()
    return item

