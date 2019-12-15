from fastFM import als
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd
from pandas import DataFrame
from src.model.BaseModel import BaseModel
from copy import deepcopy

def prepare_data(train_df,user_info,item_info,model):

    """Compute prepare data into 
    
    Args:
        train_df: a DF with [user_id, business_id, stars]
        user_info: a DF with [user_id, ..]
        item_info: a DF with [business_id, ..]
    Returns:
        X_train: a DF with format based on model
        y_train: a DF single column of stars
    """ 
    
    if(model== "None"):
        df = train_df
    elif (model == "User"):
        df = train_df.merge(user_info,how='left',on='user_id')
    elif (model == "Item"):
        df = train_df.merge(item_info,how='left',on='business_id')
    elif (model == "Both"):
        df = train_df.merge(user_info,how='left',on='user_id').\
        merge(item_info,how='left',on='business_id',suffixes=('_user','_business'))
      
    df = df.dropna()
    if('stars' in df.columns):
        X = df.drop('stars', axis=1)
        y = df['stars']
    else:
        X = df
        y = None
    
    return X, y


def get_transform(ui_pairs, X_test, pred):
    """Compare rating and predicted score
    
    Args:
        ui_pairs: a DF with [user, item, stars..]
        X_test: input data generates prediction, a DF with [user,item,stars...]
        pred: a array of predicted value
    Returns:
        a DF with format [user, item, stars, prediction]
    """

    df = deepcopy(X_test[['user_id','business_id']])
    df['prediction'] = pred
    #ui_pairs.merge(df, how = 'left', on = ['user_id','business_id'])
    
    return ui_pairs.merge(df, how = 'left', on = ['user_id','business_id'])

class FM(BaseModel):
    def __init__(self, params: dict):
        super().__init__(params) 
        #choose which model to use, {"model_info_used": "None", "User", "Item","Both"}
        self.model_info_used = self.params["model_info_used"] if "model_info_used" in self.params else "None"
        self.fm = als.FMRegression(n_iter = self.params["n_iter"] if "n_iter" in self.params else 100,
                                  l2_reg_w = self.params["l2_reg_w"] if "l2_reg_w" in self.params else 0.1,
                                  l2_reg_V = self.params["l2_reg_V"] if "l2_reg_V" in self.params else 0.5,
                                  rank = self.params["rank"] if "rank" in self.params else 2)
        self.model = None
        self.v = DictVectorizer()
        self.user_info = None
        self.item_info = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.pred = None
        self.model_name += self.model_info_used
 

    def _check_model(self):
        if not self.model:
            raise ValueError("run fit() before making any inferences")

    def fit(self, train_df: DataFrame, user_info, item_info):
        self.user_info = user_info
        self.item_info = item_info.rename(columns={"stars": "average_stars"})
        self.X_train, self.y_train = prepare_data(train_df,self.user_info,self.item_info,self.model_info_used)
        self.model = self.fm.fit(self.v.fit_transform(self.X_train.to_dict('records')), self.y_train)
        
    def transform(self, ui_pairs: DataFrame) -> DataFrame:
        # input as (user_id, business_id, stars)
        # output as (user_id, business_id, stars, prediction)
        self._check_model()
        self.X_test, self.y_test = prepare_data(ui_pairs,self.user_info,self.item_info,self.model_info_used)
        ##use v.transform rather than v.fit_transform
        self.pred = self.model.predict(self.v.transform(self.X_test.to_dict('records')))
        pair_recommendations = get_transform(ui_pairs,self.X_test,self.pred)
        return pair_recommendations
    
    def recommend_for_all_users(self, top_n):
        self._check_model()
        #recommendations = get_rec(self.score, self.userls, top_n)
        #return recommendations