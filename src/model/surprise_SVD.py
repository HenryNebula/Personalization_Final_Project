import numpy as np
import pandas as pd
from surprise import *
from surprise import accuracy
from surprise import Dataset
from surprise import Reader

from src.model.BaseModel import BaseModel

class surprise_SVD(BaseModel):
    def __init__(self, params: dict):
        super().__init__(params)
        self.model = SVD(**self.params)
    
    def fit(self, train_df, user_info, item_info):
        
        ## transform to surprise ready
        reader = Reader(rating_scale = (1, 5))
        data_s = Dataset.load_from_df(train_df[['user_id', 'business_id', 'stars']], reader)
        trainset = data_s.build_full_trainset()
        
        ## train model
        self.model.fit(trainset)
        
    def transform(self, ui_pairs):
        
        ## get predicted score
        predicted = []
        for line in ui_pairs.itertuples():
            uid = line.user_id
            iid = line.business_id
            pred = self.model.predict(uid, iid, verbose=False)
            predicted.append(pred.est)
        
        testset_copy = ui_pairs.copy()
        testset_copy['prediction'] = predicted
        
        return testset_copy






