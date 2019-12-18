import pandas as pd
import scipy.sparse as sparse
import implicit
from src.model.BaseModel import BaseModel

class BPR(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        
        self.business_mapping = None
        self.user_mapping = None
        self.sparse_user_item = None
        self.sparse_item_user = None
        self.model = None

    def fit(self, train_df, user_info, item_info):
        """
        Args:
            train_df: a DF with 'business_id', 'user_id', 'cool', 'date', 'funny', 'review_id', 'stars', and
            'useful' as column names respectively.

        """
        ## convert to int IDs
        data = train_df.loc[:,['user_id','business_id','stars']]
        data['user_int_id'] = data['user_id'].astype("category")
        data['business_int_id'] = data['business_id'].astype("category")

        data['user_int_id'] = data['user_int_id'].cat.codes
        data['business_int_id'] = data['business_int_id'].cat.codes
        
        business_map = data[["business_id", "business_int_id"]]
        user_map = data[["user_id", "user_int_id"]]
        self.business_mapping = business_map.drop_duplicates(subset=["business_id", "business_int_id"])
        self.user_mapping = user_map.drop_duplicates(subset=["user_id", "user_int_id"])
        
        ## create rating matrix
        ## implicit package requires item_user matrix
        self.sparse_user_item = sparse.csr_matrix((data['stars'].astype(float), (data['user_int_id'], data['business_int_id'])))
        self.sparse_item_user = sparse.csr_matrix((data['stars'].astype(float), (data['business_int_id'], data['user_int_id'])))
        
        #Set BRP model
        self.model = implicit.bpr.BayesianPersonalizedRanking(factors=self.params["factors"], 
                                                             regularization=self.params["regularization"], 
                                                             iterations=self.params["iteration"], 
                                                             learning_rate=self.params["learning_rate"])

        self.model.fit(self.sparse_item_user, show_progress=False)
        
        

    def transform(self, ui_pairs):
        """
        Args:
            ui_pairs: a DF with 'business_id', 'user_id' 
        
        Return:
            a DF with 'user_id', 'business_id', 'prediction‘ of score

        """
        #Get item id
        #Convert string id int int id

        item_intids = ui_pairs.merge(self.user_mapping,on = 'user_id')
        item_intids = item_intids.merge(self.business_mapping, on ='business_id' )
        itemids_user = item_intids[['user_int_id','business_int_id']].groupby('user_int_id').agg({'business_int_id':lambda x:
                                                                                                  list(x)}).reset_index()
        
        #Get business and scores pair
        list_pair = []
        for row in itemids_user.itertuples(index=False):
            list_pair.append(self.model.rank_items(row.user_int_id, self.sparse_user_item,row.business_int_id))
        
        itemids_user['pair_score'] = list_pair
        itemids_user = itemids_user.drop(columns = ['business_int_id'])
        
        #explode dataframe
        exploded = itemids_user.pair_score.apply(pd.Series).stack().reset_index(level=1, drop=True).to_frame('business_score')
        itemids_user = itemids_user.join(exploded)
        itemids_user = itemids_user.reset_index(drop=True)
        itemids_user = itemids_user.drop(columns = 'pair_score')
        
        #split nested info into two columns
        itemids_user['business_int_id'] = itemids_user['business_score'].apply(lambda x: x[0])
        itemids_user['prediction'] = itemids_user['business_score'].apply(lambda x: x[1])
        
        itemids_user = itemids_user.drop(columns = ['business_score'])
        
        #Convert int id into string id
        itemids_user = itemids_user.merge(self.user_mapping, on = 'user_int_id')
        itemids_user = itemids_user.merge(self.business_mapping, on ='business_int_id')
        
        user_business_score = itemids_user.drop(columns = ['user_int_id','business_int_id'])
        
        return user_business_score