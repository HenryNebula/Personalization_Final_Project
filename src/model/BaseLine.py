from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.window import *
from src.model.BaseModel import BaseModel

def get_BL_full(train_df):

    """Compute baseline scores
    
    Args:
        train_df: a DF with [user, item, rating..]
    Returns:
        a DF with format ["item","avg_rating","count_uni_rating","count_rank_rating","count"]
    """ 
    #prepare baseline df
    df_BL = train_df.groupBy(train_df.item)\
        .agg(count('rating').alias('count'),mean('rating').alias('avg_rating'))

    df_BL =\
        df_BL.select("*",
         rank().over(Window.orderBy(df_BL['count'].desc())).alias("rank_count"),
         rank().over(Window.orderBy(df_BL['avg_rating'].desc())).alias("rank_avg_rating"))

    ##set uniform parameters
    range_c = df_BL.agg(min('count').alias('min_c'),max('count').alias('max_c'))\
        .selectExpr("*","max_c-min_c as diff_c")
    range_r = train_df.agg(min('rating').alias('min_r'),max('rating').alias('max_r'))\
        .selectExpr("*","max_r-min_r as diff_r")

    ##different recommendations: 
    ###avg_rating: recommend based on ave_rating for each item over population
    ###count_uni_rating: recommend based on popularity, most frequent item rate 5 and least rate 1 based on count
    ###count_rank_rating: recommend based on rank of popularity, most frequent item rate 5 and least rate 1 based on rank of count
    df_BL = df_BL.crossJoin(range_c.crossJoin(range_r)).withColumn('n_item',lit(df_BL.count()))
    df_BL = df_BL.selectExpr("*", 
                             "(count-min_c) * diff_r/diff_c+min_r as count_uni_rating",
                             "(n_item - rank_count) * diff_r/(n_item-1)+min_r as count_rank_rating")

    df_BL = df_BL.select("item","avg_rating","count_uni_rating","count_rank_rating","count")

    return df_BL

def get_BL_score(df_BL, model):
    """get baseline scores for selected model
    
    Args:
        df_BL: a DF with ["item","avg_rating","count_uni_rating","count_rank_rating","count"]
        model: a string one of {"avg_rating", "count_uni_rating", "count_rank_rating"}
    Returns:
        a DF with format [item, score]
    """ 
    if model == "avg_rating":
        score = df_BL.selectExpr("item","avg_rating as score")
    elif model == "count_uni_rating":
        score = df_BL.selectExpr("item","count_uni_rating as score")
    elif model == "count_rank_rating":
        score = df_BL.selectExpr("item","count_rank_rating as score")
    else:
        score = None
    
    return score
    
def get_rec(score, userls, topN):
    """get recommendation for all users. same recommendation for all
    
    Args:
        score: a DF with format [item, score]
        userls: a DF with format [user]
        topN: a numeric number for top N to recommend
    Returns:
        a DF with format [user, recommendation = [[item, score]]]
    """ 
    rec = score.select("*",
             rank().over(Window.orderBy(score['score'].desc())).alias("rank"))\
        .filter(col('rank') <= topN)\
        .selectExpr("*","(item, score) as recommendation")\
        .agg(collect_list(struct('rank','recommendation')).alias('a'))\
        .select(sort_array('a')['recommendation'].alias('recommendations'))
      
    return userls.crossJoin(rec)

def get_transform(ui_pairs, score):
    """Compare rating and predicted score
    
    Args:
        original: a DF with [user, item, rating..]
        score: a DF with [item, score]
    Returns:
        a DF with format [user, item, rating, prediction]
    """

    trans = ui_pairs.select('user','item','rating')\
        .join(score.selectExpr('item','score as prediction'), ["item"], how="left")
    
    return trans

class BaseLine(BaseModel):
    def __init__(self, params: dict):
        super().__init__(params) 
        #choose which model to use, {"model": "avg_rating", "count_uni_rating", "count_rank_rating"}
        self.score = None
        self.BL =None
        self.userls = None

    def _check_model(self):
        if not self.score:
            raise ValueError("run fit() before making any inferences")

    def fit(self, train_df: DataFrame):
        self.BL = get_BL_full (train_df)
        self.score = get_BL_score(self.BL, **self.params)
        self.userls = train_df.select("user").distinct()

    def recommend_for_all_users(self, top_n):
        self._check_model()
        recommendations = get_rec(self.score, self.userls, top_n)
        return recommendations

    def transform(self, ui_pairs: DataFrame):
        self._check_model()
        pair_recommendations = get_transform(ui_pairs, self.score)
        return pair_recommendations