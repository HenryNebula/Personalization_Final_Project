from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.window import *
from src.model.BaseModel import BaseModel


def get_item_stats(train_df):
    """Compute baseline scores
    
    Args:
        train_df: a DF with [user, item, rating..]
    Returns:
        a DF with format ["user", "item","avg_rating","count_rank_rating"]
    """
    # prepare baseline df
    item_stats = (train_df
                  .groupBy(train_df.item)
                  .agg(count('rating').alias('count'),
                       mean('rating').alias('avg_rating'))
                  )

    item_stats.cache()

    max_count = int(item_stats.agg(max("count").alias("max_count")).collect()[0]["max_count"])
    min_count = int(item_stats.agg(min("count").alias("min_count")).collect()[0]["min_count"])

    max_rating = item_stats.agg(max("avg_rating").alias("max_rating")).collect()[0]["max_rating"]
    min_rating = item_stats.agg(max("avg_rating").alias("min_rating")).collect()[0]["min_rating"]

    # different mapping methods from ranking to scores:
    # avg_rating: score is avg_rating for each item over population
    # count_rank_rating: most popular get 5 score and least popular get 0

    return (item_stats
            .withColumn("count_rank_rating", (col("count") - min_count) / (max_count - min_count)
                        * (max_rating - min_rating) + min_rating)
            .select("item", "avg_rating", "count_rank_rating"))


def get_BL_score(item_stats, model):
    """get baseline scores for selected model
    
    Args:
        item_stats: a DF with ["item","avg_rating","count_uni_rating","count_rank_rating","count"]
        model: a string one of {"avg_rating", "count_uni_rating", "count_rank_rating"}
    Returns:
        a DF with format [item, score]
    """
    if model == "avg_rating":
        return item_stats.selectExpr("item", "avg_rating as score")
    elif model == "count_rank_rating":
        return item_stats.selectExpr("item", "count_rank_rating as score")
    else:
        raise ValueError("Invalid baseline method {}".format(model))


def get_rec(score, users, topN):
    """get recommendation for all users. same recommendation for all
    
    Args:
        score: a DF with format [item, score]
        users: user distinct column
        topN: a numeric number for top N to recommend
    Returns:
        a DF with format [user, recommendation = [[item, score]]]
    """
    rec = (score
           .withColumn("rank",
                       rank().over(Window.orderBy(col('score').desc())))
           .filter(col('rank') <= topN)
           .withColumnRenamed("score", "rating")
           .selectExpr("*", "(item, rating) as recommendation")
           .agg(collect_list(struct('rank', 'recommendation')).alias('a'))
           .select(sort_array('a')['recommendation'].alias('recommendations')))

    return users.crossJoin(rec)


def get_transform(ui_pairs, score):
    """Compare rating and predicted score
    
    Args:
        original: a DF with [user, item, rating..]
        score: a DF with [item, score]
    Returns:
        a DF with format [user, item, rating, prediction]
    """

    trans = ui_pairs.select('user', 'item', 'rating') \
        .join(score.selectExpr('item', 'score as prediction'), ["item"], how="left")

    return trans


class BaseLine(BaseModel):
    def __init__(self, params: dict):
        super().__init__(params)
        # choose which model to use, {"model": "avg_rating", "count_uni_rating", "count_rank_rating"}
        self.score = None
        self.item_stats = None
        self.users = None

    def _check_model(self):
        if not self.score or not self.item_stats or not self.users:
            raise ValueError("run fit() before making any inferences")

    def fit(self, train_df: DataFrame):
        self.item_stats = get_item_stats(train_df)
        self.score = get_BL_score(self.item_stats, **self.params)
        self.users = train_df.select("user").distinct()

    def recommend_for_all_users(self, top_n):
        self._check_model()
        recommendations = get_rec(self.score, self.users, top_n)
        return recommendations

    def transform(self, ui_pairs: DataFrame):
        self._check_model()
        pair_recommendations = get_transform(ui_pairs, self.score)
        return pair_recommendations
