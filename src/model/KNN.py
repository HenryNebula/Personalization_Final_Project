from pyspark.sql.functions import *
from pyspark.sql.window import *
from pyspark.sql import DataFrame
from src.model.BaseModel import BaseModel


def item_sim(train_df):
    """Calculate the similarity scores between every item pairs in dataframe

    Args:
        train_df: a DF with 'user', 'item', and 'rating' as column names respectively

    Returns:
        a DF with 'item_i', 'item_j' and their 'sim' score as last column
    """

    # rename
    df1 = train_df.alias("df1")
    df2 = train_df.alias('df2')

    # get item norms
    norm = df1 \
        .withColumn('i_squared', col("rating") * col("rating")) \
        .groupby(['item']) \
        .agg(sum('i_squared').alias('sum_i_squared')) \
        .withColumn('norm', sqrt('sum_i_squared')) \
        .selectExpr('item', 'norm')

    norm.cache()
    # norm.count()  # make sure it's cached
    # norm.printSchema()

    # join movielens to itself on user
    item_item = (df1
                 # self join
                 .join(df2, df1['user'] == df2['user'])
                 .select('df1.user', 'df1.item', 'df2.item', 'df1.rating', 'df2.rating')

                 # inner product
                 .withColumn('product', col('df1.rating') * col('df2.rating'))
                 .groupby(['df1.item', 'df2.item'])
                 .agg(sum('product').alias('sum_product'))
                 .select(col("df1.item").alias("item_i"), col("df2.item").alias("item_j"), 'sum_product')

                 # get norm for both items
                 .join(norm, col("item_i") == col("item"), "inner")
                 .drop("item")
                 .withColumnRenamed("norm", "norm_i")
                 .join(norm, col("item_j") == col("item"))
                 .drop("item")
                 .withColumnRenamed("norm", "norm_j")

                 # cosine similarity
                 .withColumn('sim', col("sum_product") / col("norm_i") / col("norm_j"))
                 .selectExpr('item_i', 'item_j', 'sim')
                 )
    item_item.cache()
    # item_item.count()
    return item_item


def inf_score(train_df, sim, k):
    """Rank all neighbors of items based on sim score for all items

    Args:
        train_df: a DF with 'user', 'item', and 'rating' as column names respectively
        sim: a DF with similarity score for item pairs
        k: number of neighbors used
    Returns:
        a DF with predictions for all user item pairs seen in the training set
        with format [user, item, score]
    """

    # join item on neighbors
    score = (train_df.
             # join item on neighbors
             join(sim, col("item") == col("item_i"), 'inner')
             .selectExpr("user", "item", "rating", "item_j", "sim")
             # rename relevant items
             .withColumnRenamed("item", "item_rel")
             # rename items to be inferred
             .withColumnRenamed("item_j", "item_inf")

             # rank every neighbor for each item to be inferred
             .select('*', rank().over(
                Window.partitionBy('user', 'item_inf')
                      .orderBy(col("sim").desc())).alias('rank'))
             # filter neighbors up to top-k
             .filter((col('rank') <= k))

             .withColumn('sim_rating_product', col("rating") * col("sim"))

             # calculate score
             .groupby(['user', 'item_inf'])
             .agg(sum('sim_rating_product').alias('product_sum'),
                  sum('sim').alias('sim_sum'))

             .withColumn("score", col("product_sum") / col("sim_sum"))
             .selectExpr("user", "item_inf", "score")
             )

    score.cache()
    # score.count()

    return score


def get_rec(scores, topN):
    """Give recommendations to users based on inferred ratings

    Args:
        scores: a DF with format [user, item, score]
    Returns:
        a DF with recommended items for each user
    """

    # rank items by score per user
    rec = (scores
           .withColumn("rank",
                       rank().over(Window.partitionBy('user').orderBy(col("score").desc())))
           .filter(col('rank') <= topN)

           # struct for every entry in the final nested list
           .withColumn("item_score", struct(col("item_inf").alias("item"), col("score").alias("rating")))

           # collect list
           .groupby('user')
           .agg(collect_list("item_score").alias('recommendations'))
           )

    return rec


def compare(original, predict):
    """Compare rating and predicted score

    Args:
        original: a DF with [user, item, rating..]
        predict: a DF with [user, item, score]
    Returns:
        a joined DF
        with format [user, item, rating, score]
    """
    original_renamed = (original
                        .withColumnRenamed("user", "user_gt")
                        .withColumnRenamed("item", "item_gt")
                        )
    # join back to original
    compare_df = (predict
                  .join(original_renamed, (col("user") == col("user_gt")) & (col("item_inf") == col("item_gt")))
                  .selectExpr('user_gt as user', 'item_gt as item', 'rating', 'score as prediction'))
    return compare_df


class KNN(BaseModel):
    def __init__(self, params: dict):
        super().__init__(params)
        self.score = None

    def fit(self, train_df: DataFrame):

        train = train_df.selectExpr("user", "item", "rating")

        sim = item_sim(train)
        print("finish training similarity matrix {}".format(sim.count()))
        self.score = inf_score(train, sim, **self.params)

    def recommend_for_all_users(self, topN):
        if not self.score:
            raise ValueError("do fit() before making any inferences")

        else:
            recommendations = get_rec(self.score, topN)
            return recommendations

    def transform(self, ui_pairs: DataFrame):
        pair_recommendations = compare(ui_pairs, self.score)
        return pair_recommendations