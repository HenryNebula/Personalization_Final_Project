from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import rand, approx_count_distinct, col, lit
from numpy.random import randint, seed as set_seed


class BaseModel:
    def __init__(self, params: dict):
        self.params = params

    def fit(self, train_df: DataFrame):
        raise NotImplementedError

    def recommend_for_all_users(self, topN):
        raise NotImplementedError

    def transform(self, ui_pairs: DataFrame):
        raise NotImplementedError

    def sort_params_to_list(self):
        return sorted(self.params.items(), key=lambda x: x[0])

    def get_name(self):
        return self.__class__.__name__

    def negative_sampling(self,
                          train_df: DataFrame,
                          seed=42,
                          num_neg=3):
        # apply negative sampling to training data
        if num_neg == 0:
            return train_df
        set_seed(seed)
        base = train_df.drop("ts").cache()
        item_num = train_df.agg(approx_count_distinct(col("item")).alias("count")).collect()[0]["count"]

        augment_train = base
        for i in range(num_neg):
            augment_train = (base
                             .withColumn("neg_item", (col("item") + randint(low=1, high=item_num)) % item_num)
                             .withColumn("neg_rating", lit(0))
                             .selectExpr("user", "neg_item as item", "neg_rating as rating")
                             .union(augment_train)
                             .groupBy(["user", "item"])
                             .agg(F.max("rating").alias("rating"))
                             )
        base.unpersist()
        return augment_train

    def binarize(self, train_df):
        # binarize the rating to binary records
        return (train_df
                .withColumn("bin_rating", lit(1))
                .selectExpr("user", "item", "bin_rating as rating"))
