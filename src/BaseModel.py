from pyspark.sql import DataFrame


class BaseModel:
    def __init__(self):
        pass

    def fit(self, train_df: DataFrame):
        raise NotImplementedError

    def recommend_for_all_users(self, topN):
        raise NotImplementedError

    def recommend_for_pairs(self, ui_pairs: DataFrame):
        raise NotImplementedError