from pyspark.sql import DataFrame


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