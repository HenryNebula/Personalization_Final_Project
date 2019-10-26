from pyspark.sql import DataFrame
from pyspark.ml.recommendation import ALS
from src.BaseModel import BaseModel


class ALS_MF(BaseModel):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self.als = ALS(**self.params)
        self.model = None

    def _check_model(self):
        if not self.model:
            raise ValueError("run fit() before making any inferences")

    def fit(self, train_df: DataFrame):
        self.model = self.als.fit(train_df)

    def recommend_for_all_users(self, top_n):
        self._check_model()
        recommendations = self.model.recommendForAllUsers(top_n)
        return recommendations

    def recommend_for_pairs(self, ui_pairs: DataFrame):
        self._check_model()
        ui_recommendations = self.model.transform(ui_pairs)
        return ui_recommendations
