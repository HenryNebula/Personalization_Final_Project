from pyspark.sql import DataFrame
from pyspark.ml.recommendation import ALS
from src.model.BaseModel import BaseModel


class ALS_MF(BaseModel):
    def __init__(self, params: dict):
        super().__init__(params)
        self.als = ALS(implicitPrefs=False,
                       coldStartStrategy="drop",
                       alpha=0,
                       nonnegative=False,
                       **self.params)
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

    def transform(self, ui_pairs: DataFrame):
        self._check_model()
        pair_recommendations = self.model.transform(ui_pairs)
        return pair_recommendations