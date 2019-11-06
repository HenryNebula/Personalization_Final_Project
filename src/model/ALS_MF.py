from pyspark.sql import DataFrame
from pyspark.ml.recommendation import ALS
from src.model.BaseModel import BaseModel


class ALS_MF(BaseModel):
    def __init__(self, params: dict):
        super().__init__(params)
        self.als = ALS(maxIter=self.params["maxIter"] if "maxIter" in self.params else 15,
                       rank=self.params["rank"] if "rank" in self.params else 10,
                       regParam=self.params["regParam"] if "regParam" in self.params else 0.1,
                       implicitPrefs=False,
                       coldStartStrategy="drop",
                       alpha=0,
                       nonnegative=False)

        self.model = None

    def _check_model(self):
        if not self.model:
            raise ValueError("run fit() before making any inferences")

    def fit(self, train_df: DataFrame):
        num_neg = 0 if "num_neg" not in self.params else self.params["num_neg"]
        self.model = self.als.fit(self.negative_sampling(train_df, num_neg))

    def recommend_for_all_users(self, top_n):
        self._check_model()
        recommendations = self.model.recommendForAllUsers(top_n)
        return recommendations

    def transform(self, ui_pairs: DataFrame):
        self._check_model()
        pair_recommendations = self.model.transform(ui_pairs)
        return pair_recommendations