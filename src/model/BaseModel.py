from pandas import DataFrame
from pickle import dump, load
from copy import deepcopy


class BaseModel:
    def __init__(self, params: dict):
        self.params = deepcopy(params)
        self.model = None
        self.model_name = self.__class__.__name__

    def fit(self, train_df: DataFrame, user_info: DataFrame, item_info: DataFrame):
        raise NotImplementedError

    def transform(self, ui_pairs: DataFrame) -> DataFrame:
        # input as (user_id, business_id, (stars))
        # return as (user_id, business_id, (stars), prediction)

        raise NotImplementedError

    def recommend_on_candidates(self, candidates: DataFrame, top_n):
        # input candidates as (user_id, candidate_id)
        # return as (user_id, recommendations), where recommendations is an ordered array of business_id

        candidates = candidates.rename(columns={"candidate_id": "business_id"})
        predictions = self.transform(candidates)
        predictions["rank"] = predictions.groupby("user_id")["prediction"].rank(method="min", ascending=False)
        predictions = predictions[predictions["rank"] <= top_n]
        recommendations = (predictions
                           .sort_values(by="rank", ascending=True)
                           .groupby("user_id")["business_id"]
                           .agg(list).reset_index())
        recommendations.rename(columns={"business_id": "recommendations"})
        return recommendations

    def sort_params_to_list(self):
        return sorted(self.params.items(), key=lambda x: x[0])

    def get_name(self):
        return self.model_name

    def save_model(self, path):
        if not self.model:
            print("Model object is empty for class {}, so model saving is not executed".format(self.get_name()))
        else:
            dump(self.model, open(path, "wb"))

    def load_model(self, path):
        model = load(open(path, "rb"))
        assert model, "Model is empty, check the pickle file at {}".format(path)
        self.model = model
