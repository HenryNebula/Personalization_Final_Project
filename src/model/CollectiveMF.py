import pandas as pd
from pandas import DataFrame
from cmfrec import CMF
from src.model.BaseModel import BaseModel
from copy import deepcopy


class CollectiveMF(BaseModel):
    def __init__(self, params: dict):
        super().__init__(params)
        w_main = self.params["w_main"] if "w_main" in self.params else 1.0
        self.use_user_info = self.params["use_user_info"]
        self.use_item_info = self.params["use_item_info"]
        del self.params["use_user_info"], self.params["use_item_info"]

        # sum of side info weight will be fixed at 50 percent of w_main
        if self.use_item_info and self.use_user_info:
            w_user = w_main * 0.5 * 0.6
            w_item = w_main * 0.5 * 0.4
        else:
            w_user = w_main * 0.5 * self.use_user_info
            w_item = w_main * 0.5 * self.use_item_info

        self.model = CMF(w_main=w_main,
                         w_user=w_user,
                         w_item=w_item,
                         k=self.params["k"] if "k" in self.params else 10,
                         reg_param=self.params["reg_param"] if "reg_param" in self.params else 1e-4,
                         random_seed=16)

        append_name = [["_No", "_Item"], ["_User", "_Both"]]
        self.model_name += append_name[int(self.use_user_info)][int(self.use_item_info)]

    def _check_model(self):
        if not self.model:
            raise ValueError("run fit() before making any inferences")

    def fit(self, train_df: DataFrame, user_info, item_info):
        cp_train_df = deepcopy(train_df)
        cp_train_df.rename(columns={"user_id": "UserId", "business_id": "ItemId", "stars": "Rating"},
                           errors="raise", inplace=True)

        cp_user_info = deepcopy(user_info).rename(columns={"user_id": "UserId"}) \
            if self.use_user_info else None

        cp_item_info = deepcopy(self.create_dummies(item_info)).rename(columns={"business_id": "ItemId"}) \
            if self.use_item_info else None

        cols_bin_item = [cl for cl in cp_item_info.columns if "bin" in cl] if self.use_item_info else None
        cols_bin_user = [cl for cl in cp_user_info.columns if "bin" in cl] if self.use_user_info else None

        self.model.fit(cp_train_df, item_info=cp_item_info, user_info=cp_user_info,
                       cols_bin_item=cols_bin_item, cols_bin_user=cols_bin_user)

    def transform(self, ui_pairs: DataFrame):
        self._check_model()
        predictions = self.model.predict(ui_pairs.user_id, ui_pairs.business_id)
        ui_pairs["prediction"] = predictions
        return ui_pairs

    @staticmethod
    def create_dummies(item_info):
        cat_dummy = pd.get_dummies(item_info.top_category, prefix="bin_cat")
        state_dummy = pd.get_dummies(item_info.state, prefix="bin_state")
        item_side_info = item_info.join(cat_dummy).join(state_dummy)
        del item_side_info["top_category"], item_side_info["state"], item_side_info["city"]
        return item_side_info
