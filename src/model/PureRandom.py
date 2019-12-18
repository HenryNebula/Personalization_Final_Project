import numpy as np
import pandas as pd

from src.model.BaseModel import BaseModel


class PureRandom(BaseModel):
    def __init__(self, params: dict):
        super().__init__(params)
        self.random_state = params["random_state"]

    def fit(self, train_df, user_info, item_info):
        ## transform to surprise ready
        pass

    def transform(self, ui_pairs):
        ## get predicted score
        predicted = []
        testset_copy = ui_pairs.copy()
        testset_copy['prediction'] = np.random.rand(len(ui_pairs))

        return testset_copy






