from src.utility.DBUtils import get_engine
from src.model.BaseModel import BaseModel
import pandas as pd
import numpy as np


class Summary:
    # use a sqlite database to save and fetch experiment results
    def __init__(self, db_path):
        self.__engine = get_engine(db_path)
        self.table = None
        self.__update_table()

    @staticmethod
    def __get_latest_result(df, group_list):
        latest = (df.assign(rnk=df.groupby(group_list)['ts']
                        .rank(method='first', ascending=False))) \
            .query('rnk < 2') \
            .drop(columns=["rnk"])
        return latest
    
    def __update_table(self):
        table = pd.read_sql("select * from MetaData", self.__engine, index_col="id")
        table.ts = table.ts.astype(np.datetime64)
        self.table = table.copy()

    def summarize_cv(self,
                     dataset_name,
                     metrics=None):
        
        self.__update_table()
        metrics = self.table["metric"].unique() if not metrics else metrics

        df = self.table[(self.table["dataset"] == dataset_name)
                        & (self.table["metric"].isin(metrics))
                        & (self.table["fold"] >= 0)] \
            .drop(columns=["dataset", "path"])

        # return the latest result of each fold
        df = self.__get_latest_result(df, ["model", "hyper", "metric", "fold"])

        summary = df.groupby(["model", "hyper", "metric"]) \
            .agg(mean=("value", np.mean),
                 std=("value", np.std)) \
            .reset_index(inplace=False)

        rank_summary = summary.assign(rnk=summary.groupby("metric")['mean']
                                      .rank(method='first', ascending=False)) \
            .sort_values(["metric","rnk"])

        return rank_summary

    def get_model_test_perf(self,
                            dataset_name,
                            model_name):

        self.__update_table()
        df = self.table[(self.table["dataset"] == dataset_name)
                        & (self.table["fold"] == -1)
                        & (self.table["model"] == model_name)] \
            .drop(columns=["dataset", "path", "fold"])

        df = self.__get_latest_result(df, ["model", "hyper", "metric"])

        return df
    
    def get_optimal_params(self, dataset_name, model_name, metric):
        test_perf = self.get_model_test_perf(dataset_name, model_name)
        if len(test_perf) == 0:
            return None
        ascending = True if metric in ["rmse"] else False
        filtered = test_perf[test_perf.metric == metric].sort_values("value", ascending=ascending)
        filtered.reset_index(drop=True, inplace=True)
        hyper = filtered.loc[0, "hyper"]
        value = filtered.loc[0, "value"]
        print("Best {} of {} is found as {}".format(metric, model_name, value))
           
        return hyper
    
    def get_result_for_params(self, dataset_name, model_name, hyper, metric, verbose=True):
        test_perf = self.get_model_test_perf(dataset_name, model_name)

        filtered = test_perf[(test_perf.model == model_name) & (test_perf["hyper"] == hyper) & (test_perf["metric"] == metric)].reset_index(drop=True)

        if len(filtered) == 0:
            return pd.DataFrame()

        value = filtered.loc[0, "value"]
        
        if verbose:
            print("For model {}, it has a {} of {}".format(model_name, metric, value))

        return filtered[["model", "hyper", "metric", "value", "ts"]]

    def drop_dupicate_results(self):
        return self.__get_latest_result(self.table, ["model", "hyper", "metric"]).reset_index(drop=True)