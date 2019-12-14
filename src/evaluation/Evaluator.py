from pyspark.sql import SparkSession, DataFrame as SparkDF
from pyspark.sql.functions import *
from pyspark.mllib.evaluation import RankingMetrics, RegressionMetrics
from src.data_pipeline.Config import Config
from src.model.BaseModel import BaseModel
from collections import ChainMap
from src.utility.sys_utils import construct_path
from src.utility.model_utils import load_parquet, save_parquet
from src.utility import DBUtils
import pandas


class Evaluator:
    def __init__(self,
                 metrics,
                 top_k,
                 spark: SparkSession):
        self.__test = None
        self.top_k = top_k
        self.__spark = spark

        is_ranking_metric = {"ndcg": True, "precision": True,
                             "rmse": False, "mae": False, "rsquared": False}

        diff_metrics = set(metrics).difference(set(is_ranking_metric))
        if diff_metrics:
            raise ValueError("Metric(s) {{}} not supported yet, only these are supported: {}." \
                             .format(diff_metrics, is_ranking_metric))

        self.ranking_metrics = list(filter(lambda k: is_ranking_metric[k], metrics))
        self.regression_metrics = list(filter(lambda k: not is_ranking_metric[k], metrics))

    def __load_test(self, test_df: SparkDF):
        self.__test = test_df

    def __check_test_exists(self):
        if not self.__test:
            raise ValueError("test data not loaded yet")

    def __evaluate_ranking(self, rnk_inf: SparkDF):
        test_ground_truth = self.__test.groupBy("user_id").agg(collect_list("business_id").alias("business_gt"))

        pred_with_labels = rnk_inf.join(test_ground_truth, on="user_id").drop("user_id")

        metrics = RankingMetrics(pred_with_labels.rdd)

        results = {}

        for m in self.ranking_metrics:
            metric_name = "{}@{}".format(m, self.top_k)
            if "ndcg" in m:
                results[metric_name] = metrics.ndcgAt(self.top_k)
            elif m == "precision":
                results[metric_name] = metrics.precisionAt(self.top_k)

        return results

    def __evaluate_rating(self, rat_inf: SparkDF):

        # RegressionMetrics
        pred_with_labels = (rat_inf
                            .na.drop()
                            .select(col("stars").cast("double").alias("label"),
                                    col("prediction").cast("double")))

        metrics = RegressionMetrics(pred_with_labels.rdd.map(lambda x: (x.prediction, x.label)))

        results = {}

        for m in self.regression_metrics:
            if m == "rmse":
                results[m] = metrics.rootMeanSquaredError
            elif m == "mae":
                results[m] = metrics.meanAbsoluteError
            elif m == "rsquared":
                results[m] = metrics.r2

        return results

    @staticmethod
    def __get_paths(model: BaseModel,
                    data_config: Config,
                    fold):
        options = [True, False]
        paths = []
        for opt in options:
            path = construct_path(data_config.dataset_name,
                                  model=model,
                                  fold=fold,
                                  is_ranking=opt,
                                  cache_path=data_config.cache_path)
            paths.append(path)

        return paths

    def __pandas_df_to_spark_df(self, pandas_df):
        return self.__spark.createDataFrame(pandas_df)

    def evaluate(self,
                 train_df: pandas.DataFrame,
                 test_df: pandas.DataFrame,
                 test_candidates: pandas.DataFrame,
                 user_side_info: pandas.DataFrame,
                 item_side_info: pandas.DataFrame,
                 data_config: Config,
                 model: BaseModel,
                 fold,
                 num_candidates,
                 force_rewrite,
                 caching=True):
        
        # save recommendation results to disk if caching is set to True
        # so that future evaluation will based on cached rankings to accelerate the computation

        rnk_inf_path, rat_inf_path = self.__get_paths(model, data_config, fold)

        rnk_inf, rat_inf = None, None

        if caching and not force_rewrite:
            # load cache by default (if exists)
            rnk_inf = load_parquet(self.__spark, rnk_inf_path)
            rat_inf = load_parquet(self.__spark, rat_inf_path)

        if not rnk_inf or not rat_inf or force_rewrite:
            model.fit(train_df=train_df, user_info=user_side_info, item_info=item_side_info)

            if not rnk_inf:
                recommendations = model.recommend_on_candidates(test_candidates, num_candidates)
                rnk_inf = self.__pandas_df_to_spark_df(recommendations)
                if caching:
                    save_parquet(rnk_inf_path, rnk_inf)

            if not rat_inf:
                rat_inf = model.transform(test_df)
                rat_inf = self.__pandas_df_to_spark_df(rat_inf)
                if caching:
                    save_parquet(rat_inf_path, rat_inf)

        self.__load_test(self.__pandas_df_to_spark_df(test_df))
        print("Finish inference phase")
        rank_dict = self.__evaluate_ranking(rnk_inf)
        rate_dict = self.__evaluate_rating(rat_inf)

        results = [rank_dict, rate_dict]

        for i in range(len(results)):
            path = rnk_inf_path if i == 0 else rat_inf_path
            result = results[i]

            if result:
                base_data = DBUtils.MetaData(dataset=data_config.dataset_name,
                                             model=model.get_name(),
                                             hyper=str(model.sort_params_to_list()),
                                             fold=fold,
                                             path=str(path))

                rows = DBUtils.generate_rows(result.items(), base_data)
                DBUtils.insert(rows, DBUtils.get_engine(data_config.db_path))

        return ChainMap(rank_dict, rate_dict)
