from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.mllib.evaluation import RankingMetrics
from src.data_pipeline.Config import Config
from src.model.BaseModel import BaseModel
from collections import ChainMap
from src.utility.sys_utils import construct_path
from src.utility.model_utils import load_parquet, save_parquet
from src.utility import DBUtils


class Evaluator:
    def __init__(self,
                 metrics,
                 top_k,
                 spark: SparkSession):
        self.__test = None
        self.top_k = top_k
        self.__spark = spark

        is_ranking_metric = {"ndcg": True, "precision": True}
        diff_metrics = set(metrics).difference(set(is_ranking_metric))
        if diff_metrics:
            raise ValueError("Metric(s) {{}} not supported yet, only these are supported: {}." \
                             .format(diff_metrics, is_ranking_metric))

        self.ranking_metrics = list(filter(lambda k: is_ranking_metric[k], metrics))
        self.regression_metrics = list(filter(lambda k: not is_ranking_metric[k], metrics))

    def __load_test(self, test_df: DataFrame):
        self.__test = test_df

    def __check_test_exists(self):
        if not self.__test:
            raise ValueError("test data not loaded yet")

    def __evaluate_ranking(self, rnk_inf: DataFrame):
        test_ground_truth = self.__test.groupBy("user") \
            .agg(collect_list("item").alias("item_gt"))

        pred_with_labels = rnk_inf \
            .withColumn("pred", col("recommendations.item")) \
            .join(test_ground_truth, on="user") \
            .drop("user", "recommendations")

        metrics = RankingMetrics(pred_with_labels.rdd)

        results = {}

        for m in self.ranking_metrics:
            metric_name = "{}@{}".format(m, self.top_k)
            if "ndcg" in m:
                results[metric_name] = metrics.ndcgAt(self.top_k)
            elif m == "precision":
                results[metric_name] = metrics.precisionAt(self.top_k)

        return results

    def __evaluate_rating(self, rat_inf: DataFrame):
        print("Dummy printing of test set count in Evaluator.__evaluate_rating(): {}"
              .format(self.__test.count()))
        return {}

    def __get_paths(self,
                    model: BaseModel,
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

    def evaluate(self,
                 train_df: DataFrame,
                 test_df: DataFrame,
                 data_config: Config,
                 model: BaseModel,
                 fold,
                 num_candidates,
                 force_rewrite):

        rnk_inf_path, rat_inf_path = self.__get_paths(model, data_config, fold)

        rnk_inf, rat_inf = None, None

        if not force_rewrite:
            # load cache by default (if exists)
            rnk_inf = load_parquet(self.__spark, rnk_inf_path)
            rat_inf = load_parquet(self.__spark, rat_inf_path)

        if not rnk_inf or not rat_inf or force_rewrite:
            model.fit(train_df)

            if not rnk_inf:
                rnk_inf = model.recommend_for_all_users(num_candidates)
                save_parquet(rnk_inf_path, rnk_inf)

            if not rat_inf:
                rat_inf = model.transform(test_df)
                save_parquet(rat_inf_path, rat_inf)

        self.__load_test(test_df)
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
