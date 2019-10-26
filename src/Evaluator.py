from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.mllib.evaluation import RankingMetrics


class Evaluator:
    def __init__(self,
                 test_df: DataFrame,
                 metrics,
                 top_k):
        self.test = test_df
        self.top_k = top_k

        is_ranking_metric = {"ndcg": True, "precision": True}

        diff_metrics = set(metrics).difference(set(is_ranking_metric))
        if diff_metrics:
            raise ValueError("Metric(s) {{}} not supported yet, only these are supported: {}." \
                             .format(diff_metrics, is_ranking_metric))

        self.ranking_metrics = list(filter(lambda k: is_ranking_metric[k], metrics))
        self.regression_metrics = list(filter(lambda k: not is_ranking_metric[k], metrics))

    def evaluate_ranking(self, recommendations: DataFrame):

        test_ground_truth = self.test.groupBy("user") \
            .agg(collect_list("item").alias("item_gt"))

        pred_with_labels = recommendations \
            .withColumn("pred", col("recommendations.item")) \
            .join(test_ground_truth, on="user") \
            .drop("user", "recommendations")

        metrics = RankingMetrics(pred_with_labels.rdd)

        results = {}

        for m in self.ranking_metrics:
            if m == "ndcg":
                results[m] = metrics.ndcgAt(self.top_k)
            elif m == "precision":
                results[m] = metrics.precisionAt(self.top_k)

        return results

    def evaluate_rating(self, ui_recommends: DataFrame):
        print(self.test)

    @classmethod
    def print_name(cls):
        print(cls.__name__)
