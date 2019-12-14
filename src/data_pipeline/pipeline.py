from pyspark.sql import Window, DataFrame, SparkSession
from pyspark.sql.functions import *
from src.model.BaseModel import BaseModel
from src.evaluation.Evaluator import Evaluator
from src.utility.model_utils import get_counts
from collections import defaultdict


def filter_outlier_user(data: DataFrame,
                        threshold=500):
    columns = data.columns
    user_record_counts = get_counts(data)
    filtered_user = user_record_counts.filter(col("record_count") <= threshold)
    return (data
            .join(filtered_user, col("user") == col("filteredUser"))
            .selectExpr(*columns))


def sample_rows(df: DataFrame,
                seed,
                thresh,
                column="user"):
    return (df
            .dropDuplicates([column])
            .withColumn("rand", rand(seed))
            .filter(col("rand") <= thresh)
            .selectExpr("{0} as filtered_{0}".format(column)))


def sample_dataset(data: DataFrame,
                   item_sample_ratio,
                   user_sample_ratio,
                   seed=24):

    if item_sample_ratio <= 0:
        raise ValueError("Item sample ratio {} must be greater than 0".format(item_sample_ratio))

    if user_sample_ratio == 1 and item_sample_ratio == 1:
        return data
    # fix user ratio as 0.1 of the original dataset
    data.cache()
    subset = (data
              .join(sample_rows(data, seed, user_sample_ratio, "user"), col("user") == col("filtered_user"))
              .join(sample_rows(data, seed, item_sample_ratio, "item"), col("item") == col("filtered_item"))
              .selectExpr("user", "item", "rating", "ts"))

    data.unpersist()
    subset.cache()
    print("Using sampled subset with {0:E} records".format(subset.count()))

    return subset


def train_test_split(data: DataFrame,
                     ratio_range=(0, 0.2),
                     partition_by="user",
                     seed=42):
    lb, rb = ratio_range

    order_func = rand(seed=seed)
    window = Window.partitionBy(partition_by).orderBy(order_func)
    df_with_rank = data.withColumn("percentRank", percent_rank().over(window))

    condition = (col("percentRank") >= lb) & (col("percentRank") <= rb)

    train = df_with_rank.filter(~condition).drop("percentRank")
    test = df_with_rank.filter(condition).drop("percentRank")

    print("Using split of range {}, test set contains {} of {} records in total.".format(
        ratio_range, test.count(), data.count()))

    return train, test


def test_evaluation(data_loader,
                    model: BaseModel,
                    spark: SparkSession,
                    metrics=("ndcg", "precision", "rmse"),
                    num_candidates=50,
                    top_k=10,
                    force_rewrite=False,
                    caching=True,
                    oracle_type=None):
    """
    evaluate a certain model on the test set

    :param data_loader: DataLoader object to load data
    :param model: BaseModel object
    :param spark: Spark Session
    :param metrics: a list of metrics to evaluate
    :param num_candidates: number of candidates used in caching, much bigger than top-k
    :param top_k: number of final recommendations, e.g., NDCG@k
    :param force_rewrite: rewrite existing model results or not
    :param caching: use caching system to save model outputs or not
    :param oracle_type: show evaluation result on the same dataset used in the training process
    :return:
    """
    oracle_options = (None, "train", "test")
    oracle_type = oracle_type if oracle_type in oracle_options else None

    evaluator = Evaluator(metrics, top_k, spark)

    train = data_loader.get_train_set() if oracle_type != "test" else data_loader.get_test_set()
    test = data_loader.get_test_set() if oracle_type != "train" else data_loader.get_train_set()

    if oracle_type:
        force_rewrite = True

    pref_dict = evaluator.evaluate(train_df=train,
                                   test_df=test,
                                   user_side_info=data_loader.get_user_side_info(),
                                   item_side_info=data_loader.get_item_side_info(),
                                   test_candidates=data_loader.get_test_candidates(),
                                   data_config=data_loader.get_config(),
                                   model=model,
                                   fold=-1,
                                   num_candidates=num_candidates,
                                   force_rewrite=force_rewrite,
                                   caching=caching)
    return pref_dict
