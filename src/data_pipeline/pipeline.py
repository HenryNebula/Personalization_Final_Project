from pyspark.sql import Window, DataFrame, SparkSession
from pyspark.sql.functions import *
from src.model.BaseModel import BaseModel
from src.evaluation.Evaluator import Evaluator
from collections import defaultdict
# todo: fix cyclic importing issue of DataLoader class


def sample_dataset(data: DataFrame,
                   sample_ratio=0.2,
                   seed=24):

    if sample_ratio <= 0:
        raise ValueError("{} must be greater than 0".format(sample_ratio))

    if sample_ratio >= 1:
        return data

    user_window = Window.orderBy("user")
    item_window = Window.orderBy("item")

    mod = int(1 / sample_ratio)

    subset = (data
              .filter(col("user") % mod == 0)
              .filter(col("item") % mod == 0)
              .withColumn("user_dense", dense_rank().over(user_window))
              .withColumn("item_dense", dense_rank().over(item_window))
              .selectExpr("user_dense as user", "item_dense as item", "rating", "ts"))

    # subset.cache()
    print("Using sampled subset with {0:E} records".format(subset.count()))

    return subset


def train_test_split(data: DataFrame,
                     ratio_range=(0, 0.2),
                     partition_by="user",
                     seed=42):
    lb, rb = ratio_range
    # TODO: min ratings filtering

    order_func = rand(seed=seed)
    window = Window.partitionBy(partition_by).orderBy(order_func)
    df_with_rank = data.withColumn("percentRank", percent_rank().over(window))

    condition = (col("percentRank") >= lb) & (col("percentRank") <= rb)

    train = df_with_rank.filter(~condition).drop("percentRank")
    test = df_with_rank.filter(condition).drop("percentRank")

    print("Using split of range {}, test set contains {} of {} records in total.".format(
        ratio_range, test.count(), data.count()))

    return train, test


def cross_validation(data_loader,
                     model: BaseModel,
                     spark: SparkSession,
                     k_fold=5,
                     metrics=("ndcg", "precision"),
                     num_candidates=200,  # a threshold for evaluation, much bigger than top_k
                     top_k=5,
                     force_rewrite=False):
    result = defaultdict(list)

    if k_fold < 3:
        ValueError("{} must be great than 3 to be realistic".format(k_fold))

    evaluator = Evaluator(metrics, top_k, spark)

    train_whole = data_loader.get_train_set()
    train_whole.cache()
    exp_count = int(train_whole.count()) / k_fold
    for k in range(k_fold):
        ratio_range = [k / k_fold, (k + 1) / k_fold]
        train, test = train_test_split(data_loader.get_train_set(),
                                       ratio_range,
                                       **data_loader.get_split_params())

        pref_dict = evaluator.evaluate(train_df=train,
                                       test_df=test,
                                       data_config=data_loader.get_config(),
                                       model=model,
                                       fold=k,
                                       num_candidates=num_candidates,
                                       force_rewrite=force_rewrite)
        real_count = int(test.count())
        correction = real_count / exp_count
        for m in pref_dict:
            result[m].append(pref_dict[m] * correction)
        # TODO: add summary value

    return result


def test_evaluation(data_loader,
                    model: BaseModel,
                    spark: SparkSession,
                    metrics=("ndcg", "precision"),
                    num_candidates=200,  # a threshold for evaluation, much bigger than top_k
                    top_k=5,
                    force_rewrite=False,
                    oracle_type=None):

    oracle_options = (None, "train", "test")
    oracle_type = oracle_type if oracle_type in oracle_options else None

    evaluator = Evaluator(metrics, top_k, spark)

    train = data_loader.get_train_set() if oracle_type != "test" else data_loader.get_test_set()
    test = data_loader.get_test_set() if oracle_type != "train" else data_loader.get_train_set()

    if oracle_type:
        force_rewrite = True

    pref_dict = evaluator.evaluate(train_df=train,
                                   test_df=test,
                                   data_config=data_loader.get_config(),
                                   model=model,
                                   fold=-1,
                                   num_candidates=num_candidates,
                                   force_rewrite=force_rewrite)
    return pref_dict
