from pyspark.sql import Window, DataFrame, SparkSession
from pyspark.sql.functions import *
from src.model.BaseModel import BaseModel
from src.evaluation.Evaluator import Evaluator
from collections import defaultdict
# todo: fix cyclic importing issue of DataLoader class


def train_test_split(data: DataFrame,
                     ratio_range=(0, 0.2),
                     min_ratings=1,
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

    test.cache()
    print("Using split of range {}, test set contains {} of {} records in total.".format(
        ratio_range, test.count(), data.count()))

    return train, test


def cross_validation(data_loader,
                     model: BaseModel,
                     spark: SparkSession,
                     k_fold=5,
                     metrics=("ndcg", "precision"),
                     num_candidates=500,  # a threshold for evaluation, much bigger than top_k
                     top_k=5,
                     force_rewrite=False):
    result = defaultdict(list)

    if k_fold < 3:
        ValueError("{} must be great than 3 to be realistic".format(k_fold))

    evaluator = Evaluator(metrics, top_k, spark)

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
        for m in pref_dict:
            result[m].append(pref_dict[m])
        # TODO: add summary value

    return result


def test_evaluation(data_loader,
                    model: BaseModel,
                    spark: SparkSession,
                    metrics=("ndcg", "precision"),
                    num_candidates=500,  # a threshold for evaluation, much bigger than top_k
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
