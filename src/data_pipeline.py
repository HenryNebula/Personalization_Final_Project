from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import Window, WindowSpec
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pathlib import Path
from src.BaseModel import BaseModel
from src.Evaluator import Evaluator
from collections import defaultdict


def load_data(spark: SparkSession,
              dataset="ml-1m",
              base_path="data/",
              file_name="ratings.dat") -> DataFrame:

    cur_file_path = Path(__file__).absolute()
    base_path = cur_file_path.parent.parent / base_path

    schema = StructType()
    header = ["user", "item", "rating", "ts"]
    schema.add(StructField(header[0], IntegerType())) \
        .add(StructField(header[1], IntegerType())) \
        .add(StructField(header[2], FloatType())) \
        .add(StructField(header[3], LongType()))

    data_path = base_path / dataset / file_name
    print(data_path)
    data_path = data_path.as_uri()

    if dataset == "ml-1m":
        raw_data = spark.sparkContext.textFile(data_path)
        lines = raw_data.map(lambda l: l.split("::")) \
            .map(lambda part: [int(part[0]), int(part[1]), float(part[2]), int(part[3])])
        df = spark.createDataFrame(lines, schema)
    elif dataset == "ml-20m":
        df = spark.read.csv(data_path, sep=",",
                            schema=schema, enforceSchema=True, header=True)
    else:
        raise ValueError("{} is an unknown dataset".format(dataset))

    return df


def train_test_split(data: DataFrame,
                     ratio_range=(0, 0.2),
                     min_ratings=1,
                     partition_by="user",
                     rank_method="random",
                     seed=42):

    lb, rb = ratio_range
    # TODO: min ratings filtering

    order_func = rand(seed=seed) if rank_method == "random" else "ts"
    window = Window.partitionBy(partition_by).orderBy(order_func)
    df_with_rank = data.withColumn("percentRank", percent_rank().over(window))

    condition = (col("percentRank") >= lb) & (col("percentRank") <= rb)

    train = df_with_rank.filter(~condition).drop("percentRank")
    test = df_with_rank.filter(condition).drop("percentRank")

    print("Using split of range {}, test set contains {} of {} records in total.".format(
        ratio_range, test.count(), data.count()))

    return train, test


def cross_validation(full_train: DataFrame,
                     model: BaseModel,
                     k_fold=5,
                     metrics=("ndcg", "precision"),
                     num_candidates=500, # a threshold for evaluation, much bigger than top_k
                     top_k=5,
                     split_params=None):

    if split_params is None:
        split_params = {"partition_by": "user", "rank_method": "random"}

    result = defaultdict(list)

    for m in metrics:
        result[m] = []

    if k_fold < 3:
        ValueError("{} must be great than 3 to be realistic".format(k_fold))

    for k in range(k_fold):
        ratio_range = [k / k_fold, (k + 1) / k_fold]
        train, test = train_test_split(full_train, ratio_range, **split_params)

        model.fit(train)
        pref_dict = evaluate_model(train, test, model, metrics, num_candidates, top_k)
        for m in result:
            result[m] += pref_dict[m]
        # TODO: add summary value

    return result


def evaluate_model(train: DataFrame,
                   test: DataFrame,
                   model: BaseModel,
                   metrics=("ndcg", "precision"),
                   num_candidates=500,
                   top_k=5):

    perf_dict = defaultdict(list)

    model.fit(train)
    recommendations = model.recommend_for_all_users(num_candidates)

    evaluator = Evaluator(test, metrics, top_k)
    rank_dict = evaluator.evaluate_ranking(recommendations)

    for m in rank_dict:
        perf_dict[m].append(rank_dict[m])

    return perf_dict