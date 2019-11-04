from pyspark.sql import SparkSession, DataFrame
from src.model.BaseModel import BaseModel
from pathlib import Path


def get_spark(name="Recsys", cores=2) -> SparkSession:
    spark = (SparkSession
             .builder
             .appName(name)
             .master("local[{}]".format(cores))
             .config("spark.memory.offHeap.enabled", True)
             .config("spark.memory.offHeap.size", "16g")
             .config("spark.local.dir", "/tmp/spark-temp")
             .getOrCreate())

    spark.sparkContext.setCheckpointDir("/tmp/spark-temp/chkpts")
    return spark


def construct_path(dataset,
                   model: BaseModel,
                   fold,
                   cache_path,
                   is_ranking=True):
    params = model.sort_params_to_list()
    prefix = "Ranking" if is_ranking else "Rating"

    params_str = "-".join(list(map(lambda x: "{}_{}".format(x[0], x[1]),
                                   params)))

    path = Path(cache_path) \
           / Path(dataset) \
           / Path(model.__class__.__name__) \
           / Path(params_str) \
           / Path("{}_fold_{}.parquet".format(prefix, fold))

    return path
