from pyspark.sql import SparkSession, DataFrame
from pathlib import Path
from functools import wraps
from time import time


def save_parquet(path: Path, df: DataFrame):
    parent = Path(path.parent)
    if parent.exists():
        print("Rewriting files in {}".format(path))
    else:
        print("Creating directory and start writing ...")
        parent.mkdir(parents=True)
    df.write.parquet(path.absolute().as_uri(), mode="overwrite")


def load_parquet(spark: SparkSession, path: Path) -> DataFrame:
    if path.exists():
        print("Using cached file from {}".format(path))
        data = spark.read.parquet(path.absolute().as_uri())
    else:
        data = None
    return data


def get_counts(df: DataFrame,
               group="user"):
    return (df
            .groupBy(group)
            .count()
            .withColumnRenamed("count", "record_count")
            .withColumnRenamed("user", "filteredUser"))


def count_df_to_list(stats: DataFrame):
    return [int(row.record_count) for row in stats.collect()]


def timeit(f):
    # decorator for timing a function
    # using sample code from https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        print("It took function {0} {1: 2.4f} seconds to run with parameters [{2}, {3}]"
              .format(f.__name__, te-ts, args, kwargs))
        return result
    return wrap
