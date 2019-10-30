from pyspark.sql import SparkSession, DataFrame
from pathlib import Path


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