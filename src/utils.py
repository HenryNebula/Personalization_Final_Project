from pyspark.sql import SparkSession


def get_spark(name="Recsys") -> SparkSession:
    spark = SparkSession \
        .builder \
        .appName(name) \
        .getOrCreate()
    return spark