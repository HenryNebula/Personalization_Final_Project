from pyspark.sql import SparkSession
from pathlib import Path
from functools import wraps
from time import time
import pandas
import pyspark
from pyspark.sql.types import *


def save_parquet(path: Path, df: pyspark.sql.DataFrame):
    parent = Path(path.parent)
    if parent.exists():
        print("Rewriting files in {}".format(path))
    else:
        print("Creating directory and start writing ...")
        parent.mkdir(parents=True)
    df.write.parquet(path.absolute().as_uri(), mode="overwrite")


def load_parquet(spark: SparkSession, path: Path) -> pyspark.sql.DataFrame:
    if path.exists():
        print("Using cached file from {}".format(path))
        data = spark.read.parquet(path.absolute().as_uri())
    else:
        data = None
    return data


def pandas_to_spark(df_pandas: pandas.DataFrame, spark: SparkSession) -> pyspark.sql.DataFrame:
    # Given pandas dataframe, it will return a spark's dataframe
    # from https://gist.github.com/zaloogarcia/11508e9ca786c6851513d31fb2e70bfc

    def equivalent_type(f):
        if f == 'datetime64[ns]':
            return DateType()
        elif f == 'int64':
            return LongType()
        elif f == 'int32':
            return IntegerType()
        elif f == 'float64':
            return FloatType()
        else:
            return StringType()

    def define_structure(string, format_type):
        try:
            typo = equivalent_type(format_type)
        except:
            typo = StringType()
        return StructField(string, typo)

    columns = list(df_pandas.columns)
    types = list(df_pandas.dtypes)
    struct_list = []
    for column, typo in zip(columns, types):
        struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return spark.createDataFrame(df_pandas, p_schema)


def get_counts(df: pyspark.sql.DataFrame, group="user"):
    return (df
            .groupBy(group)
            .count()
            .withColumnRenamed("count", "record_count")
            .withColumnRenamed("user", "filteredUser"))


def count_df_to_list(stats: pyspark.sql.DataFrame):
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
