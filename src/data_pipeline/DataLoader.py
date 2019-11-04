from pyspark.sql import SparkSession, Window, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pathlib import Path
from src.data_pipeline.Config import parse_config
from src.data_pipeline.pipeline import train_test_split, sample_dataset, filter_outlier_user


class DataLoader:
    def __init__(self,
                 spark: SparkSession,
                 dataset_name,
                 config_name="default_config.json"):
        self.__spark = spark
        self.__dataset_name = dataset_name
        self.__config = parse_config(config_name, dataset_name)
        self.__raw_data = self.__load_data()
        self.__split_params = {
            "partition_by": self.__config.partition_by,
        }

        self.__train_data = None
        self.__test_data = None
        self.__split_dataset()

    def __load_data(self) -> DataFrame:
        cur_file_path = Path(__file__).absolute()
        data_path = cur_file_path.parent.parent.parent / self.__config.data_file
        data_path = data_path.as_uri()

        schema = StructType()
        header = ["user", "item", "rating", "ts"]
        schema.add(StructField(header[0], IntegerType())) \
            .add(StructField(header[1], IntegerType())) \
            .add(StructField(header[2], FloatType())) \
            .add(StructField(header[3], LongType()))

        if "ml-1m" in self.__dataset_name:

            raw_data = self.__spark.sparkContext.textFile(data_path)
            lines = raw_data.map(lambda l: l.split("::")) \
                .map(lambda part: [int(part[0]), int(part[1]), float(part[2]), int(part[3])])
            df = self.__spark.createDataFrame(lines, schema)
        elif "ml-20m" in self.__dataset_name:
            df = self.__spark.read.csv(data_path, sep=",",
                                       schema=schema, enforceSchema=True, header=True)
        else:
            raise ValueError("{} is an unknown dataset".format(self.__dataset_name))

        return df

    def __filter_raw_data(self):
        filtered_df = filter_outlier_user(self.__raw_data)
        return sample_dataset(filtered_df,
                              user_sample_ratio=self.__config.user_sample_ratio,
                              item_sample_ratio=self.__config.item_sample_ratio)

    def __split_dataset(self):
        config = self.__config
        self.__train_data, self.__test_data = train_test_split(self.__filter_raw_data(),
                                                               ratio_range=(0, 0.2),
                                                               partition_by=config.partition_by)

        # self.__train_data.cache()
        # print("caching training set with {} rows".format(self.__train_data.count()))

    def get_dataset_name(self):
        return self.__dataset_name

    def get_train_set(self):
        return self.__train_data

    def get_raw_data(self):
        return self.__raw_data

    def get_test_set(self):
        return self.__test_data

    def get_spark(self):
        return self.__spark

    def get_cache_path(self):
        return self.__config.cache_path

    def get_split_params(self):
        return self.__split_params

    def get_config(self):
        return self.__config