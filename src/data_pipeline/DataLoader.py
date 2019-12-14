import pandas as pd
from pandas import DataFrame
from pathlib import Path
from src.data_pipeline.Config import parse_config


class DataLoader:
    def __init__(self,
                 dataset_name,
                 config_name="default_config.json"):
        self.__dataset_name = dataset_name
        self.__config = parse_config(config_name, dataset_name)

    @staticmethod
    def __construct_abspath(path):
        cur_file_path = Path(__file__).absolute()
        abs_path = cur_file_path.parent.parent.parent / path
        return abs_path

    @staticmethod
    def explicit_to_implicit(df: DataFrame) -> DataFrame:
        df.stars = 1
        return df

    def __load_rating_file(self, file_path) -> DataFrame:
        rating_df = pd.read_csv(self.__construct_abspath(file_path))[["user_id", "business_id", "stars"]]
        return rating_df if not self.__config.as_implicit else self.explicit_to_implicit(rating_df)

    def __load_test_candidates(self, file_path) -> DataFrame:
        return pd.read_parquet(self.__construct_abspath(file_path))

    def __load_side_info(self, file_path) -> DataFrame:
        return pd.read_csv(self.__construct_abspath(file_path))

    def get_dataset_name(self):
        return self.__dataset_name

    def get_train_set(self):
        return self.__load_rating_file(self.__config.train_rating_path)

    def get_test_set(self):
        return self.__load_rating_file(self.__config.test_rating_path)

    def get_test_candidates(self):
        return self.__load_test_candidates(self.__config.test_neg_samples_path)

    def get_user_side_info(self):
        return self.__load_side_info(self.__config.user_side_info_path)

    def get_item_side_info(self):
        return self.__load_side_info(self.__config.item_side_info_path)

    def get_cache_path(self):
        return self.__config.cache_path

    def get_config(self):
        return self.__config