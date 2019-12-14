from pathlib import Path
import json
from collections import namedtuple
import os

Config = namedtuple("Config", ["dataset_name",
                               "train_rating_path",
                               "test_rating_path",
                               "test_neg_samples_path",
                               "user_side_info_path",
                               "item_side_info_path",
                               "as_implicit",
                               "db_path",
                               "cache_path"])

Config.__new__.__defaults__ = ("user_10_item_1_exp",
                               "data/user_10_item_1/yelp.train.csv",
                               "data/user_10_item_1/yelp.test.csv",
                               "data/user_10_item_1/test_candidates.parquet",
                               "data/side_info/user.del_friends.csv",
                               "data/side_info/business_features.csv",
                               False,
                               "sqlite:////home/ds2019/log/meta_data.db",
                               "/home/ds2019/log")


def parse_config(config_file_name, dataset_name) -> Config:
    root_path = Path(__file__).parent.parent.parent
    config_path = root_path / Path("config") / Path(config_file_name)

    if not config_path.exists():
        raise ValueError("{} is not a valid path to a config file".format(config_path))
    with open(config_path) as f:
        config_json = json.loads(f.read())

    if dataset_name not in config_json:
        raise ValueError("Config for {} not found.".format(dataset_name))

    config_dict = config_json[dataset_name]

    required_config_params = ["data_root_path", "user_thresh", "item_thresh",
                              "as_implicit", "db_path", "cache_path"]
    keys = {}

    for key in required_config_params:
        if key not in config_dict:
            raise RuntimeError("A required config parameter {} is not found in the config file ({})"
                               .format(key, config_path))

    for key in Config._fields:
        if key in config_dict:
            keys[key] = config_dict[key]

    keys["dataset_name"] = dataset_name
    data_sub_dir = "user_{}_item_{}".format(config_dict["user_thresh"], config_dict["item_thresh"])
    correct_dataset_name = data_sub_dir + "_exp" if not keys["as_implicit"] else data_sub_dir + "_imp"
    assert correct_dataset_name == dataset_name, "Config ({}) doesn't match with dataset name ({}), " \
                                                 "please double check the name, user_thresh, item_thresh " \
                                                 "and as_implicit fields".format(correct_dataset_name, dataset_name)

    data_dir = os.path.join(config_dict["data_root_path"], data_sub_dir)
    keys["train_rating_path"] = os.path.join(data_dir, "yelp.train.csv")
    keys["test_rating_path"] = os.path.join(data_dir, "yelp.test.csv")
    keys["test_neg_samples_path"] = os.path.join(data_dir, "test_candidates.parquet")
    keys["user_side_info_path"] = os.path.join(config_dict["data_root_path"], "side_info/user.del_friends.csv")
    keys["item_side_info_path"] = os.path.join(config_dict["data_root_path"], "side_info/business_features.csv")

    return Config(**keys)
