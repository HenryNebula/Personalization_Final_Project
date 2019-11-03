from pathlib import Path
import json
from collections import namedtuple


Config = namedtuple("Config", ["dataset_name",
                               "partition_by",
                               "sample_ratio",
                               "data_file",
                               "db_path",
                               "cache_path"])

Config.__new__.__defaults__ = ("ml-1m-full",
                               "user",
                               1,
                               "data/ml-1m/ratings.dat",
                               "sqlite:////home/ds2019/log/meta_data.db",
                               "/home/ds2019/log")


def parse_config(config_file_name, dataset_name) -> Config:
    root_path = Path(__file__).parent.parent.parent
    config_path = root_path / Path("config") / Path(config_file_name)

    if not config_path.exists():
        raise ValueError("{} is not a valid path to a config file".format(config_path))
    with open(config_path) as f:
        config_json = json.loads(f.read())

    if not dataset_name in config_json:
        raise ValueError("Config for {} not found.".format(dataset_name))

    config_dict = config_json[dataset_name]
    config_dict["dataset_name"] = dataset_name

    keys = {}

    for key in config_dict:
        if key in Config._fields:
            keys[key] = config_dict[key]
        else:
            print("Warning: {} is not a valid parameter in config {}"
                  .format(key, dataset_name))
    return Config(**keys)