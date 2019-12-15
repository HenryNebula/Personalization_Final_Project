## Instruction for writing a valid config file

1. Every config is a json file, containing a nested dict
2. Each entry is a config dict for a specific dataset, e.g.,
    ```json
      {"user_10_item_1_exp": {
        "data_root_path": "/media/ExtHDD01/recsys_data/yelp_dataset/tidy_data/",
        "user_thresh": 10,
        "item_thresh": 1,
        "as_implicit": false,
        "db_path": "sqlite:////media/ExtHDD01/log/metadata.db",
        "cache_path": "/media/ExtHDD01/log"}}
    ```
3. Each keyword implies data path, filter threshold 
    * data_root_path: directory containing data files
    * user_thresh: positive int, each user rates at least this number of business in the original dataset
    * item_thresh: positive int, each business is rated at least this number of times in the original dataset
    * as_implicit: whether to transform explicit ratings to implicit ones
    * db_path: a database uri specifying the location of database
    * cache_path: a path where you want to save all the results for accelerating future evaluation (this path should exist before running the code)

Note that once one of the parameters is changed, you should always give the dataset a different name, otherwise it will lead to inconsistency in the database. Also the user_thresh, item_thresh value and as_implicit
should be consistent with the dataset name.