## Instruction for writing a valid config file

1. Every config is a json file, containing a nested dict
2. Each entry is a config dict for a specific dataset, e.g.,
    ```json
      {"ml-1m-full": {
        "partition_by": "user",
        "item_sample_ratio": 1,
        "user_sample_ratio": 1,
        "data_file": "data/ml-1m/ratings.dat",
        "db_path": "sqlite:////home/ds2019/log/meta_data.db",
        "cache_path": "/home/ds2019/log"
      }}
    ```
3. Each keyword implies how to sample the dataset or where to save the results
    * partition_by: "user" or "item", means that the train-test split is based on each user (i.e. leave some items out) or based on each item (i.e. leave some users out).
    * item_sample_ratio: 0~1, how much ratio of items to include in the sampled subset
    * user_sample_ratio: 0~1, 0.1 by default, how much ratio of items to include in the sampled subset
    * data_file: where to find the original dataset
    * db_path: a database uri specifying the location of database
    * cache_path: a path where you want to save all the results for accelerating future evaluation

Note that once one of the parameters is changed, you should always give the dataset a different name, otherwise it will lead to inconsistency in the database.