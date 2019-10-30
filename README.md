# Personalization_Proj1
First project for IEOR 4571: Personalizaton Theory and Application

## Describtion

Build and test a demo Recsys on ML-20M dataset, with two fundamental groups of algorithms, namely neighborhood-based and MF-based models, using PySpark. It is expected to mimic the scenario of a a digital media company that needs to recommend movies. Several underlying problems including sampling of dataset, evaluation metrics and business analysis of the output require careful considerations.

## Problems to solve

* Data sampling method
* Evaluation Metric
* Algorithm selection
* Correctness of implementation
* Model fine-tuning
* Scalability
* Insightful explanation using visualization tools

## System design

* Data pipeline (load/sample/split)
* Model implementation (unified API)
* Inference and evaluation module (HR@K/Precision@K/Recall@5/NDCG@5)
* Parameter tuning system and logger
* Result interpretation (graph+table)
* Re-iterate training method


## Structure of repo

* [Data loading modules](src/data_pipeline/), including data loader, train-test split and cross-validation. 
    * [Data_Loader](src/data_pipeline/Data_Loader.py) Major class for loading data, it splits train/test set once initilized
    * [Data_Loader](src/data_pipeline/Config.py) parses and loads configuration from json files under config folder The default is good for a try. If you want to create a different configuration, be sure to change the name of the configuration (i.e, the name of dataset). Otherwisem it will cause inconsistency in the database. Later commits will try to check this inconsistency automatically and throw an error if the check fails.
    * [pipeline](src/data_pipeline/pipeline.py) includes cross_validation and test_performance function.
    
* Models:
    * [Base model](src/model/BaseModel.py) all models should inherit from this base model by overloading three specfic functions, namely fit(), transform() and recommend_for_all_users(). Check the annotation of the input parameters for more details.
    * [ALS_MF](src/model/ALS_MF.py) an inheritance example which is wrapper using ALS method from Spark Mllib.
    * kNN model is still on progress

* Evaluations:
    * [Evaluator](src/evaluation/Evaluator.py) no need to construct an instance of this manually, high level functions in pipeline parts call it already.

* Utilities:
    * [Database tools](src/utility/DBUtils.py) handle the definition and insertion of a database, where the records of cross_validation and tests are saved. It will bring much convenience of tuning hyper parameters and model selections.
    * [Summary](src/utility/Summary.py) provides functions to query the database, basically summarzing the cross validation results across all folds and fetching result for the final test evaluation.
    * [Others] other two files end in "utils" include functions for handling paths and model result saving.

## Local testing and debugging
Say if you want to debug your own model using existing methods and functions, follow the following step:
1. Create a directory called "test" under the root directory and after this step, the repo should look like this:
    ```bash
    .
    ├── data
    ├── instructions
    ├── LICENSE
    ├── README.md
    ├── recsys_all_in_one.ipynb
    ├── src
    └── test
    ```
    It will not be included in version control since it's set ignored.
2. Copy [this notebook](./recsys_all_in_one.ipynb) to your "test" directory.

3. Do all of your testing in the copied notebook. Happy debugging!

## Test