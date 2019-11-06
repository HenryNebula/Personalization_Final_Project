# Personalization_Proj1
First project for IEOR 4571: Personalizaton Theory and Application

## Description

Build and test a demo Recsys on ML-20M dataset, with two fundamental groups of algorithms, namely neighborhood-based and MF-based models, using PySpark. It is expected to mimic the scenario of a digital media company that needs to recommend movies. **The final report can be found [here](report/Personalization_Project1_report.pdf).**

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
* Inference and evaluation module (Precision@K/NDCG@K)
* Parameter tuning system and logger with the help of sqlite database
* Result interpretation (graph+table)
* Re-iterate training method


## Structure of source code

* Data loading modules, including data loader, train-test split and cross-validation. 
    * [Data_Loader](src/data_pipeline/Data_Loader.py) Major class for loading data, it splits train/test set once initialized
    * [Data_Loader](src/data_pipeline/Config.py) parses and loads configuration from json files under config folder The default is good for a try. If you want to create a different configuration file, be sure to change the name of the configuration (i.e, the name of dataset) and follow the instructions listed [here](config/README.md). Otherwise it will cause inconsistency in the database. Later commits will try to check this inconsistency automatically and throw an error if the check fails.
    * [pipeline](src/data_pipeline/pipeline.py) includes cross_validation and test_performance function.
    
* Models:
    * [Base model](src/model/BaseModel.py) all models should inherit from this base model by overloading three specific functions, namely fit(), transform() and recommend_for_all_users(). Check the annotation of the input parameters for more details.
    * [ALS_MF](src/model/ALS_MF.py) an inheritance example which is wrapper using ALS method from Spark MLlib.
    * [kNN](src/model/KNN.py) an item based KNN model, implemented from scratch using PySpark

* Evaluations:
    * [Evaluator](src/evaluation/Evaluator.py) no need to construct an instance of this manually, high level functions in pipeline parts call it already.

* Utilities:
    * [Database tools](src/utility/DBUtils.py) handle the definition and insertion of a database, where the records of cross_validation and tests are saved. It will bring much convenience of tuning hyper parameters and model selections.
    * [Summary](src/utility/Summary.py) provides functions to query the database, basically summarizing the cross validation results across all folds and fetching result for the final test evaluation.
    * [Others] other two files end with "utils" include functions for handling paths and model result saving.

* Exploratory Data Analysis:
    * Code and Notebooks of exploratory data analysis and final result plotting can be found [here](src/eda).

## Local testing and debugging
Say if you want to debug your own model using existing methods and functions, follow the following step:

1. extract ML20M dataset first from the zip file
    ```bash
    # suppose you are in the root dir of this repo
    cd data/
    unzip ml-20m.zip
    ``` 

2. Create a directory called "test" under the root directory and after this step, the repo should look like this:
    ```bash
    .
    ├── config
    ├── data
    ├── LICENSE
    ├── README.md
    ├── recsys_all_in_one.ipynb
    ├── report
    ├── requirements.txt
    ├── src
    └── test
    ```
    It will not be included in version control since it's set ignored.
3. Copy [this notebook](./recsys_all_in_one.ipynb) to your "test" directory.

4. Do all of your testing in the copied notebook. Happy debugging!

Note that all package dependencies are specified in the [requirements](requirements.txt) file. Make sure you have [spark](https://spark.apache.org/) installed (for this project, we use the latest version 2.4.4) and environment variables already set up. Also if you want to use the database logging system, make sure to have [sqlite3](https://www.sqlite.org/download.html) installed in your environment.
