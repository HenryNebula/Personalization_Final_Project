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

* [Data pipeline function](src/data_pipeline.py), including data loading function, train-test split and cross-validation. A more detailed example can be found at [this notebook]() at root dir
* Models:
    * [Base model](src/BaseModel.py) to inherit from
    * [ALS_MF](src/ALS_MF.py) an inheritance example, using ALS method
    * kNN model is still on progress
   
   Note that every new model should override at least two functions of the base class, which are [fit](./src/BaseModel.py) and [recommend_for_all_users](./src/BaseModel.py).
* [Evaluation module](src/Evaluator.py), only two ranking metrics are supported now (i.e., NDCG and Precision)
* [Utilities](src/utils.py) like constructing a spark session

## Local testing and debug
Say if you want to debug your own model using existing methods and functions, follow the following step:
1. Create a directory called "test" under the root directory and after this step, the repo should look like this:
    ```bash
    .
    ├── data
    ├── doc
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