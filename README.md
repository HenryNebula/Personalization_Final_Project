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

## Structure of system

* Data pipeline (load/sample/split)
* Model implementation (unified API)
* Inference and evaluation module (HR@K/Precision@K/Recall@5/NDCG@5)
* Parameter tuning system and logger
* Result interpretation (graph+table)
* Re-iterate training method
