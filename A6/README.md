# Assignment A6: Evaluating rankings

## Task

Implement specific measures for evaluating rankings.

## Specific steps

For this assignment, you will need to install an additional package, `ir_datasets`, provided in the requirements.txt file. For implementing evaluation functions in this task, only packages that are part of the Anaconda Python 3.11+ distribution are allowed.

### Data loading

Data loading functions need to be implemented for both system ranking and ground truth. For system ranking, implement `load_rankings()`, which takes a path to rankings file and returns a dictionary with query IDs as keys and a list of documents for each query as values. For ground truth, implement `load_ground_truth()` to return a dictionary with query IDs as keys and a set of documents for each query as values.

**NB! The docstrings in the functions contain more information about the data loading structure.**

### Evaluation

Given system rankings as a list and ground truth as a set of documents implement three binary evaluation measures, Precision@k, Average Precision, and Reciprocal Rank. Additionally, implement a method that computes the mean evaluation measure (for any of the above) over a set of queries. This method should take dictionaries for system rankings and ground truth where keys are query IDs and values are a list (for system rankings) and a set (for ground truth) of documents.
