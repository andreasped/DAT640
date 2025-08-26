# Assignment A3: Training and evaluating a spam classifier

## Task

Build a spam classifier that is able to correctly classify the incoming emails to company email accounts as either spam or genuine non-spam ("ham") emails. You will need to implement:

  - Data loading,
  - Text preprocessing,
  - Feature extraction,
  - Training a classifier, and
  - Evaluating predictions made by a trained classifier.

## Background

This exercise assumes that you have some prior experience with using machine learning models for classification tasks. The focus of this assignment is on applying existing classification algorithms to textual data. If you need a refresher or lack some of the necessary background knowledge, don't worry—we've provided resources to help you get up to speed. You can review the relevant theory in these [slides on text classification](https://stavanger.instructure.com/files/1759439/download?download_frd=1). Additionally, we recommend the scikit-learn tutorial on working with text data, which discusses the task of text classification in detail: <https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html>.

## Specific steps

For this assignment, only packages that are part of the Anaconda Python 3.11+ distribution are allowed.
However, in this assignment, you are not expected to implement any particular solution from scratch, but utilize existing libraries, where possible. We strongly recommend using `scikit-learn`, but you are free to choose any other library.

Data used in the assignment should be downloaded under a `data` folder and the paths should not be changed.

### Data

For your convenience, we have prepared a ham and spam dataset that you can train and evaluate your classifier on. It's important to note that this is the same dataset and partitioning (i.e. the dataset has been partitioned into `train` and `test` splits).

Example structure of both datasets:

```
Id      Label   Email
train/000/000   ham     "Received: from NAHOU-MSMBX01V ([192.168.110.39]) by ...
train/000/002   ham     "Received: from NAHOU-MSMBX01V ([192.168.110.39]) by ...
```

For the splits, the labels (`ham` or `spam`) are provided to you. The Id, Lablel, and Email in the files are separated by *tab* (`\t` in Python).

Your submission will be graded primarily on tests in `test_public.py`, but there will also be a hidden test that will be applied during grading, where your classifier gets applied to the `hidden` split and the performance that your model achieves will be graded.

### Load data

In order to train or evaluate the spam classifier, you must load the data from file.
Implement the function `load_file` which takes a file path as input and returns a tuple of two lists

  - list of file's contents as a string and
  - list of labels converted to integers; For `spam`, use 1, and for `ham` use a 0.

### Text preprocessing

The text data should be preprocessed in order to improve the quality of the feature extraction. Implement the function that does this for you. You have wide discretion in selecting which text preprocessing gives the best result when training and evaluating your classifier later.

You need to implement the function `preprocess` which takes a string---such as that returned by `load_file`---as input and returns another string comprising preprocessed text.

At a minimum, will need to implement punctuation removal and stopword removal. For example:

```
id = "train/000/000"
label = "ham" 
email = "Received: from NAHOU-MSMBX01V ([192.168.110.39]) by ... AcEAUaYbkE2KMWxCEdWxEABQi+MJ2QATr4SA From: ""Hu, Sylvia"" ..."
```

After preprocessing, the email content should be:

```
id = "train/000/000"
label = "ham" 
email = "received nahou msmbx01v 192 168 110 39 aceauaybke2kmwxcedwxeabqi mj2qatr4sa hu sylvia"
```

### Extract features

The preprocessed text can then be turned into feature vectors in some way, the final step before applying a machine learning algorithm to the training data.

You must implement the function `extract_features`, which takes two list of strings, a train split and a validation/test split, and returns two numerical array-like objects representing the two data splits.

### Train classifier

Implement the function that takes data produced by your feature extraction to train a classifier.

You must implement the function `train`, which takes two numerical array-like objects (representing the instances and their labels) as input and returns a trained model object. The model object must have a `predict` method that takes as input a numerical array-like object (representing instances) and returns a numerical array-like object (representing predicted labels).

### Evaluate trained classifier

Implement the function `evaluate`, which takes as input a trained model, as well as a labeled dataset, and returns the following evaluated performance metrics: recall, precision, F1, and accuracy.

## NB

For the hidden test, points are awarded proportionally to the performance of the classifier with the maximum score achieved at 99% accuracy. This should be achievable with simple preprocessing and a good choice of feature extraction and classifier.
