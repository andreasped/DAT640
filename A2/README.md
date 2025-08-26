# Assignment A2: Binary text classification evaluation

## Task

Implement functions for metrics that evaluate the performance of a binary text classifier.

## Background

In a binary classification task, the confusion matrix is a table that helps visualize the performance of a classification model by showing the counts of correct and incorrect predictions for each class. It is structured as follows:

<center>

|||
|--|--|
| TN | FP |
| FN | TP |

</center>

The rows represent the actual classes (what the true labels are), while the columns represent the predicted classes (what the model predicted).
TN, FP, FN, and TP are the numbers of true negatives, false positives, false negatives, and true positives, respectively.
These are defined as:

  * **True Positives (TP)**: the model correctly predicts the positive class.
  * **False Positives (FP)**: the model incorrectly predicts the positive class for a negative instance.
  * **False Negatives (FN)**: the model incorrectly predicts the negative class for a positive instance.
  * **True Negatives (TN)**: the model correctly predicts the negative class.

The confusion matrix provides the necessary components for computing key performance measures:

  * **Accuracy**: Number of correctly classified instances out of all instances.
    $$Acc = \frac{TP+TN}{TP+TN+FP+FN}$$
  * **Precision**: Number of instances correctly identified as positive out of the total instances identified as positive.
    $$P = \frac{TP}{TP+FP}$$
  * **Recall**: Number of instances correctly identified as positive out of the total number of actual positives.
    $$R = \frac{TP}{TP+FN}$$
  * **F1-score**: The harmonic mean of precision and recall.
    $$F1 = \frac{2 \cdot P \cdot R}{P+R}$$
  * **False positive rate**: Number of instances incorrectly identified as positive out of the total number of actual negatives.
    $$FPR = \frac{FP}{FP+TN}$$
  * **False negative rate**: Number of instances incorrectly identified as negative out of the total number of actual positives.
    $$FNR = \frac{FN}{FN+TP}$$

You may also review these [slides on text classification](https://stavanger.instructure.com/files/1759439/download?download_frd=1) for additional details and examples.

## Specific steps

For this assignment, you are not allowed to import any packages and must implement the functions from scratch.

Implement evaluation functions that will be used to calculate performance metrics when comparing some predicted classes with the corresponding actual classes in a binary classification setting.

First, implement `get_confusion_matrix()`, which takes the lists `actual` and `predicted`, as input and returns a 2x2 matrix as a list of lists:
`[[tn, fp], [fn, tp]]`

Then, implement the computation of Accuracy, Precision, Recall, F1-score, False positive rate, and False Negative rate. For each measure, implement a function that takes the lists `actual` and `predicted`, and returns a decimal number according to the measure definition. Use the function `get_confusion_matrix()` to access the true/false positives/negatives needed.
