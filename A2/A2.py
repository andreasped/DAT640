def get_confusion_matrix(
    actual: list[int], predicted: list[int]
) -> list[list[int]]:
    """Computes confusion matrix from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        List of two lists of length 2 each, representing the confusion matrix.
    """
    n = min(len(actual), len(predicted))
    actual = actual[:n]
    predicted = predicted[:n]

    tp = tn = fp = fn = 0

    for a, p in zip(actual, predicted):
        if a not in (0, 1) or p not in (0, 1):
            continue
        if a == 1 and p == 1:
            tp += 1
        elif a == 0 and p == 0:
            tn += 1
        elif a == 0 and p == 1:
            fp += 1
        elif a == 1 and p == 0:
            fn += 1

    return [[tn, fp], [fn, tp]]


def accuracy(actual: list[int], predicted: list[int]) -> float:
    """Computes the accuracy from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Accuracy as a float.
    """
    [[tn, fp], [fn, tp]] = get_confusion_matrix(actual, predicted)
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return acc


def precision(actual: list[int], predicted: list[int]) -> float:
    """Computes the precision from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Precision as a float.
    """
    [[_, fp], [_, tp]] = get_confusion_matrix(actual, predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precision


def recall(actual: list[int], predicted: list[int]) -> float:
    """Computes the recall from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Recall as a float.
    """
    [[_, _], [fn, tp]] = get_confusion_matrix(actual, predicted)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall


def f1(actual: list[int], predicted: list[int]) -> float:
    """Computes the F1-score from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of harmonic mean of precision and recall.
    """
    p = precision(actual, predicted)
    r = recall(actual, predicted)
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return f1


def false_positive_rate(actual: list[int], predicted: list[int]) -> float:
    """Computes the false positive rate from lists of actual or predicted
        labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of number of instances incorrectly classified as positive divided
            by number of actually negative instances.
    """
    [[tn, fp], [_, _]] = get_confusion_matrix(actual, predicted)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return fpr


def false_negative_rate(actual: list[int], predicted: list[int]) -> float:
    """Computes the false negative rate from lists of actual or predicted
        labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of number of instances incorrectly classified as negative divided
            by number of actually positive instances.
    """

    [[_, _], [fn, tp]] = get_confusion_matrix(actual, predicted)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return fnr
