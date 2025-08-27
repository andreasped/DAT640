from numpy import ndarray
import re
import unicodedata
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


def load_data(path: str) -> tuple[list[str], list[int]]:
    """Loads data from file. Each except first (header) is a datapoint
    containing ID, Label, Email (content) separated by "\t". Lables should be
    changed into integers with 1 for "spam" and 0 for "ham".

    Args:
        path: Path to file from which to load data

    Returns:
        List of email contents and a list of lobels coresponding to each email.
    """
    emails = []
    labels = []
    with open(path, "r", encoding="utf-8") as file:
        next(file, None)

        for line in file:
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            _, label, email = parts
            label = label.strip().lower()
            if label not in ("spam", "ham"):
                raise ValueError(f"Unknown label {label}")
            labels.append(1 if label == "spam" else 0)

            # Remove enclosing quotes if present (this is a fix for something that gave an error in the test)
            if email.startswith('"') and email.endswith('"'):
                email = email[1:-1]
            emails.append(email)
    return emails, labels


def preprocess(doc: str) -> str:
    """Preprocesses text to prepare it for feature extraction.

    Args:
        doc: String comprising the unprocessed contents of some email file.

    Returns:
        String comprising the corresponding preprocessed text.
    """
    if not doc:
        return ""

    doc = unicodedata.normalize("NFKD", doc)
    doc = doc.lower()
    doc = re.sub(r"[^a-z0-9\s]", " ", doc)
    tokens = doc.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 2]

    return " ".join(tokens)


def preprocess_multiple(docs: list[str]) -> list[str]:
    """Preprocesses multiple texts to prepare them for feature extraction.

    Args:
        docs: List of strings, each consisting of the unprocessed contents
            of some email file.

    Returns:
        List of strings, each comprising the corresponding preprocessed
            text.
    """
    return [preprocess(doc) for doc in docs]


def extract_features(
    train_dataset: list[str], test_dataset: list[str]
) -> tuple[ndarray, ndarray] | tuple[list[float], list[float]]:
    """Extracts feature vectors from a preprocessed train and test datasets.

    Args:
        train_dataset: List of strings, each consisting of the preprocessed
            email content.
        test_dataset: List of strings, each consisting of the preprocessed
            email content.

    Returns:
        A tuple of of two lists. The lists contain extracted features for
          training and testing dataset respectively.
    """
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))

    # Fit on training data and transform both train and test datasets
    X_train = vectorizer.fit_transform(train_dataset).toarray()
    X_test = vectorizer.transform(test_dataset).toarray()

    return X_train, X_test


def train(X: ndarray, y: list[int]) -> object:
    """Trains a classifier on extracted feature vectors.

    Args:
        X: Numerical array-like object (2D) representing the instances.
        y: Numerical array-like object (1D) representing the labels.

    Returns:
        A trained model object capable of predicting over unseen sets of
            instances.
    """
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model


def evaluate(y: list[int], y_pred: list[int]) -> tuple[float, float, float, float]:
    """Evaluates a model's predictive performance with respect to a labeled
    dataset.

    Args:
        y: Numerical array-like object (1D) representing the true labels.
        y_pred: Numerical array-like object (1D) representing the predicted
            labels.

    Returns:
        A tuple of four values: recall, precision, F_1, and accuracy.
    """
    tp = sum(1 for true, pred in zip(y, y_pred) if true == pred == 1)
    tn = sum(1 for true, pred in zip(y, y_pred) if true == pred == 0)
    fp = sum(1 for true, pred in zip(y, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y, y_pred) if true == 1 and pred == 0)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return recall, precision, f1, accuracy


if __name__ == "__main__":
    print("Loading data...")
    train_data_raw, train_labels = load_data("data/train.tsv")
    test_data_raw, test_labels = load_data("data/test.tsv")

    print("Processing data...")
    train_data = preprocess_multiple(train_data_raw)
    test_data = preprocess_multiple(test_data_raw)

    print("Extracting features...")
    train_feature_vectors, test_feature_vectors = extract_features(
        train_data, test_data
    )

    print("Training...")
    classifier = train(train_feature_vectors, train_labels)

    print("Applying model on test data...")
    predicted_labels = classifier.predict(test_feature_vectors)

    print("Evaluating...")
    recall, precision, f1, accuracy = evaluate(test_labels, predicted_labels)

    print(f"Recall:\t{recall}")
    print(f"Precision:\t{precision}")
    print(f"F1:\t{f1}")
    print(f"Accuracy:\t{accuracy}")
