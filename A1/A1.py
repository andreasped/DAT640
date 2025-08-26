import collections


def get_word_frequencies(doc: str) -> dict[str, int]:
    """Extracts word frequencies from a document.

    Args:
        doc: Document content given as a string.

    Returns:
        Dictionary with words as keys and their frequencies as values.
    """
    for i in [',', '.', ':', ';', '?', '!']:
        doc = doc.replace(i, ' ')
    tokens = doc.split()
    return dict(collections.Counter(tokens))


def get_word_feature_vector(
    word_frequencies: dict[str, int], vocabulary: list[str]
) -> list[int]:
    """Creates a feature vector for a document, comprising word frequencies
        over a vocabulary.

    Args:
        word_frequencies: Dictionary with words as keys and frequencies as
            values.
        vocabulary: List of words.

    Returns:
        List of length `len(vocabulary)` with respective frequencies as values.
    """
    return [word_frequencies.get(word, 0) for word in vocabulary]
