PREFIX = "##"
UNKNOWN = "<unk>"


def initialize_vocabulary(word_corpus: list[str]) -> set[str]:
    """Initializes the vocabulary with characters present in the corpus.

    Args:
        word_corpus: Corpus of words.

    Returns:
        Initial vocabulary.
    """
    # TODO
    ...


def tokenize(word: str, vocabulary: set[str]) -> list[str]:
    """Tokenizes a word using the vocabulary.

    The tokenizer splits the word using the longest possible tokens in the
    vocabulary. For example, if the word is "surfing", and the vocabulary
    contains the tokens "sur", "surf", and "ing", then the tokenizer will
    return ["surf", "##ing"].
    Returns <unk> token if the word cannot be fully tokenized.

    Args:
        word: Word to tokenize.
        vocabulary: Vocabulary.

    Returns:
        List of tokens.
    """
    word = word.lower()
    tokens = []
    while word.lstrip(PREFIX):
        for i in range(len(word)):
            if word[: len(word) - i] in vocabulary:
                tokens.append(word[: len(word) - i])
                word = PREFIX + word[len(word) - i :]
                break
        else:
            return [UNKNOWN]
    return tokens


def score(
    pair_freq: int, subword_token1_freq: int, subword_token2_freq: int
) -> float:
    """Calculates the score for merging two subword tokens.

    Args:
        pair_freq: Frequency of the pair.
        subword_token1_freq: Frequency of the first subword token.
        subword_token2_freq: Frequency of the second subword token.

    Returns:
        Score.
    """
    return pair_freq / (subword_token1_freq * subword_token2_freq)


def get_new_subword_token(
    data: list[tuple[list[str], int]], vocabulary: set[str]
) -> tuple[str, float]:
    """Finds the new subword token to add to the vocabulary.

    The new subword token is the pair of tokens that maximizes the score. In
    case of ties, the pair that appears first in the vocabulary is chosen.

    Args:
        data: List of tokenized words and their frequencies.
        vocabulary: Vocabulary.

    Returns:
        New subword token and its score.
    """
    # TODO
    ...


def train(
    word_corpus: list[tuple[str, int]],
    vocabulary: set[str],
    num_iterations: int | None = 4,
    max_vocab_size: int | None = None,
) -> set[str]:
    """Executes the WordPiece training algorithm.

    The algorithm iteratively merges subword tokens to create new ones. It stops
    when the number of iterations is reached or when the vocabulary reaches
    the maximum size.

    Args:
        word_corpus: Corpus of words and their frequencies.
        vocabulary: Vocabulary.
        num_iterations: Number of iterations to train the vocabulary. Defaults
            to 4.
        max_vocab_size: Maximum size of the vocabulary. Defaults to None.

    Returns:
        Vocabulary.
    """
    # TODO
    ...


def tokenize_corpus(
    corpus: list[tuple[str, int]], vocabulary: set[str]
) -> list[tuple[list[str], int]]:
    """Tokenizes the corpus using the vocabulary.

    Args:
        corpus: Corpus of words and their frequencies.
        vocabulary: Vocabulary.

    Returns:
        List of tokenized words and their frequencies.
    """
    # TODO
    ...
