from collections import Counter


PREFIX = "##"
UNKNOWN = "<unk>"


def initialize_vocabulary(word_corpus: list[str]) -> set[str]:
    """Initializes the vocabulary with characters present in the corpus.

    Args:
        word_corpus: Corpus of words.

    Returns:
        Initial vocabulary.
    """
    vocab = set()
    for word in word_corpus:
        if not isinstance(word, str):
            raise ValueError("All elements in word_corpus must be string")
        word = word.lower().strip()
        if not word:
            continue
        vocab.add(word[0])  # Add first word without prefix
        for ch in word[1:]:
            if not ch.isalnum():  # Only alphanumeric characters
                continue
            vocab.add(PREFIX + ch)  # Subsequent characters with ##
    if not vocab:
        raise ValueError("Vocabulary cannot be initialized from an empty or invalid corpus")
    return vocab


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
    token_freq = Counter()
    pair_freq = Counter()

    for tokens, freq in data:
        if not tokens or freq <= 0:
            continue
        for t in tokens:
            token_freq[t] += freq
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_freq[pair] += freq

    best_pair = None
    best_score = float("-inf")

    for (a, b), pfreq in pair_freq.items():
        if pfreq <= 0 or token_freq[a] <= 0 or token_freq[b] <= 0:
            continue
        try:
            s = score(pfreq, token_freq[a], token_freq[b])
        except ZeroDivisionError:
            continue
        if s > best_score:
            best_score = s
            best_pair = (a, b)

    if not best_pair:
        return ("", 0.0)

    # Merge into new subword token
    new_token = best_pair[0] + best_pair[1].lstrip(PREFIX)
    return new_token, best_score


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
    for _ in range(num_iterations):
        if max_vocab_size and len(vocabulary) >= max_vocab_size:
            break

        try:
            data = tokenize_corpus(word_corpus, vocabulary)
        except Exception as e:
            raise ValueError(f"Error during tokenization: {e}")

        # Find best new token
        new_token, _ = get_new_subword_token(data, vocabulary)
        if not new_token:
            break

        if max_vocab_size and len(vocabulary) + 1 > max_vocab_size:
            break

        vocabulary.add(new_token)

    if not vocabulary:
        raise ValueError("Vocabulary is empty after training")

    return vocabulary


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
    tokenized = []
    for word, freq in corpus:
        if not isinstance(word, str) or not isinstance(freq, int) or freq <= 0:
            continue
        try:
            tokens = tokenize(word, vocabulary)
            tokenized.append((tokens, freq))
        except Exception as e:
            print(f"Error tokenizing word '{word}': {e}")
            continue

    if not tokenized:
        raise ValueError("Tokenized corpus is empty")
    return tokenized
