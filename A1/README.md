# Assignment A1: Document feature extraction

## Task

Implement functions that perform simplistic feature extraction for a text classifier. Text classification is the task of categorizing text into predefined classes, such as determining whether an email is spam or not (binary classification) or classifying emails into categories like work, personal, and promotions (multi-class classification).

## Specific steps

For this assignment, you are only allowed to import specialized container datatypes (i.e., the `collections` package).

Implement two functions that will be used to extract feature vectors from raw text documents.

### Extract feature vectors from documents

First, implement `get_word_frequencies()`, which takes a string `doc` as input and returns a Python dictionary (`dict`) where the tokenized words in the `doc` string are the keys, and the number of times that word occurs in `doc` is the corresponding value. You need to split words based on whitespaces and punctuation marks (specifically: `,`, `.`, `:`, `;`, `?`, `!`).

### Create feature vectors

Second, implement `get_word_feature_vector()`, which takes `word_frequencies` and `vocabulary` as input. `vocabulary` is a list of tokens in a specific order, representing the vocabulary over the entire corpus. Return a list with the same length as `vocabulary`, with the frequencies of each word in the specified order, and 0 in all other elements.

Out-of-vocabulary terms should be ignored.
