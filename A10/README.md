# Assignment A10: Subword tokenization

## Task

Implement the bottom-up version of the WordPiece subword tokenization algorithm.  

### Background

The WordPiece algorithm is a subword tokenization algorithm. It has two versions: top-down and bottom-up. In this assignment, you will implement the bottom-up version of the algorithm.

The algorithm is as follows ([Schuster and Nakajima, 2012](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)):

1. Initialize the vocabulary with all the characters in the training set.
2. Build a language model on the training set.
3. Generate a new subword token by merging two subword tokens that maximize the likelihood of the training set.
4. Repeat from step 2 until the vocabulary size reaches the maximum size or the maximum number of iterations is reached.

## Specific steps

For this assignment, only packages that are part of the Anaconda Python 3.11+ distribution are allowed.

### Vocabulary initialization

You need to implement the function `initialize_vocabulary`. This function takes a list of words and returns a vocabulary. The vocabulary should be initialized with all the characters in the training set. Note that a prefix `##` should be added to tokens that are within a word. For example, for the word `hello`, the initial vocabulary should contain the following tokens: `["h", "##e", "##l", "##l", "##o"]`.

### Training

You need to implement the function `train`. This function takes a list of words and a vocabulary and returns a vocabulary. The function should build a language model on the training set and generate a new subword token by merging two subword tokens that maximize the likelihood of the training set. The function should repeat the process until the vocabulary size reaches the maximum size or the maximum number of iterations is reached.
The function `get_new_subword_token` should be used to find the new subword token. The algorithm computes a score for each pair of subword tokens and returns the pair with the highest score. The score is computed as follows:

$$score(a,b) = { c_{ab} \over c_a * c_b} $$

where $c_x$ represents the frequency of the subword token $x$.
