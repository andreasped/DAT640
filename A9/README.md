# Assignment A9: Pseudo-relevance feedback

## Task

Implement a pseudo-relevance feedback algorithm based on the RM3 method. The RM3 algorithm enhances the original query by incorporating terms from the top-ranked documents, thus improving the retrieval performance.

## Background

Pseudo-relevance feedback (PRF) enhances information retrieval by using the top-ranked documents from an initial query to refine and expand the query, thus improving subsequent retrieval results. The RM3 algorithm, a popular PRF method, combines the original query terms with high-frequency terms from the top documents to create a more effective query. This technique helps capture additional relevant information and mitigates the limitations of the original query's vocabulary.

Given an initial query $q$ and a set of documents $D$, the RM3 algorithm performs the following steps:

1. **Rank documents**: Use a ranking function (e.g., BM25 from `rank_bm25`) to rank documents based on the initial query.
2. **Select top documents**: Select the top $k$ documents from the ranked list.
3. **Extract feedback terms**: Compute the frequencies of the terms from the top $k$ documents to identify potential feedback terms $t$. Select top $m$ most frequent terms as potential query expansion terms $T$.
4. **Compute term weights**:
   1. For the $m$ most frequent feedback terms from top $k$ documents:
     $$P(t | T) = \frac{tf_t^{D_k}}{\sum_{t' \in T} tf_{t'}^{T}}$$
     where $tf_t^{D_k}$ is the term frequency of term $t$ in top $k$ documents, and $tf_{t'}^{T}$ is a term frequency of the term $t'$ from the $m$ most frequent feedback terms selected for query expansion.

   2. For the original query terms:
     $$P(t | q) = \frac{tf_w^q}{|q|}$$
     where $tf_t^q$ is the term frequency of term $t$ in the query $q$, and $|q| $ is the length of the query.
5. **Create new (expanded) query by interpolating the original and expansion query terms**:
   $$P(t) = \lambda P(t | q) + (1 - \lambda) P(t | T)$$
   where $\lambda $ is a parameter controlling the influence of the original query terms and the feedback terms.

## Specific steps

For this assignment, you will need to install additional packages, `rank_bm25` and `nltk`, provided in the requirements.txt file.

For developing your solution, you are provided with a document collection (`documents.jsonl`) and a set of queries (`queries`). All the functions for loading and pre-processing are provided. More details about the provided data are in the docstrings.

Data used in the assignment should be downloaded under a `data` folder and the paths should not be changed.

Your task is to implement the following functions:

  - `get_top_collection_terms()` that performs steps  3 and 4.1 of the above algorithm.
  - `interpolate_terms()` that performs step 5 of the above algorithm, and
  - provide the missing implementation of step 4.2 in `get_expanded_query()`

Note: You may adjust the parameters `prf_num_documents`, `prf_num_terms`, and `_LAMBDA` to see how they affect retrieval performance.
