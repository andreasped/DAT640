# Assignment A5-a: Scoring documents using BM25/BM25F

## Task

Implement a term-at-a-time scoring strategy to score all documents in the collection with respect to an input query. In addition to a simple scoring function, implement retrieval model **BM25**, as well as its fielded variant **BM25F**.

In the public tests, the implementations will be tested using "dummy" (toy-sized) collections while for the private tests we will be using a real data collection.

## Specific steps

For this assignment, only packages that are part of the Anaconda Python 3.11+ distribution are allowed.

### Retrieval models

Each of the three retrieval models inherits from the abstract `Scorer` class. All the subclasses inherit the method `score_collection`, which applies `score_term` in a term-at-a-time manner, without overwriting it. Implement each of the retrieval models by implementing model's scoring function in the corresponding `score_term` method.

#### Simple scoring

For scoring documents, employ the following simple retrieval function:

$$score(d,q) = \sum_{t \in q} w_{t,d} \times w_{t,q}$$

Where $w_{t,d}$ should simply be the number of occurrences of term $t$ in the document. Similarly, $w_{t,q}$ is set to number of times term $t$ appears in the query.

#### BM25 scoring

$$score(d,q) = \sum_{t \in q} \frac{c_{t,d}\times (1+k_1)}{c_{t,d} + k_1(1-b+b\frac{|d|}{avgdl})} \times idf_t$$

IDF is to be computed as $idf_t=\text{log}(\frac{N}{n_t})$, where $N$ is the total number of documents in the collection and $n_t$ is the number of documents containing term $t$. Note that $\log$ is the natural logarithm (i.e., `math.log()`).

#### BM25F scoring

$$score(d,q) = \sum_{t \in q} \frac{\tilde{c}_{t,d}}{k_1 + \tilde{c}_{t,d}} \times idf_t$$

$$\tilde{c}_{t,d} = \sum_i w_i \times \frac{c_{t,d_i}}{B_i}$$

where

  * $i$ corresponds to the field index
  * $w_i$ is the field weight
  * $B_i$ is soft normalization for field $i$
  
$$B_i = (1-b_i+b_i\frac{|d_i|}{avgdl_i})$$

IDF values should be computed based on the body field using natural-base logarithm.
