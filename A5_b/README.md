# Assignment A5-b: Scoring documents using LM/MLM

## Task

Implement **LM** retrieval model and its fielded variant **MLM**.

In the public tests, the implementations will be tested using "dummy" (toy-sized) collections while for the private tests we will be using a real data collection.

## Specific steps

For this assignment, only packages that are part of the Anaconda Python 3.11+ distribution are allowed.

### Retrieval models

Each of the two retrieval models inherits from the abstract `Scorer` class. All the subclasses inherit the method `score_collection`, which applies `score_term` in a term-at-a-time manner, without overwriting it. Implement each of the retrieval models by implementing model's scoring function in the corresponding `score_term` method.

#### LM Scoring

Implement the language model approach for ranking documents. Specifically, we score document based on query log likelihood:

$$score(d,q) = \sum_{t \in q} c_{t,q} \times \log P(t|\theta_d) = \sum_{t \in q} c_{t,q} \times \log \Big ( (1-\lambda) \frac{c_{t,d}}{|d|} + \lambda P(t|C) \Big )$$

The collection (background) term probability $P(t|C)$ is a maximum likelihood estimate, which is simply the term's relative frequency across the entire collection. (Hint: mind that we operate on a single field here, so only that field's content is considered as the collection.)

#### MLM scoring

Using multiple document fields, the document language model is taken to be a linear combination of the (smoothed) field language models:

$P(t|\theta_d) = \sum_i w_i P(t|\theta_{d_i})$ ,

where $w_i$ is the field weight for field $i$ (and $\sum_i w_i = 1$).

The field language models $P(t|\theta_{d_i})$ are computed as follows (we use Jelinek-Mercer smoothing):

$P(t|\theta_{d_i}) = (1-\lambda_i) P(t|d_i) + \lambda_i P(t|C_i)$,

where

  * $\lambda_i$ is a field-specific smoothing parameter
  * $P(t|d_i) = \frac{f_{t,d_i}}{|d_i|}$ is the empirical field language model (term's relative frequency in the document field). $f_{t,d_i}$ is the raw frequency of $t$ in field $i$ of $d$. $|d_i|$ is the length (number of terms) in field $i$ of $d$.
  * $P(t|C_i) = \frac{\sum_{d'}f_{t,d'_i}}{\sum_{d'}|d'_i|}$ is the collecting field language model (term's relative frequency in that field across the entire collection)
  
(Hint: You'll need to substitute $P(t|\theta_d)$ back into the log query likelihood formula shown under the LM scoring section.)
