---
tags:
  - nlp
  - approach
aliases:
  - TF-IDF
---
Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic used in NLP to show how important a word (term) is to a document in a corpus. It increases proportionally to the number of times a word appears in the document and is offset by the frequency of the word in the corpus.

### Term Frequency (TF)
Measures how frequently a term appears within a specific document. It assumes that words appearing more often in a document are more important to that document. There are multiple ways to calculate it:

*   **Raw Count:** Simplest form. $TF(t, d) = f(t, d)$, where $f(t, d)$ is the raw count of term in document.
*   **Boolean Frequency:** $TF(t, d) = 1$ if *t* is present in *d*, and $0$ otherwise.
*   **Logarithmic Scaling:** $TF(t, d) = log(1 + f_{t,d})$.
*   **Augmented Frequency:** Normalizes the raw frequency by the frequency of the most frequent term in the document to prevent bias towards longer documents: $TF(t, d) = 0.5 + 0.5 \cdot \frac{f_{t,d}}{\max\{f_{t',d}: t' \in d\}}$
* 
### Inverse Document Frequency (IDF)
Measures how much information a term provides across the entire corpus. It estimates the "informativeness" or "rarity" of a term.

The standard formula is:
$IDF(t, D) = log(N / df(t))$

Where:
*   $N$: Total number of documents in the corpus $D$. $N = |D|$.
*   $df(t)$: Document Frequency of term. The number of documents in the corpus that contain the term.

**Smoothing:** To avoid division by zero (if $df(t) = 0$) and to prevent the IDF score from becoming zero for terms appearing in all documents ($df(t) = N$), smoothing is typically applied. A common variant is:
$IDF(t, D) = log(N / (df(t) + 1)) + 1$
Scikit-learn by default adds 1 to the numerator:
$IDF(t, D) = log((N + 1) / (df(t) + 1)) + 1$

## TF-IDF Calculation
$TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)$
## Vectorization using TF-IDF
TF-IDF is used to transform a collection of text documents into a numerical feature matrix (document-term matrix). The steps are the following
1.  **Vocabulary Creation:** Identify all unique terms across the entire corpus.
2.  **Matrix Construction:** Create a matrix where rows represent documents and columns represent terms from the vocabulary.
3.  **Filling the Matrix:** The value in cell $(i, j)$ is the TF-IDF score of the `j`-th term in the `i`-th document.
4.  **Output:** This results in a sparse matrix, where most entries are zero because a single document usually contains only a small subset of the overall vocabulary.
## Links

*   [Wikipedia: tf–idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
*   [Scikit-learn: TfidfVectorizer Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
