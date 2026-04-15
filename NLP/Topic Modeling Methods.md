---
tags:
  - nlp
  - unsupervised-learning
  - algorithm
aliases:
  - Topic Modeling Algorithms
---

A survey of the main topic modeling methods, ordered roughly by historical development (matrix factorisation → probabilistic generative models → neural → embedding-based). For the broader context — when to use topic modeling at all, how to evaluate results, and how to handle production concerns — see [[Topic Modeling]].

## Classical Methods

### Latent Semantic Analysis (LSA / LSI)

[Deerwester et al., 1990](https://doi.org/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9). The earliest approach, based on [[Dimensionality Reduction|dimensionality reduction]] via Singular Value Decomposition (SVD).

Given a [[Term Frequency-Inverse Document Frequency|TF-IDF]] weighted document-term matrix $X$:

$$X_{m \times n} = U_{m \times K} \, \Sigma_{K \times K} \, V^{T}_{K \times n}$$

where $U$ captures term-topic relationships, $\Sigma$ contains singular values, $V$ captures document-topic relationships, and $K$ is the number of retained dimensions.

Truncating to the top $K$ singular values forces synonyms to cluster together and partially resolves polysemy by projecting terms and documents into a shared latent space.

> [!info] Polysemy
> Polysemy is when a single word carries multiple meanings (*bank* = river bank / financial institution; *apple* = fruit / company). Bag-of-words methods treat every occurrence of a word as the same token, so they cannot distinguish these senses from context. Embedding-based methods handle polysemy naturally because contextual encoders produce different vectors for the same word in different contexts.

**Advantages:** deterministic, fast, captures co-occurrence patterns.
**Disadvantages:** topic dimensions can have negative weights (hard to interpret as probabilities), no probabilistic foundation, requires specifying $K$.

### Probabilistic LSA (pLSA)

[Hofmann, 1999](https://arxiv.org/abs/1301.6705). Adds a probabilistic interpretation to LSA:

$$P(w \mid d) = \sum_{k=1}^{K} P(w \mid z_k) \, P(z_k \mid d)$$

where $z_k$ is a latent topic. Trained via Expectation-Maximization. pLSA bridges LSA and LDA historically — it introduced documents-as-topic-mixtures, but lacks priors on the document-topic distributions, so the number of parameters grows linearly with corpus size and the model cannot assign probabilities to unseen documents.

### Latent Dirichlet Allocation (LDA)

[Blei, Ng, Jordan, 2003](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf). The canonical probabilistic topic model. Fixes pLSA's overfitting by adding Dirichlet priors to both document-topic and topic-word distributions. Each document's topic distribution $\theta_d$ and each topic's word distribution $\phi_k$ are inferred from observed words via Gibbs sampling or variational inference.

See [[LDA]] for the full generative process, inference methods, hyperparameters, folding-in for inductive inference, and guided variants.

### Non-negative Matrix Factorization (NMF)

[Lee & Seung, 1999](https://www.nature.com/articles/44565). Factorises a document-term matrix $V$ into two non-negative matrices:

$$V_{m \times n} \approx W_{m \times K} \, H_{K \times n}, \qquad \min_{W, H \geq 0} \|V - WH\|_F^2$$

where $W$ captures document-topic weights and $H$ captures topic-word weights. The non-negativity constraint produces naturally sparse, additive, parts-based representations.

NMF typically uses [[Term Frequency-Inverse Document Frequency|TF-IDF]] input (unlike LDA, which needs integer counts for its multinomial generative assumption). TF-IDF down-weights generic high-frequency words, which tends to produce cleaner topics.

Scikit-learn solver tip: use `solver='cd'` (coordinate descent) with the default Frobenius loss for speed. Switch to `solver='mu'` (multiplicative update) if optimising for Kullback-Leibler divergence (`beta_loss='kullback-leibler'`), which produces more LDA-like probabilistic topics.

**Advantages:** fast, deterministic given initialisation, clean sparse topics, works well with TF-IDF.
**Disadvantages:** requires specifying $K$, no probabilistic interpretation, results depend on initialisation.

NMF is often an underrated baseline. If LDA disappoints, try NMF with TF-IDF before reaching for anything fancier.

## Neural Topic Models

### ProdLDA

[Srivastava & Sutton, 2017](https://arxiv.org/abs/1703.01488). The first effective neural variational inference approach for LDA. Uses a VAE architecture: an encoder maps bag-of-words input to topic proportions, a decoder reconstructs the document.

The key innovation is approximating the Dirichlet prior with a logistic normal distribution, avoiding the "component collapsing" problem where earlier neural approaches produced degenerate topics. Trained by maximising the ELBO. Much faster inference than Gibbs sampling, and handles new documents without retraining.

### Embedded Topic Model (ETM)

[Dieng, Ruiz, Blei, 2020](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00325/96463). Bridges topic modeling with [[Word Embeddings|word embeddings]]. Each word and each topic live in the same embedding space; the probability of word $w$ under topic $k$ is:

$$P(w \mid k) \propto \exp(\mathbf{e}_w^\top \mathbf{t}_k)$$

where $\mathbf{e}_w$ is the word embedding and $\mathbf{t}_k$ is the topic embedding. Can use pre-trained embeddings ([[Word2Vec]], [[GloVe]]) or learn them jointly. Handles large vocabularies and rare words better than standard LDA because semantically similar words share embedding structure.

### Contextualised Topic Models (CTM / CombinedTM)

[Bianchi et al., 2021](https://aclanthology.org/2021.eacl-main.143/). Extends ProdLDA by feeding sentence-transformer embeddings into the encoder alongside (or instead of) bag-of-words representations. Two variants:

- CombinedTM — BoW + sentence embeddings. Typically best coherence.
- ZeroShotTM — sentence embeddings only. Enables cross-lingual topic modeling: train on English, infer topics on German text without parallel corpora.

## Embedding-based Methods

### BERTopic

[Grootendorst, 2022](https://arxiv.org/abs/2203.05794). A modular pipeline — embed documents with sentence-transformers, reduce dimensionality with UMAP, cluster with HDBSCAN, and label clusters with class-based TF-IDF. Automatically determines the number of topics and supports dynamic, hierarchical, guided, and online modes.

See [[BERTopic]] for the full pipeline, the corrected c-TF-IDF formula, failure modes, inductive inference, and guided topic modeling.

### Top2Vec

[Angelov, 2020](https://arxiv.org/abs/2008.09470). Jointly embeds documents and words, applies UMAP + HDBSCAN to find dense clusters. Topic vectors are cluster centroids; topic words are the nearest words in embedding space. Similar philosophy to BERTopic, predates it, and uses embedding distance rather than c-TF-IDF for topic representation.

## LLM-assisted Topic Discovery

A practical hybrid that works well: run BERTopic for clustering, then use an LLM as the representation model to label each cluster. This gives the scalability of embedding-based clustering with human-readable LLM-generated labels, without paying per-document LLM costs.

Fully LLM-based approaches (e.g., TopicGPT) use an LLM to generate topic labels directly from document samples, optionally in a hierarchical refinement loop. Labels are immediately readable and you can specify granularity in natural language, but cost scales with corpus size and results are non-deterministic.

## Other Notable Approaches

### BigARTM

[Vorontsov & Potapenko, 2015](https://www.jmlr.org/papers/volume18/16-296/16-296.pdf). Extends pLSA with additive regularisation — multiple regularisers (sparsity, decorrelation, hierarchy, label supervision) combine into a single objective:

$$L(\Phi, \Theta) + \sum_i \tau_i R_i(\Phi, \Theta) \to \max$$

This flexibility allows simultaneous optimisation for topic sparsity, inter-topic distinctness, and incorporation of side information. No longer actively developed (last feature release 2019, wheels re-uploaded for newer Python versions in 2023 without feature changes). Most of what made it attractive — decorrelated, sparse, hierarchical topics — can now be achieved with BERTopic's modular pipeline.

### Dynamic Topic Models

[Blei & Lafferty, 2006](https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf). Extends LDA for temporally ordered corpora. Topic distributions evolve over time via Gaussian noise:

$$\alpha_t \mid \alpha_{t-1} \sim \mathcal{N}(\alpha_{t-1}, \sigma^2 I)$$

Captures how topics change meaning over time (e.g., "computing" in 1960 vs 2020). Available in Gensim (`DtmModel`) and tomotopy. BERTopic's `.topics_over_time()` provides a simpler alternative by recalculating c-TF-IDF per time bin, though it is not a true generative model.

### Hierarchical and nonparametric variants

- Hierarchical LDA (hLDA) — [Blei et al., 2003](https://proceedings.neurips.cc/paper/2003/file/7b41bfa5085806f1b4378907ec5a5993-Paper.pdf). Discovers a tree-structured topic hierarchy using the nested Chinese Restaurant Process. Does not require specifying $K$ or tree depth.
- Hierarchical Dirichlet Process (HDP) — [Teh et al., 2006](https://doi.org/10.1198/016214506000000302) ([preprint](https://people.eecs.berkeley.edu/~jordan/papers/hdp.pdf)). Nonparametric extension of LDA that infers $K$ from data. Available in Gensim and tomotopy.
- Correlated Topic Model (CTM) — [Blei & Lafferty, 2005](https://arxiv.org/abs/1206.3098). Replaces the Dirichlet prior with a logistic normal, allowing topics to be correlated (e.g., "genetics" and "biology" co-occur more often than "genetics" and "cooking").
- Structural Topic Model (STM) — incorporates document-level metadata (date, source, author) as covariates that affect topic prevalence and content. Useful when topics are not independent of document attributes.

## When to Use What

| Scenario | Recommended |
|---|---|
| Short texts (tweets, reviews) | BERTopic, Top2Vec |
| Long documents (papers, articles) | LDA, NMF, BERTopic |
| Mixed-membership (document in multiple topics) | LDA, NMF, CTM |
| Automatic number of topics | BERTopic, Top2Vec, HDP |
| Resource-constrained / fast results | NMF (fastest), LDA |
| State-of-the-art coherence | BERTopic |
| Temporal topic evolution | BERTopic `.topics_over_time()`, DTM |
| Cross-lingual topics | ZeroShotTM / CombinedTM |
| Large vocabulary with rare words | ETM |
| Hierarchical topic structure | hLDA, BERTopic (hierarchical mode) |
| Maximum flexibility via regularisation | BigARTM |
| Metadata-aware topics (date, author, source) | STM |
| Feature engineering for a downstream classifier | LDA (probabilistic $\theta_d$), NMF |
| LLM-quality labels at scale | BERTopic with LLM representation model |
| One-shot theme summary, small corpus | Prompt an LLM directly |

In current practice, BERTopic is the default starting point for new projects. LDA and NMF remain relevant for resource-constrained settings, mixed-membership requirements, and established pipelines.

## Tools and Libraries

| Library | Methods | Notes |
|---|---|---|
| **Gensim** | LDA, LSA/LSI, HDP, LdaMallet wrapper | Memory-efficient streaming; best for classical methods |
| **scikit-learn** | NMF, LDA, TruncatedSVD (LSA) | Tight integration with ML pipelines; in-memory ceiling |
| **BERTopic** | BERTopic (dynamic, hierarchical, guided, online) | Modular; supports LLM-based representation tuning |
| **Top2Vec** | Top2Vec | Simple API; automatic $K$ |
| **tomotopy** | LDA, HDP, CTM, DTM, SLDA, LLDA | Very fast C++ backend; many LDA variants |
| **OCTIS** | LDA, NMF, ETM, ProdLDA, CTM, and more | Benchmarking framework with hyperparameter optimisation |
| **contextualized-topic-models** | CombinedTM, ZeroShotTM | Cross-lingual support |
| **Mallet** | LDA (Gibbs sampling) | Often produces the best LDA results; Gensim has a wrapper |

> [!example]- Code example (scikit-learn NMF & LDA, BERTopic)
> ```python
> from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
> from sklearn.decomposition import NMF, LatentDirichletAllocation
> from sklearn.datasets import fetch_20newsgroups
>
> data, _ = fetch_20newsgroups(shuffle=True, random_state=1,
>                              remove=('headers', 'footers', 'quotes'),
>                              return_X_y=True)
> docs = data[:2000]
> n_topics = 10
>
> # --- NMF with TF-IDF ---
> tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
> X_tfidf = tfidf.fit_transform(docs)
> nmf = NMF(n_components=n_topics, random_state=1, init='nndsvd').fit(X_tfidf)
>
> # --- LDA with raw counts ---
> tf = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
> X_tf = tf.fit_transform(docs)
> lda = LatentDirichletAllocation(n_components=n_topics, random_state=0).fit(X_tf)
>
> def show_topics(model, feature_names, n_words=10):
>     for i, topic in enumerate(model.components_):
>         words = [feature_names[j] for j in topic.argsort()[:-n_words - 1:-1]]
>         print(f"Topic {i}: {', '.join(words)}")
>
> show_topics(nmf, tfidf.get_feature_names_out())
> show_topics(lda, tf.get_feature_names_out())
>
> # --- BERTopic ---
> from bertopic import BERTopic
> topic_model = BERTopic(random_state=42)
> topics, probs = topic_model.fit_transform(docs)
> topic_model.get_topic_info()
> ```

For a BERTopic example with LLM-based labels and soft assignment, see [[BERTopic]]. For a gensim LDA example with folding-in for new documents, see [[LDA]].

### Links

- [Blei, Ng, Jordan — Latent Dirichlet Allocation (2003)](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
- [Grootendorst — BERTopic: Neural topic modeling with a class-based TF-IDF procedure (2022)](https://arxiv.org/abs/2203.05794)
- [Blei's Topic Modeling page](http://www.cs.columbia.edu/~blei/topicmodeling.html)
- [BERTopic documentation](https://maartengr.github.io/BERTopic/index.html)
- [Gensim topic modeling documentation](https://radimrehurek.com/gensim/auto_examples/index.html)
- [OCTIS — Comparing Topic Models](https://github.com/mind-lab/octis)
- [scikit-learn Topic Extraction with NMF and LDA](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html)
- [Structural Topic Model](https://www.structuraltopicmodel.com/)
- [BigARTM documentation](http://docs.bigartm.org/en/stable/index.html)
