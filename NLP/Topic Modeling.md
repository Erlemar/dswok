---
tags:
  - nlp
  - concept
  - topic-modeling
  - unsupervised
aliases:
  - Topic Modeling
  - Topic Models
prereqs:
  - "[[Term Frequency-Inverse Document Frequency]]"
  - "[[Word Embeddings]]"
---
Topic modeling is an unsupervised technique for discovering abstract themes in a document collection, where a document is whatever unit of text you are analyzing (article, review, tweet, paragraph, support ticket). Each theme (topic) is expressed through a distribution over words (classical methods) or a cluster in embedding space labeled by representative words (embedding-based methods).

The key contrast with classification: classification uses predefined labels, whereas topic modeling discovers them from scratch. As a result, you get structure without supervision, but you have to decide afterward whether it's the one you wanted.

The field evolved from matrix factorization (LSA, NMF) through probabilistic generative models (pLSA, LDA) to modern neural and embedding-based approaches ([[BERTopic]], ETM). Find the list of approaches in [[Topic Modeling Methods]]. [[LDA]] and [[BERTopic]] have separate notes due to their significance.

## Key assumptions

- Documents are mixtures of topics (classical and neural methods), or each document belongs to one primary cluster (embedding-based methods with hard assignment)
- Topics are distributions over words (classical/neural) or clusters in embedding space labeled by representative words (BERTopic, Top2Vec)
- Bag-of-words for classical methods; embedding-based methods inherit contextual understanding from their encoder

![[Topic Modeling Mixture.excalidraw.light.svg]]

## When topic modeling is not the right tool

- You already have labels. Use supervised classification, or zero-shot classification with an LLM.
- You want stable, reproducible labels. Most topic models drift between runs and across retraining; downstream consumers hate that.
- You can afford an LLM to label everything. If the corpus fits in a context window (or you can pay per-document), prompt the LLM for theme summaries directly. Alternatively, use an LLM to generate synthetic labels on a subset and train a cheap supervised classifier; this can be better than raw topics.
- Your corpus is small (fewer than a few thousand documents). Embed with a sentence transformer and do clustering.
- You mainly need semantic similarity, not interpretable themes. Use embeddings and retrieval.
- Very short texts (tweets, queries, chat snippets). Classical topic models will struggle due to sparsity. Embedding-based methods or specialized short-text models (Biterm Topic Model) handle this better.

## When topic modeling is still the right tool

- You need interpretability or want to explore the data.
- You need cheap and/or fast inference. LDA's folding-in and BERTopic's nearest-centroid assignment cost milliseconds per document; per-document LLM calls cost orders of magnitude more.
- You want to create features.
- You want to track theme prevalence over time. Dynamic topics or BERTopic's `.topics_over_time()` turn a corpus-plus-timestamps into a time series of themes.

## Evaluation

### Coherence metrics

Coherence measures how semantically related the top-$N$ words per topic are.

UMass Coherence — based on document co-occurrence of top word pairs from the training corpus:

$$C_{\text{UMass}} = \frac{2}{N(N-1)} \sum_{i=2}^{N} \sum_{j=1}^{i-1} \log \frac{D(w_i, w_j) + 1}{D(w_j)}$$

where $D(w_i, w_j)$ is the count of documents containing both words. Ranges roughly $[-14, 0]$; less negative is better.

NPMI (Normalized Pointwise Mutual Information) — normalizes PMI to $[-1, 1]$:

$$\text{NPMI}(w_i, w_j) = \frac{\log \frac{P(w_i, w_j)}{P(w_i) \, P(w_j)}}{-\log P(w_i, w_j)}$$

Showed the best overall correlation with human judgment in classical topic model evaluations. Typical "good" values: 0.05–0.25+. [Hoyle et al. (2021)](https://arxiv.org/abs/2107.02173) showed that automated coherence metrics correlate poorly with human judgment for neural topic models — treat them as a rough sanity check, not a leaderboard.

[C_v](https://dl.acm.org/doi/10.1145/2684822.2685324). A composite measure combining NPMI with cosine similarity of sliding-window word vectors. Ranges $[0, 1]$. Gensim's default coherence metric.

### Perplexity

For probabilistic models, perplexity measures how well the model predicts held-out documents:

$$\text{Perplexity} = \exp\!\left(-\frac{\sum_{d} \log p(\mathbf{w}_d)}{\sum_{d} N_d}\right)$$

Lower is better, but perplexity and human judgment are negatively correlated — models that optimize perplexity tend to produce less interpretable topics ([Chang et al., 2009](https://papers.nips.cc/paper/2009/hash/f92586a25bb3145facd64ab20fd554ff-Abstract.html)).

### Topic diversity

- Topic Uniqueness (TU): fraction of top-$N$ words that appear in only one topic. TU = 1 means zero word overlap across topics.
- Proportion of unique words: percentage of distinct words across all topics' top-$N$ lists.

### Manual inspection

Automated metrics catch errors but may approve topics that a human reader would reject. Things to check:

- Read the top 10–20 words of every topic.
- Read three to five representative documents per topic.
- Ask whether two topics should be merged or whether a topic is actually two.
- Try to assign a short human-readable label; if you can't, the topic is probably noise.

### Human evaluation

- Word intrusion: insert a random word into a topic's top words; humans identify the intruder. Higher accuracy means a more coherent topic.
- Topic intrusion: show a document with its assigned topics plus one random topic; humans identify the misfit.
- Direct rating: judges rate topic quality on a Likert scale for coherence, usefulness, and interpretability.

### Visualization

- pyLDAvis — the standard interactive tool for classical models (LDA). Projects topics into 2D via Jensen-Shannon divergence + Principal Coordinate Analysis (`js_PCoA`, the default `mds` option) and compares within-topic term frequencies against corpus-wide frequencies. See the [pyLDAvis demo](https://github.com/bmabey/pyLDAvis) for an example screenshot.
- BERTopic built-in visualizations — UMAP projections of the document space overlaid with cluster boundaries, intertopic distance maps, topic hierarchies, and topics-over-time. See the [BERTopic visualization gallery](https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html).
![[Pasted image 20260415205906.png]]
## Choosing the number of topics

For methods that require $K$ (LDA, NMF, ProdLDA):

- Run multiple values, plot coherence vs $K$, look for a peak or elbow.
- Coherence plots often plateau rather than peak — there are usually several reasonable values depending on desired granularity.
- Inspect topics manually at candidate $K$ values. Repeated keywords across topics signal too many; merged themes signal too few.
- Compare candidate $K$ values on a train/validation split, but rely mainly on coherence plus manual inspection.
- Starting range: try $K \in [5, 50]$.

BERTopic and Top2Vec sidestep this via HDBSCAN, where `min_cluster_size` indirectly controls granularity. Hierarchical merging allows post-hoc reduction.

## Preprocessing

Classical methods (LDA, NMF, LSA):

- Tokenize, lowercase, remove punctuation.
- Remove stopwords — use domain-specific lists, not just generic ones (e.g., "patient" in medical corpora; "thanks", "regards", "http" in email; company boilerplate).
- Lemmatize (optional; improves interpretability).
- Filter rare terms and overly frequent ones.
- Consider bigrams/trigrams for multi-word concepts.
- NMF and LSA work best with [[Term Frequency-Inverse Document Frequency|TF-IDF]] weighting; LDA uses raw counts.

Embedding-based methods (BERTopic, Top2Vec):

- Minimal preprocessing — the [[Transformer|transformer]] handles semantics.
- Do not remove stopwords or lemmatize before embedding.
- Clean up URLs, HTML tags, and special characters.
- Very short documents (<10 words) produce poor embeddings — aggregate or filter.

### Scale

Rough thresholds for what you can run on one machine:

- Mallet LDA — single-machine, Java. Handles millions of documents if you have RAM and patience; best topic quality for classical LDA.
- Gensim LDA — streams from disk; handles millions of documents with constant memory via online variational Bayes.
- scikit-learn LDA — loads the corpus into memory; practical ceiling around 100k documents before OOM on typical machines.
- BERTopic — embedding compute is the main cost (GPU-hours per million documents with `all-MiniLM-L6-v2`). UMAP memory grows roughly quadratically; fails around 1–2M documents without `low_memory=True` or approximate nearest neighbors. HDBSCAN handles the million-document range if UMAP output fits.
- NMF — very fast for the matrix sizes it handles; typically memory-bounded by the document-term matrix.

### Incremental and online updates

- Gensim — online VB is the default training mode; `.update(new_corpus)` folds in additional documents without retraining. Hyperparameters can be re-tuned or held fixed.
- BERTopic — `.partial_fit()` enables streaming updates with a subset of documents at a time; `merge_models()` combines independently trained models. See [BERTopic online topic modeling docs](https://maartengr.github.io/BERTopic/getting_started/online/online.html).
- River — a streaming-ML library with a `river.feature_extraction` module that supports incremental TF-IDF and can feed NMF.
- scikit-learn NMF — supports `partial_fit` for mini-batch updates via `MiniBatchNMF`.

### Topic drift and alignment

When retraining on a newer corpus, the topic IDs produced by the new model do not correspond to the old ones. There are multiple ways to deal with it:

- Topic alignment via the Hungarian algorithm. Compute a similarity matrix between old and new topic-word distributions (or embedding centroids), then solve the optimal assignment. Libraries: `scipy.optimize.linear_sum_assignment`.
- BERTopic `merge_models` — merges topics from models trained on disjoint data; can also be used as a rolling-update mechanism where old topics are preserved.
- Manual curation —  human review of new topics against the previous ones.

### Multilingual corpora

- BERTopic with a multilingual sentence encoder — `paraphrase-multilingual-MiniLM-L12-v2` or `distiluse-base-multilingual-cased-v2` handles 50+ languages; clusters documents across languages if you want unified topics.
- ZeroShotTM / CombinedTM — designed for cross-lingual transfer. Train on one language, infer topics on another without parallel corpora.
- Per-language classical pipelines — for LDA/NMF on mixed-language corpora, segment by language first, then train separate models per language. Topics will not be aligned across languages by default.

### What the output actually looks like

- Top words per topic (typically 10–20) — the canonical interpretable label.
- Representative documents per topic — 3–5 exemplars that make the topic concrete for humans.
- Document-topic matrix — either probabilistic ($\theta_d$ from LDA) or a single cluster ID (BERTopic default) with optional soft distribution via `.approximate_distribution()`.
- Topic prevalence over time — if the corpus has timestamps, aggregate topic assignments per time bin. Useful for trend reports.
- Curated human-readable labels — after manual inspection, replace "Topic 3" with "Refund and Billing Complaints". This is what gets handed to analysts and PMs.

### Post-processing

- Merge near-duplicate topics — coherence plots don't catch topics that say the same thing with different top words. Check pairwise cosine similarity on topic-word vectors or topic centroids.
- Drop junk topics — boilerplate topics ("thanks, please, regards"), formatting artifacts, or topics that match no documents meaningfully. BERTopic's topic -1 (outliers) is the extreme case.
- Rename topics — generic c-TF-IDF labels rarely survive user feedback. Rewrite as short noun phrases that an analyst would recognize.
- Inspect outliers — the noise cluster can be the most interesting one. Documents that don't fit may be early signals of emerging themes.

### Links

- [Roder et al. — Exploring the Space of Topic Coherence Measures (2015)](https://dl.acm.org/doi/10.1145/2684822.2685324)
- [Chang et al. — Reading Tea Leaves: How Humans Interpret Topic Models (2009)](https://papers.nips.cc/paper/2009/hash/f92586a25bb3145facd64ab20fd554ff-Abstract.html)
- [Hoyle et al. — Is Automated Topic Model Evaluation Broken? (2021)](https://arxiv.org/abs/2107.02173)
- [pyLDAvis](https://github.com/bmabey/pyLDAvis)
- [BERTopic visualization gallery](https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html)
