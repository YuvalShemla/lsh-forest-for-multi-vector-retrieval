# Muvera + LSH Forest

> **Goal:**  Bring locality‑sensitive hashing (LSH) forests to the MUVERA multi‑vector retrieval pipeline so that *each document embedding is guaranteed to fall into a unique bucket*. The hypothesis is that density‑adaptive partitioning will reduce centroid loss, yielding higher recall—especially on skewed or clustered corpora—while retaining MUVERA’s single‑vector retrieval speed.

> **Paper:**  [MUVERA (OpenReview 2024)](https://openreview.net/pdf?id=X3ydKRcQr6)

**Idea:** replace the data‑oblivious SimHash partition with a **density‑adaptive LSH Forest** that keeps hashing until *each bucket holds at most one document vector*.  
This should:

* preserve fine‑grained token information (without the centroid loss),  
* adapt to non‑uniform vector distributions.  
* remain compatible with MUVERA’s single‑vector retrieval pipeline.

We will evaluate whether the LSH Forest variant improves recall, latency, and robustness on real IR benchmarks.

---

## Implementation Tentative Roadmap 

### Stage 1 — Baseline plumbing

1. **Install & import ColBERTv2 and vector DBs**  
2. **Load the dataset & inspect schema**  
3. **Load ColBERTv2** ( <https://github.com/stanford-futuredata/ColBERT> , <https://huggingface.co/colbert-ir/colbertv2.0> )  
   * Test on a handful of documents and queries  
4. **Verify ColBERT IR workflow** (query ↔ document embeddings)  
5. **Prototype SimHash baseline** (e.g. <https://pkg.go.dev/github.com/Cogile/simhash-lsh>)  

### Stage 2 — LSH Forest integration 

6. **Learn & prototype LSH Forest** (<https://scikit-learn.org/0.17/modules/generated/sklearn.neighbors.LSHForest.html>)  
7. *(Optional)* experiment with other random‑projection familie for the final dimensionality reduction  
8. **Build an evaluation pipeline** (recall@k, latency, memory)  
9. **Plot & analyze results** (SimHash vs LSH Forest)  

---

## Theory Tentative Roadmap

### LSH Forest ↔ MUVERA Compatibility

1. **Mapping: Formalize how MUVERA’s SimHash partition can correspond to LSH Forest leaves.**
2. **Approximation Guarantees: Show that enforcing one‑vector‑per‑leaf still permits fixed‑dimensional encodings.**  
3. **Complexity Bound: Derive how added tree depth impacts FDE dimensionality and build time.** 
4. **Practical Cases: Identify corpora where density‑adaptive splits may overfit or be problomatic, and cases where it is much better.**

> **Papers:**  [LSH Forest (Bawa et al., 2005)](https://dl.acm.org/doi/10.1145/1060745.1060840), [LSH Forest: Practical Algorithms Made Theoretical (Andoni et al.,2017)](https://www.cs.columbia.edu/~andoni/papers/ddtrees.pdf)

## Repository structure:

