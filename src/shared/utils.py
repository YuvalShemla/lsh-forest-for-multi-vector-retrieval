import numpy as np
import time
import os
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from shared.lsh_forest import LSHForest, MultiDocLSHForest, RandomHyperplaneLSH

# =======================
# Chamfer
# =======================
def chamfer(queries: np.ndarray, vectors: np.ndarray):
        """Takes two sets of vectors and calculates chamfer
        
        O()
        """

        # (n, m) matrix of all the pairwise dot products
        dot_products = queries @ vectors.T

        # sum the max value for each query (row)
        chamfer = np.sum(np.max(dot_products, axis=1))
        return chamfer

def score(score_fn, query_vecs, doc_vecs, **kwargs):
    n_q, n_d = len(query_vecs), len(doc_vecs)
    score = score_fn(query_vecs, doc_vecs, **kwargs)
    return score


# =======================
# Data Loading and Encoding
# =======================
def load_and_encode_data(n, q, m, d):
    if os.path.exists("../msmarco/corpus_vectors.npy") and os.path.exists("../msmarco/query_vectors.npy"):
        print("Corpus and query vectors found. Skipping encoding.")
        vectors = np.load("../msmarco/corpus_vectors.npy")
        queries = np.load("../msmarco/query_vectors.npy")
    else:
        print("Loading raw data from MS Marco")
        corpus, raw_queries, _ = GenericDataLoader(data_folder="../msmarco/").load(split="train")

        # Sample n documents and 1 query
        corpus = dict(list(corpus.items())[:n])
        query_text = list(raw_queries.values())[0]

        print("Encoding with SentenceTransformer")
        model = SentenceTransformer('msmarco-distilbert-base-v4')

        vectors = np.zeros((n, m, d), dtype=np.float32)
        for i, text in enumerate([doc['text'] for doc in corpus.values()]):
            sentences = text.split('. ')
            embeddings = model.encode(sentences[:m], convert_to_tensor=True).cpu().numpy()
            vectors[i, :min(m, embeddings.shape[0]), :min(d, embeddings.shape[1])] = embeddings[:m, :d]

        queries = np.zeros((q, d), dtype=np.float32)
        query_sentences = query_text.split('. ')
        query_embeddings = model.encode(query_sentences[:q], convert_to_tensor=True).cpu().numpy()
        queries[:min(q, query_embeddings.shape[0]), :min(d, query_embeddings.shape[1])] = query_embeddings[:q, :d]

        np.save("../msmarco/corpus_vectors.npy", vectors)
        np.save("../msmarco/query_vectors.npy", queries)

    return vectors, queries

# =======================
# LSH Forest Construction
# =======================
def build_simple_lsh(vectors, l, k, km, d):
    forests = [LSHForest(RandomHyperplaneLSH(d), l, k, km) for _ in range(len(vectors))]
    for i, forest in enumerate(forests):
        forest.batch_insert(vectors[i])
    return forests

def build_multidoc_lsh(vectors, l, k, km, d):
    forest = MultiDocLSHForest(RandomHyperplaneLSH(d), l, k, km)
    forest.batch_insert(vectors)
    return forest

# =======================
# Experiment Execution
# =======================
def experiment(variable, values, defaults, forest_type, best_fn, vectors, queries, d):
    sims, docs = [], []
    start_total = time.perf_counter()

    for value in values:
        parameters = defaults.copy()
        parameters[variable] = value

        start = time.perf_counter()
        if forest_type == "simple":
            forest = build_simple_lsh(vectors, parameters['l'], parameters['k'], parameters['km'], d)
        elif forest_type == "multidoc":
            forest = build_multidoc_lsh(vectors, parameters['l'], parameters['k'], parameters['km'], d)
        sim, doc = best_fn(forest, queries, parameters['a'])
        end = time.perf_counter()

        sims.append(sim)
        docs.append(doc)
        print(f"{variable}={value} completed in {end - start:.4f} seconds.")

    end_total = time.perf_counter()
    print(f"\nFinished experiment with varying {variable} in {end_total - start_total:.4f} seconds.")

    return sims, docs
