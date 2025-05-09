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
        """Takes two sets of vectors and calculates chamfer"""

        # (n, m) matrix of all the pairwise dot products
        dot_products = queries @ vectors.T

        # sum the max value for each query (row)
        chamfer = np.sum(np.max(dot_products, axis=1))
        return chamfer

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

def recall(best, approximates, k):
    "Calculates Recall@k for sorted lists of document ids"

    assert len(best) == len(approximates), "must have same number of elements"

    true_set = set(best[:k])
    predicted_set = set(approximates[:k])

    return len(true_set & predicted_set) / k
