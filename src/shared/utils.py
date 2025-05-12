import numpy as np
import time
import matplotlib.pyplot as plt
import os
import random
from shared.lsh_forest import LSHForest, MultiDocLSHForest, RandomHyperplaneLSH
from shared.recursive_lsh_forest import RecursiveLSHForest, Node

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer



def timed(func, *args):
    start = time.perf_counter()
    output = func(*args)
    end = time.perf_counter()
    print(f"{func.__name__} took {end - start:.4f} sec")
    return output


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


# Recursive LSH Forest functions 

def build_rec_forest(vectors, l, km, d, max_split_ratio=1.2, max_hash_attempts=1000):
    """
    Build a recursive LSH forest with the given parameters.
    
    Parameters:
    -----------
    vectors : np.ndarray
        The vectors to build the forest from
    l : int
        Number of trees in the forest
    km : int
        Maximum depth of trees
    d : int
        Dimension of vectors
    max_split_ratio : float
        Maximum allowed ratio between split group sizes
    max_hash_attempts : int
        Maximum number of hash function attempts
        
    Returns:
    --------
    RecursiveLSHForest
        The built LSH forest
    """
    lsh_family = RandomHyperplaneLSH(dim=d)
    forest = RecursiveLSHForest(
        lsh_family=lsh_family,
        l=l,
        km=km,
        max_split_ratio=max_split_ratio,
        max_hash_attempts=max_hash_attempts
    )
    forest.build_forest(vectors)
    return forest


def query_rec(forest, query, max_candidates, k=1):
    """
    Query the recursive LSH forest for k nearest neighbors.
    Collects max_candidates from each tree separately and then combines them.
    
    Parameters:
    -----------
    forest : RecursiveLSHForest
        The LSH forest to query
    query : np.ndarray
        The query vector
    max_candidates : int
        Maximum number of candidates to consider from each tree
    k : int
        Number of nearest neighbors to return (default=1)
        
    Returns:
    --------
    List[int]
        List of k nearest neighbor indices
    """
    all_candidates = set()
    
    # Process each tree separately
    for tree_idx in range(forest.l):
        tree_candidates = set()
        
        # Get leaf node for this tree
        root = forest.roots[tree_idx]
        node = root
        while node.left and node.right and node.hash_func:
            if node.hash_func(query) == 0:
                node = node.left
            else:
                node = node.right
        
        # Start with the leaf node's vectors
        tree_candidates.update(node.vector_ids)
        
        # If we need more candidates, go up the tree
        while len(tree_candidates) < max_candidates and node.parent:
            # Get parent's vectors
            new_candidates = set(node.parent.vector_ids)
            available_new = list(new_candidates - tree_candidates)
            
            if len(available_new) > 0:
                # Add new candidates up to max_candidates
                remaining_slots = max_candidates - len(tree_candidates)
                if len(available_new) <= remaining_slots:
                    tree_candidates.update(available_new)
                else:
                    # Randomly sample from new candidates
                    sampled = random.sample(available_new, remaining_slots)
                    tree_candidates.update(sampled)
            
            # Move up to parent
            node = node.parent
        
        # Add this tree's candidates to the overall set
        all_candidates.update(tree_candidates)
    
    # Convert to list and compute distances
    candidates_list = list(all_candidates)
    if not candidates_list:
        return []
        
    # Compute distances and get k nearest neighbors
    candidate_vectors = np.array([forest.data[i] for i in candidates_list])
    dists = np.linalg.norm(candidate_vectors - query, axis=1)
    sorted_indices = np.argsort(dists)
    return [candidates_list[i] for i in sorted_indices[:k]]


def experiment_rec(variable, values, defaults, vectors, queries, d):
    """
    Run an experiment varying a parameter and measuring performance.
    Implements parameter sweep directly for recursive LSH forest.
    
    Parameters:
    -----------
    variable : str
        Name of parameter to vary ('l', 'km', 'max_split_ratio', 'max_hash_attempts', 'max_candidates')
    values : List[Any]
        Values to try for the parameter
    defaults : Dict
        Default values for other parameters
    vectors : np.ndarray
        Target vectors
    queries : np.ndarray
        Query vectors
    d : int
        Dimension of vectors
        
    Returns:
    --------
    Tuple[List[float], List[float]]
        Lists of median ranks and mean ranks for each parameter value
    """

    
    # Create results directory if it doesn't exist
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results/test_query'))
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize results storage
    median_ranks = []
    mean_ranks = []
    
    # Get default parameters
    l = defaults.get('l', 1)
    km = defaults.get('km', 12)
    max_split_ratio = defaults.get('max_split_ratio', 1.2)
    max_hash_attempts = defaults.get('max_hash_attempts', 1000)
    max_candidates = defaults.get('max_candidates', 100)
    
    # Run parameter sweep
    for value in values:
        # Update the parameter being tested
        if variable == 'l':
            l = value
        elif variable == 'km':
            km = value
        elif variable == 'max_split_ratio':
            max_split_ratio = value
        elif variable == 'max_hash_attempts':
            max_hash_attempts = value
        elif variable == 'max_candidates':
            max_candidates = value
            
        # Build forest with current parameters
        forest = build_rec_forest(
            vectors=vectors,
            l=l,
            km=km,
            d=d,
            max_split_ratio=max_split_ratio,
            max_hash_attempts=max_hash_attempts
        )
        
        # Run queries and collect ranks
        ranks = []
        for query in queries:
            # Get approximate nearest neighbors
            approx_nn = query_rec(forest, query, max_candidates, k=1)
            if not approx_nn:
                continue
                
            # Calculate true distances to all vectors
            true_dists = np.linalg.norm(vectors - query, axis=1)
            true_ranks = np.argsort(true_dists)
            
            # Find rank of approximate nearest neighbor
            approx_idx = approx_nn[0]
            rank = np.where(true_ranks == approx_idx)[0][0]
            ranks.append(rank)
            
        # Calculate statistics
        if ranks:
            median_ranks.append(np.median(ranks))
            mean_ranks.append(np.mean(ranks))
        else:
            median_ranks.append(float('inf'))
            mean_ranks.append(float('inf'))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(values, median_ranks, 'b-', label='Median Rank')
    plt.plot(values, mean_ranks, 'r--', label='Mean Rank')
    plt.xlabel(f'{variable}')
    plt.ylabel('Rank')
    plt.title(f'Effect of {variable} on Search Quality')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(os.path.join(results_dir, f'{variable}_sweep_n{len(vectors)}_q{len(queries)}_d{d}.png'))
    plt.close()
    
    return median_ranks, mean_ranks


def recall_rec(best, approximates, k=10):
    """
    Calculate recall@k for a single query.
    
    Parameters:
    -----------
    best : List[int]
        True top-k nearest neighbors
    approximates : List[int]
        Approximate top-k nearest neighbors from LSH forest
    k : int
        Number of neighbors to consider (default=10)
        
    Returns:
    --------
    float
        Recall@k score (between 0 and 1)
    """
    # Take top k from both lists
    true_set = set(best[:k])
    predicted_set = set(approximates[:k])
    
    # Calculate recall
    return len(true_set & predicted_set) / k


def load_and_prepare_beir_dataset(
    dataset: str = "hotpotqa",
    n: int = 1000,
    min_m: int = 100,
    min_q: int = 15,
    split: str = "test",
    save_encoded: bool = True
):
    """
    Load and prepare a BEIR dataset, selecting the top n longest documents,
    embedding them into multi-vector representations, and saving the vectors.

    Parameters:
    -----------
    dataset : str
        Name of the BEIR dataset (e.g., "msmarco", "hotpotqa").
    n : int
        Number of longest documents to use.
    min_m : int
        Minimum number of words per document for the multi-vector representation.
    min_q : int
        Minimum number of words for the query (attempts to find the shortest query meeting this).
    split : str
        Split of the dataset to use ("test", "train", etc.).
    save_encoded : bool
        Whether to save the encoded vectors to disk.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, dict, str]
        Encoded document vectors, query vectors, corpus dictionary, and query text.
    """
    # Model selection based on dataset
    model_mapping = {
        "msmarco": 'msmarco-distilbert-base-v4',
        "hotpotqa": 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
        "nq": 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
        "scidocs": 'sentence-transformers/all-mpnet-base-v2',
        "arguana": 'sentence-transformers/all-mpnet-base-v2',
        "quora": 'sentence-transformers/all-mpnet-base-v2'
    }
    
    model_name = model_mapping.get(dataset.lower(), 'msmarco-distilbert-base-v4')
    print(f"Using model: {model_name} for dataset: {dataset}")

    # Set the dataset path
    dataset_dir = f"../{dataset}"
    corpus_file = f"{dataset_dir}/corpus.npy"
    query_file = f"{dataset_dir}/query.npy"
    vectors_file = f"{dataset_dir}/corpus_vectors.npy"
    query_vectors_file = f"{dataset_dir}/query_vectors.npy"

    # Load + download BEIR dataset
    if not os.path.exists(f'{dataset_dir}/corpus.jsonl'):
        print(f"{dataset.capitalize()} dataset not found. Downloading...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        util.download_and_unzip(url, "../")

        zip_path = f"{dataset_dir}/{dataset}.zip"
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("Zip file deleted.")

    # Loading Corpus and Queries
    print("Looking for raw corpus and query data.")
    if os.path.exists(corpus_file) and os.path.exists(query_file):
        print("Raw corpus and query data found. Loading from local files.")
        corpus = np.load(corpus_file, allow_pickle=True).item()
        queries = np.load(query_file, allow_pickle=True).item()
    else:
        print(f"Loading raw data from {dataset.capitalize()}")
        corpus, queries, _ = GenericDataLoader(data_folder=dataset_dir).load(split=split)

        # Sort corpus by document length (descending)
        sorted_corpus = sorted(corpus.items(), key=lambda x: len(x[1]['text'].split()), reverse=True)
        corpus = dict(sorted_corpus[:n])  # Only keep the top n longest documents

        if save_encoded:
            np.save(corpus_file, corpus)
            np.save(query_file, queries)
    
    # Selecting the shortest query meeting the minimum length (min_q)
    query_texts = [q for q in queries.values() if len(q.split()) >= min_q]
    if query_texts:
        query = min(query_texts, key=len)
        print(f"Selected query with {len(query_texts)} words: {query}")
    else:
        query = max(queries.values(), key=len)  # Default to the longest query available
        print(f"No query met the minimum length. Using longest query: {query}")

    # Embedding Corpus and Queries
    print("Embedding corpus and queries")
    model = SentenceTransformer(model_name)

    print("Converting corpus into list of text")
    corpus_texts = [doc['text'] for doc in corpus.values()]

    # Determine the minimum number of words in the selected long documents
    min_corpus_words = min(len(text.split()) for text in corpus_texts)
    m = max(min_corpus_words, min_m)
    print(f"Setting vectors per document (m) to {m}")

    print("Encoding documents")
    d = model.get_sentence_embedding_dimension()
    vectors = np.zeros((len(corpus), m, d), dtype=np.float32)

    for i, text in enumerate(corpus_texts):
        words = text.split(' ')
        total_words = len(words)
        block_size = total_words // m
        remainder = total_words % m

        blocks = []
        start = 0
        for j in range(m):
            end = start + block_size + (1 if j < remainder else 0)
            blocks.append(" ".join(words[start:end]))
            start = end

        # Generate embeddings for each block
        block_embeddings = model.encode(blocks, convert_to_tensor=True)
        block_embeddings = block_embeddings.cpu().numpy()
        
        # Populate the document vectors with the processed embeddings
        vectors[i] = block_embeddings

    # Encoding the Query (Single Query)
    print("Encoding the query")
    query_words = query.split(' ')
    query_word_embeddings = model.encode(query_words, convert_to_tensor=True)
    queries = query_word_embeddings.cpu().numpy()

    # Save encoded vectors
    if save_encoded:
        np.save(vectors_file, vectors)
        np.save(query_vectors_file, queries)

    print(f"Data loading and encoding complete. Found {vectors.shape[0]} document vectors of shape {vectors.shape} and query vector of shape {queries.shape}.")

    return vectors, queries, corpus, query, d


def main():
    """
    Test the recursive LSH forest implementation with parameter sweep for l and recall test.
    """
    # Test configurations
    n_vectors = 10_000  # Number of target vectors
    n_queries = 100    # Number of query vectors
    dim = 50          # Vector dimension
    
    # Generate random vectors
    print("Generating test vectors...")
    vectors = np.random.randn(n_vectors, dim)
    queries = np.random.randn(n_queries, dim)
    
    # Default parameters
    defaults = {
        'km': 15,
        'max_split_ratio': 2,
        'max_hash_attempts': 100,
        'max_candidates': 100
    }
    
    # Test case 1: Parameter sweep for l
    print("\nTest Case 1: Parameter sweep for number of trees (l)")
    l_values = list(range(1, 11))  # l from 1 to 10
    median_ranks, mean_ranks = experiment_rec(
        variable='l',
        values=l_values,
        defaults=defaults,
        vectors=vectors,
        queries=queries,
        d=dim
    )
    print(f"Median ranks for l sweep: {median_ranks}")
    print(f"Mean ranks for l sweep: {mean_ranks}")
    
    # Test case 2: Recall test
    print("\nTest Case 2: Recall test")
    forest = build_rec_forest(
        vectors=vectors,
        l=10,  # Using middle value from l sweep
        km=defaults['km'],
        d=dim,
        max_split_ratio=defaults['max_split_ratio'],
        max_hash_attempts=defaults['max_hash_attempts']
    )
    
    # Calculate true nearest neighbors here only for the first query from query vector
    query = queries[0] 
    true_dists = np.linalg.norm(vectors - query, axis=1)
    true_nn = np.argsort(true_dists)[:100]  # Top 100 true nearest neighbors
    
    # Get approximate nearest neighbors
    approx_nn = query_rec(forest, query, max_candidates=defaults['max_candidates'], k=100)
    
    # Calculate recall
    recall_score = recall_rec(true_nn, approx_nn, k=100)
    print(f"Recall@100: {recall_score:.2f}")


if __name__ == "__main__":
    main()