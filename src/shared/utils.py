import numpy as np
import time
import matplotlib.pyplot as plt
import os
from .lsh_forest import LSHForest, MultiDocLSHForest, RandomHyperplaneLSH



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
    from .recursive_lsh_forest import RecursiveLSHForest
    
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
    Uses a level-by-level approach across all trees to collect candidates.
    
    Parameters:
    -----------
    forest : RecursiveLSHForest
        The LSH forest to query
    query : np.ndarray
        The query vector
    max_candidates : int
        Maximum number of candidates to consider
    k : int
        Number of nearest neighbors to return (default=1)
        
    Returns:
    --------
    List[int]
        List of k nearest neighbor indices
    """
    from .recursive_lsh_forest import Node
    import random
    
    # First, collect leaf nodes from all trees
    all_candidates = set()
    current_level_nodes = []
    
    # Get all leaf nodes from all trees
    for tree_idx in range(forest.l):
        root = forest.roots[tree_idx]
        # Find leaf node for this tree
        node = root
        while node.left and node.right and node.hash_func:
            if node.hash_func(query) == 0:
                node = node.left
            else:
                node = node.right
        current_level_nodes.append(node)
        all_candidates.update(node.vector_ids)
    
    # If we need more candidates, start going up the trees level by level
    while len(all_candidates) < max_candidates:
        next_level_nodes = []
        new_candidates = set()
        
        # Get parent nodes for all current level nodes
        for node in current_level_nodes:
            if node.parent:
                next_level_nodes.append(node.parent)
                new_candidates.update(node.parent.vector_ids)
        
        if not next_level_nodes:  # No more parents to consider
            break
            
        # Add new candidates up to max_candidates
        remaining_slots = max_candidates - len(all_candidates)
        available_new = list(new_candidates - all_candidates)
        if len(new_candidates) <= remaining_slots:
            all_candidates.update(new_candidates)
        elif len(available_new) > 0:
            # Randomly sample from new candidates
            sample_size = min(remaining_slots, len(available_new))
            sampled = random.sample(available_new, sample_size)
            all_candidates.update(sampled)
        
        current_level_nodes = next_level_nodes
    
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
        l=5,  # Using middle value from l sweep
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