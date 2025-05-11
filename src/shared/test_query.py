import numpy as np
import matplotlib.pyplot as plt
from recursive_lsh_forest import RecursiveLSHForest
from lsh_forest import RandomHyperplaneLSH
from typing import Optional, List, Callable, Dict, Any
import statistics

# Parameters
n_targets = 1000
n_queries = 10
dim = 100
max_split_ratio = 1.2
max_hash_attempts = 1000
max_depth = 12
max_candidates = 100

# 1. Generate random target vectors
np.random.seed(42)
target_vectors = np.random.randn(n_targets, dim).astype(np.float32)

# 2. Build the LSH forest
lsh_family = RandomHyperplaneLSH(dim=dim)
forest = RecursiveLSHForest(
    lsh_family=lsh_family,
    l=1,
    km=max_depth,
    max_split_ratio=max_split_ratio,
    max_hash_attempts=max_hash_attempts
)
forest.build_forest(target_vectors)


def collect_candidates_with_parent_augmentation(root, query, max_candidates):
    node = root
    parent = None
    # Descend to leaf, keeping track of parent
    while node.left and node.right and node.hash_func:
        parent = node
        if node.hash_func(query) == 0:
            node = node.left
        else:
            node = node.right
    # Start with leaf candidates
    candidates = set(node.vector_ids)
    # Ascend, adding parent vector_ids
    parent = node.parent
    while parent and len(candidates) < max_candidates:
        candidates.update(parent.vector_ids)
        parent = parent.parent
    return list(candidates)[:max_candidates]


# 3. Generate random query vectors
query_vectors = np.random.randn(n_queries, dim).astype(np.float32)

# 4. For each query, traverse the tree and compute NN rank
ranks = []
for i, q in enumerate(query_vectors):
    print(f"\nQuery {i+1}/{n_queries}")
    candidates = collect_candidates_with_parent_augmentation(forest.roots[0], q, max_candidates)
    print(f"  Number of candidates collected: {len(candidates)}")
    print(f"  Candidate indices: {candidates}")
    if not candidates:
        print("  No candidates found.")
        ranks.append(None)
        continue
    dists_candidates = np.linalg.norm(target_vectors[candidates] - q, axis=1)
    best_candidate_idx_in_candidates = np.argmin(dists_candidates)
    best_candidate_global_idx = candidates[best_candidate_idx_in_candidates]
    print(f"  Best candidate index in candidates: {best_candidate_idx_in_candidates}")
    print(f"  Best candidate global index: {best_candidate_global_idx}")
    print(f"  Distance to best candidate: {dists_candidates[best_candidate_idx_in_candidates]}")
    dists_all = np.linalg.norm(target_vectors - q, axis=1)
    sorted_indices = np.argsort(dists_all)
    rank = np.where(sorted_indices == best_candidate_global_idx)[0][0] + 1  # 1-based rank
    print(f"  Rank of best candidate among all targets: {rank}")
    ranks.append(rank)

# 5. Plot histogram of ranks (excluding None)
valid_ranks = [r for r in ranks if r is not None]
plt.figure(figsize=(8, 6))
plt.hist(valid_ranks, bins=range(1, max(valid_ranks)+2), color='skyblue', edgecolor='black', align='left')
plt.xlabel('Rank of True Nearest Neighbor in Candidates')
plt.ylabel('Number of Queries')
plt.title('Histogram of True NN Rank in LSH Forest Candidates')
plt.grid(True, alpha=0.3)
plt.show()

def parameter_sweep(
    param_name: str,
    param_range: List[Any],
    n_targets: int = 1000,
    n_queries: int = 50,
    dim: int = 100,
    l: int = 1,
    km: int = 12,
    max_split_ratio: float = 1.2,
    max_hash_attempts: int = 1000,
    max_candidates: int = 100
) -> None:
    """
    Sweep over a parameter and analyze how it affects the median rank of nearest neighbors.
    
    Parameters:
    -----------
    param_name : str
        Name of parameter to sweep over ('max_depth', 'max_split_ratio', 'max_hash_attempts')
    param_range : List[Any]
        List of values to try for the parameter
    n_targets : int
        Number of target vectors to generate
    n_queries : int
        Number of query vectors to test
    dim : int
        Dimension of the vectors
    l : int
        Number of trees in the forest
    km : int
        Maximum depth of trees
    max_split_ratio : float
        Maximum allowed ratio between split group sizes
    max_hash_attempts : int
        Maximum number of hash function attempts
    max_candidates : int
        Maximum number of candidates to consider
    """
    # Base parameters dictionary
    base_params = {
        'l': l,
        'km': km,
        'max_split_ratio': max_split_ratio,
        'max_hash_attempts': max_hash_attempts,
        'max_candidates': max_candidates
    }
    
    median_ranks = []
    mean_ranks = []
    
    for param_value in param_range:
        print(f"\nTesting {param_name} = {param_value}")
        
        # Generate vectors
        np.random.seed(42)  # For reproducibility
        target_vectors = np.random.randn(n_targets, dim).astype(np.float32)
        query_vectors = np.random.randn(n_queries, dim).astype(np.float32)
        
        # Build forest with current parameter
        lsh_family = RandomHyperplaneLSH(dim=dim)
        forest_params = {**base_params, param_name: param_value}
        test_forest = RecursiveLSHForest(
            lsh_family=lsh_family,
            l=forest_params['l'],
            km=forest_params['km'],
            max_split_ratio=forest_params['max_split_ratio'],
            max_hash_attempts=forest_params['max_hash_attempts']
        )
        test_forest.build_forest(target_vectors)
        
        # Run queries
        ranks = []
        for q in query_vectors:
            candidates = collect_candidates_with_parent_augmentation(
                test_forest.roots[0], 
                q, 
                forest_params['max_candidates']
            )
            if not candidates:
                continue
                
            dists_candidates = np.linalg.norm(target_vectors[candidates] - q, axis=1)
            best_candidate_idx_in_candidates = np.argmin(dists_candidates)
            best_candidate_global_idx = candidates[best_candidate_idx_in_candidates]
            
            dists_all = np.linalg.norm(target_vectors - q, axis=1)
            sorted_indices = np.argsort(dists_all)
            rank = np.where(sorted_indices == best_candidate_global_idx)[0][0] + 1
            ranks.append(rank)
        
        if ranks:
            median_ranks.append(statistics.median(ranks))
            mean_ranks.append(statistics.mean(ranks))
        else:
            median_ranks.append(None)
            mean_ranks.append(None)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, median_ranks, 'bo-', label='Median Rank')
    plt.plot(param_range, mean_ranks, 'ro-', label='Mean Rank')
    plt.xlabel(param_name)
    plt.ylabel('Rank')
    plt.title(f'Effect of {param_name} on Nearest Neighbor Search Performance\n'
              f'(n_targets={n_targets}, n_queries={n_queries}, dim={dim})\n'
              f'Base params: l={l}, km={km}, max_split_ratio={max_split_ratio}, '
              f'max_hash_attempts={max_hash_attempts}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add parameter values as x-tick labels
    plt.xticks(param_range, [str(x) for x in param_range], rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Base parameters
    dim = 50
    n_targets = 1000
    n_queries = 20
    l = 1
    km = 12
    max_split_ratio = 2
    max_hash_attempts = 1000
    max_candidates = 10
    
    # Test different max_depth values
    parameter_sweep(
        param_name='km',
        param_range=[4, 5, 6, 7, 8, 9, 10, 11],
        n_targets=n_targets,
        n_queries=n_queries,
        dim=dim,
        l=l,
        km=km,
        max_split_ratio=max_split_ratio,
        max_hash_attempts=max_hash_attempts,
        max_candidates=max_candidates
    )
    
    # Test different max_split_ratio values
    parameter_sweep(
        param_name='max_split_ratio',
        param_range=[1.1, 1.2, 1.5, 1.8, 2.0],
        n_targets=n_targets,
        n_queries=n_queries,
        dim=dim,
        l=l,
        km=km,
        max_split_ratio=max_split_ratio,
        max_hash_attempts=max_hash_attempts,
        max_candidates=max_candidates
    )
    
    # Test different max_hash_attempts values
    parameter_sweep(
        param_name='max_hash_attempts',
        param_range=[1, 2, 5, 10, 20, 40, 80, 160],
        n_targets=n_targets,
        n_queries=n_queries,
        dim=dim,
        l=l,
        km=km,
        max_split_ratio=max_split_ratio,
        max_hash_attempts=max_hash_attempts,
        max_candidates=max_candidates
    )
