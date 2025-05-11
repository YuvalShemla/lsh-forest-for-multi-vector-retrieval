import numpy as np
import matplotlib.pyplot as plt
from recursive_lsh_forest import RecursiveLSHForest
from lsh_forest import RandomHyperplaneLSH
from typing import Optional, List, Callable

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
