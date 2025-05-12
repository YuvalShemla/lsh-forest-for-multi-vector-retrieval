import random
from typing import List, Tuple, Optional, Callable, Sequence, Dict, Set
from dataclasses import dataclass
from collections import Counter
import numpy as np
import heapq
from enum import Enum


class DepthWeightScheme(Enum):
    EXPONENTIAL = "exp"
    LINEAR = "linear"
    LOGISTIC = "logistic"


@dataclass
class ScorerConfig:
    depth_scheme: DepthWeightScheme = DepthWeightScheme.EXPONENTIAL
    gamma: float = 0.5  # for exp
    lin_clip: bool = True
    popularity: bool = False
    beta: float = 1.0  # strength of IDF
    skip_root: bool = True
    weight_floor: float = 1e-6  # early-stop threshold


class Node:
    """A binary tree node for the document-aware LSH forest implementation."""
    
    def __init__(self, depth: int, parent: Optional['Node'] = None):
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.depth = depth
        self.parent = parent
        self.vector_ids: List[int] = []  # IDs of vectors in this node
        self.doc_counts: Dict[int, int] = {}  # {doc_id: count of vectors}
        self.hash_func: Optional[Callable] = None
        self.trial_attempts: int = 0


class ForestVote:
    """
    A forest-based voting system that uses LSH trees to score and rank documents.
    Each tree in the forest contributes votes to documents based on their similarity
    to the query vectors.
    """
    
    def __init__(
        self, 
        lsh_family: "LSHFamily", 
        l: int = 10, 
        km: int = 64,
        max_hash_attempts: int = 1000,
        max_split_ratio: float = 2.0
    ):
        self.lsh_family = lsh_family
        self.l = l
        self.km = km
        self.max_hash_attempts = max_hash_attempts
        self.max_split_ratio = max_split_ratio
        self.data: List[np.ndarray] = []  # The vectors themselves
        self.doc_of_vec: List[int] = []   # Document ID for each vector
        self.n_docs: int = 0              # Total number of documents
        self.roots: List[Node] = []
        self.nodes: List[List[Node]] = []  # List of nodes for each tree

    def build_forest(self, vectors: Sequence[np.ndarray], doc_ids: List[int]):
        """Build the forest from vectors and their document IDs."""
        self.data = list(vectors)
        self.doc_of_vec = doc_ids
        self.n_docs = max(doc_ids) + 1
        self.roots = []
        self.nodes = []
        
        for _ in range(self.l):
            root = Node(0, parent=None)
            self.roots.append(root)
            tree_nodes = [root]
            self.nodes.append(tree_nodes)
            self._build_tree(root, list(range(len(vectors))), tree_nodes)

    def _build_tree(
        self, 
        node: Node, 
        vector_indices: List[int],
        tree_nodes: List[Node]
    ):
        """Recursively build a tree by splitting vectors based on hash values."""
        # Store vectors and compute document counts
        node.vector_ids = vector_indices.copy()
        node.doc_counts = Counter(self.doc_of_vec[idx] for idx in vector_indices)
        
        # Base cases
        if len(vector_indices) <= 1 or node.depth >= self.km:
            return
            
        best_split = None
        best_ratio = float('inf')
        node.trial_attempts = 0
        
        for _ in range(self.max_hash_attempts):
            node.trial_attempts += 1
            hash_func = self.lsh_family.sample()
            left_indices = []
            right_indices = []
            
            for idx in vector_indices:
                if hash_func(self.data[idx]) == 0:
                    left_indices.append(idx)
                else:
                    right_indices.append(idx)
                    
            if left_indices and right_indices:
                ratio = max(len(left_indices), len(right_indices)) / min(len(left_indices), len(right_indices))
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_split = (hash_func, left_indices, right_indices)
                if ratio <= self.max_split_ratio:
                    node.hash_func = hash_func
                    left_node = Node(node.depth + 1, parent=node)
                    right_node = Node(node.depth + 1, parent=node)
                    tree_nodes.extend([left_node, right_node])
                    node.left = left_node
                    node.right = right_node
                    self._build_tree(left_node, left_indices, tree_nodes)
                    self._build_tree(right_node, right_indices, tree_nodes)
                    return
                    
        if best_split is not None:
            hash_func, left_indices, right_indices = best_split
            node.hash_func = hash_func
            left_node = Node(node.depth + 1, parent=node)
            right_node = Node(node.depth + 1, parent=node)
            tree_nodes.extend([left_node, right_node])
            node.left = left_node
            node.right = right_node
            self._build_tree(left_node, left_indices, tree_nodes)
            self._build_tree(right_node, right_indices, tree_nodes)

    def _get_path(self, query_vec: np.ndarray, root: Node) -> List[Node]:
        """Get the path from root to leaf for a query vector."""
        path = []
        current = root
        
        while current is not None:
            path.append(current)
            if current.hash_func is None:  # Leaf node
                break
            if current.hash_func(query_vec) == 0:
                current = current.left
            else:
                current = current.right
                
        return path

    def _depth_weight(self, level: int, L: int, config: ScorerConfig) -> float:
        """Compute the weight based on node depth."""
        if config.depth_scheme == DepthWeightScheme.EXPONENTIAL:
            return config.gamma ** level
        elif config.depth_scheme == DepthWeightScheme.LINEAR:
            weight = (level + 1) / (L + 1)
            return min(1.0, weight) if config.lin_clip else weight
        else:  # Logistic
            alpha = 2.0  # Default parameter
            k = L / 2    # Default parameter
            return 1 / (1 + np.exp(alpha * (k - level)))

    def _popularity_weight(self, node: Node, config: ScorerConfig) -> float:
        """Compute the popularity penalty weight."""
        if not config.popularity:
            return 1.0
        idf = np.log(1 + self.n_docs / len(node.doc_counts))
        return idf ** config.beta

    def _score_node(
        self,
        node: Node,
        scores: Dict[int, float],
        seen_docs: Set[int],
        config: ScorerConfig,
        child_doc_counts: Optional[Dict[int, int]] = None
    ) -> None:
        """
        Score a single node based on its depth and remaining document counts.
        
        The scoring process:
        1. Calculate depth-based weight for this node
        2. For each document in this node:
           - Subtract counts already seen in children
           - Score remaining counts based on node's depth
           - Apply popularity penalty if configured
        3. Add contributions to final scores
        
        Args:
            node: Current node being scored
            scores: Accumulated scores for each document
            seen_docs: Set of documents already processed
            config: Scoring configuration parameters
            child_doc_counts: Document counts from child nodes to subtract
        """
        # Calculate weight based on this node's depth
        w_depth = self._depth_weight(node.depth, node.depth, config)
        if w_depth < config.weight_floor:
            return
            
        # Process each document in this node
        remaining_counts = {}
        for doc_id, cnt in node.doc_counts.items():
            # Skip if we've already processed this document
            if doc_id in seen_docs:
                continue
                
            # Calculate how many vectors of this document remain after subtracting child counts
            remaining = cnt
            if child_doc_counts and doc_id in child_doc_counts:
                remaining -= child_doc_counts[doc_id]
                
            # Only score if there are remaining vectors
            if remaining > 0:
                remaining_counts[doc_id] = remaining
                
        # Score remaining documents
        for doc_id, remaining in remaining_counts.items():
            # Calculate contribution: weight * (remaining vectors / total vectors)
            contribution = w_depth * remaining / len(node.vector_ids)
            
            # Apply popularity penalty if configured
            contribution *= self._popularity_weight(node, config)
            
            # Add to accumulated scores
            scores[doc_id] = scores.get(doc_id, 0.0) + contribution
            
            # Mark this document as processed
            seen_docs.add(doc_id)

    def _score_tree(self, query_vec: np.ndarray, root: Node, scores: Dict[int, float], config: ScorerConfig) -> Dict[int, float]:
        """
        Score a single tree using bottom-up traversal.
        
        Process:
        1. Find the leaf node for the query vector
        2. Start from leaf and work up to root
        3. At each node:
           - Score remaining document counts
           - Pass current node's counts up to parent
        4. Skip root node if configured
        
        Args:
            query_vec: Query vector to score against
            root: Root node of the tree
            scores: Accumulated scores for each document
            config: Scoring configuration parameters
            
        Returns:
            Updated scores dictionary
        """
        # Find the leaf node by traversing down the tree
        current = root
        path = []
        while current is not None:
            path.append(current)
            if current.hash_func is None:  # Leaf node
                break
            if current.hash_func(query_vec) == 0:
                current = current.left
            else:
                current = current.right
                
        if not path:
            return scores
            
        # Track which documents we've already processed
        seen_docs: Set[int] = set()
        
        # Start from leaf and work up to root
        child_doc_counts = None
        for node in reversed(path):
            # Skip root node if configured
            if node is root and config.skip_root:
                break
                
            # Score this node based on remaining document counts
            self._score_node(node, scores, seen_docs, config, child_doc_counts)
            
            # Store this node's counts for parent to subtract
            child_doc_counts = node.doc_counts.copy()
        
        return scores

    def query(self, query_vectors: List[np.ndarray], config: ScorerConfig) -> Dict[int, float]:
        """
        Query the forest with a set of query vectors and return scores for each document.
        """
        # Initialize scores for each document
        doc_scores = {doc_id: 0.0 for doc_id in range(self.n_docs)}
        total_contributions = 0
        
        print(f"\nDebug: Processing {len(query_vectors)} query vectors against {len(self.roots)} trees")
        
        # Process each query vector
        for q_idx, q_vec in enumerate(query_vectors):
            print(f"\nDebug: Processing query vector {q_idx + 1}/{len(query_vectors)}")
            query_scores = {doc_id: 0.0 for doc_id in range(self.n_docs)}
            
            # Score each tree
            for t_idx, root in enumerate(self.roots):
                tree_scores = self._score_tree(q_vec, root, doc_scores.copy(), config)
                
                # Normalize tree scores
                if tree_scores:
                    max_score = max(tree_scores.values())
                    if max_score > 0:
                        for doc_id in tree_scores:
                            tree_scores[doc_id] /= max_score
                
                # Add to query scores
                for doc_id, score in tree_scores.items():
                    query_scores[doc_id] += score
                
                print(f"Debug: Tree {t_idx + 1}/{len(self.roots)} - Max score: {max(tree_scores.values()):.4f}")
            
            # Normalize query scores
            if query_scores:
                max_query_score = max(query_scores.values())
                if max_query_score > 0:
                    for doc_id in query_scores:
                        query_scores[doc_id] /= max_query_score
            
            # Update running average
            total_contributions += 1
            for doc_id in doc_scores:
                doc_scores[doc_id] = (doc_scores[doc_id] * (total_contributions - 1) + query_scores[doc_id]) / total_contributions
        
        print(f"\nDebug: Final scores after {total_contributions} contributions")
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"Debug: Doc {doc_id} final score: {score:.4f}")
        
        return doc_scores

    def get_top_k(self, query_vecs: List[np.ndarray], k: int, config: ScorerConfig) -> List[Tuple[int, float]]:
        """Get top-k documents for a query."""
        scores = self.query(query_vecs, config)
        return heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])


def analyze_forest(forest: ForestVote) -> Dict:
    """Analyze the forest structure and return diagnostic metrics."""
    metrics = {
        'leaf_counts': [],
        'depth_histograms': [],
        'max_docs_per_node': [],
        'trial_attempts_stats': []
    }
    
    for tree_nodes in forest.nodes:
        # Leaf count
        leaf_count = sum(1 for n in tree_nodes if len(n.vector_ids) == 1)
        metrics['leaf_counts'].append(leaf_count)
        
        # Depth histogram
        depth_hist = Counter(n.depth for n in tree_nodes)
        metrics['depth_histograms'].append(depth_hist)
        
        # Max docs per node
        max_docs = max(len(n.doc_counts) for n in tree_nodes)
        metrics['max_docs_per_node'].append(max_docs)
        
        # Trial attempts stats
        attempts = [n.trial_attempts for n in tree_nodes]
        metrics['trial_attempts_stats'].append({
            'mean': np.mean(attempts),
            'max': max(attempts)
        })
        
        # Verify invariants
        for node in tree_nodes:
            if node.left and node.right:
                # Check if child doc counts sum to parent
                left_docs = Counter(node.left.doc_counts)
                right_docs = Counter(node.right.doc_counts)
                combined = left_docs + right_docs
                assert combined == Counter(node.doc_counts), "Document count invariant violated"
                
                # Check if vector counts match
                assert len(node.vector_ids) == sum(node.doc_counts.values()), "Vector count invariant violated"
    
    return metrics 