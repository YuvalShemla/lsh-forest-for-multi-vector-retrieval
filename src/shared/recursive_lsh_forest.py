import random
from typing import List, Tuple, Optional, Callable, Sequence, Any
import numpy as np


class Node:
    """A binary tree node for the recursive LSH forest implementation."""
    
    def __init__(self, depth: int, parent: Optional['Node'] = None):
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.depth = depth
        self.parent = parent  # direct reference to parent node
        self.vector_ids: List[int] = []  # IDs of all vectors that passed through this node
        self.hash_func: Optional[Callable] = None  # The hash function for this node
        self.trial_attempts: int = 0  # Number of hash function attempts at this node


class RecursiveLSHForest:
    """
    A recursive implementation of LSH Forest that builds trees by recursively splitting
    vectors based on hash values.
    
    Parameters:
    -----------
    lsh_family : LSHFamily
        The LSH family to use for hashing
    l : int
        Number of trees in the forest
    km : int
        Maximum depth of any tree
    max_hash_attempts : int
        Maximum number of attempts to find a splitting hash function
    max_split_ratio : float
        Maximum allowed ratio between the sizes of the two groups after a split.
        For example, if max_split_ratio = 2.0, then the larger group can be at most
        twice the size of the smaller group.
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
        self.roots: List[Node] = []
        self.nodes: List[List[Node]] = []  # List of nodes for each tree
        
    def build_forest(self, vectors: Sequence[np.ndarray]):
        """Build the entire forest from a set of vectors."""
        self.data = list(vectors)
        self.roots = []
        self.nodes = []
        
        for _ in range(self.l):
            root = Node(0, parent=None)
            self.roots.append(root)
            tree_nodes = [root]  # Start with root node
            self.nodes.append(tree_nodes)
            self._build_tree(root, list(range(len(vectors))), tree_nodes)
            
    def _build_tree(
        self, 
        node: Node, 
        vector_indices: List[int],
        tree_nodes: List[Node]
    ):
        """
        Recursively build a tree by splitting vectors based on hash values.
        
        Parameters:
        -----------
        node : Node
            Current node being processed
        vector_indices : List[int]
            Indices of vectors to be processed at this node
        tree_nodes : List[Node]
            List of all nodes in the current tree
        """
        # Always store all vectors that pass through this node
        node.vector_ids = vector_indices.copy()
        
        # Base cases
        if len(vector_indices) <= 1 or node.depth >= self.km:
            return
            
        best_split = None
        best_ratio = float('inf')
        node.trial_attempts = 0  # Reset for this node
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
            return
        # If we couldn't find any valid split, just keep all vectors in this node
        
    def query(
        self,
        q: np.ndarray,
        m: int = 1,
        dist: Callable[[np.ndarray, np.ndarray], float] = None
    ) -> List[Tuple[int, float]]:
        """
        Query the forest for the m nearest neighbors of q.
        
        Parameters:
        -----------
        q : np.ndarray
            Query vector
        m : int
            Number of nearest neighbors to return
        dist : Callable
            Distance function to use (defaults to Hamming distance)
            
        Returns:
        --------
        List[Tuple[int, float]]
            List of (index, distance) pairs for the m nearest neighbors
        """
        if dist is None:
            dist = lambda a, b: np.count_nonzero(a != b)  # Hamming distance
            
        candidates = set()
        
        # Query each tree
        for tree_idx, root in enumerate(self.roots):
            self._query_tree(root, q, candidates)
            
        # Rank and return m best
        scored = [(i, dist(q, self.data[i])) for i in candidates]
        scored.sort(key=lambda x: x[1])
        return scored[:m]
        
    def _query_tree(
        self,
        node: Node,
        q: np.ndarray,
        candidates: set
    ):
        """Recursively query a single tree."""
        if not node:
            return
            
        # Add vectors from this node
        candidates.update(node.vector_ids)
        
        # If we have children, continue traversal
        if node.left and node.right and node.hash_func:
            if node.hash_func(q) == 0:
                self._query_tree(node.left, q, candidates)
            else:
                self._query_tree(node.right, q, candidates) 