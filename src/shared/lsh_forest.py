import random
from typing import List, Tuple, Optional, Callable, Sequence, Any

import numpy as np


# 1.  LSH family abstractions
class LSHFamily:
    """Generic (r,cr,p1,p2)-sensitive family.
    `sample()` returns a hash function h: obj -> int/bytes."""

    def __init__(self, dim: int):
        self.dim = dim

    def sample(self) -> Callable[[np.ndarray], int]:
        raise NotImplementedError


class BitSamplingLSH(LSHFamily):
    """One-bit sampling for {0,1}^d   –  IM98 / HIM12."""

    def sample(self):
        bit = random.randrange(self.dim)
        return lambda x, b=bit: int(x[b])


# 2.  Trie nodes (LSH Trees)
class TrieNode:
    __slots__ = ("children", "pivot_ids", "depth", "parent")

    def __init__(self, depth: int, parent: Optional["TrieNode"]):
        self.children: dict[int, TrieNode] = {}
        self.pivot_ids: List[int] = []  # ids of objects kept as pivots
        self.depth = depth  # how many hash digits fixed so far
        self.parent = parent


# 3.  LSH Forest container
class LSHForest:
    """
    l   – number of trees
    k   – # pivots to cache per internal node (ddtrees §4, Alg. 2)
    km  – max label length (2005 paper, §5.1)
    """

    def __init__(self, lsh_family: LSHFamily, l: int = 10, k: int = 4, km: int = 64):
        self.lsh_family = lsh_family
        self.l = l
        self.k = k
        self.km = km
        # per-tree hash-function sequences (one per depth position)
        self.hash_seqs: List[List[Callable]] = [
            [lsh_family.sample() for _ in range(km)] for _ in range(l)
        ]
        self.roots: List[TrieNode] = [TrieNode(0, None) for _ in range(l)]
        self.data: List[np.ndarray] = []  # the points themselves

    def insert(self, vec: np.ndarray):
        idx = len(self.data)
        self.data.append(vec)
        for t in range(self.l):
            self._insert_into_tree(t, idx)

    def batch_insert(self, vecs: Sequence[np.ndarray]):
        for v in vecs:
            self.insert(v)

    def query(
        self,
        q: np.ndarray,
        m: int = 1,
        dist: Callable[[np.ndarray, np.ndarray], float] = None,
    ) -> List[Tuple[int, float]]:
        """Return the m nearest ids (and distances) using the synchronous two-phase
        DESCEND + SYNCHASCEND procedure :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}.
        """
        if dist is None:
            dist = lambda a, b: np.count_nonzero(a != b)  # Hamming
        # ---- top-down phase ----
        leaves, depths = [], []
        for t in range(self.l):
            node, depth = self._descend(t, q)
            leaves.append(node)
            depths.append(depth)
        # ---- bottom-up phase ----
        M = max(self.k * self.l, m)  # cf. paper’s M = cl or tuned dynamically
        cand_ids: set[int] = set()
        lvl = max(depths)
        while lvl >= 0 and (len(cand_ids) < M or len(cand_ids) < m):
            for ti in range(self.l):
                if depths[ti] == lvl:
                    cand_ids.update(leaves[ti].pivot_ids)
                    # move pointer one level up
                    leaves[ti] = leaves[ti].parent if leaves[ti].parent else leaves[ti]
                    depths[ti] -= 1
            lvl -= 1
        # rank and return m best
        scored = [(i, dist(q, self.data[i])) for i in cand_ids]
        scored.sort(key=lambda x: x[1])
        return scored[:m]

    def _insert_into_tree(self, t: int, idx: int):
        node = self.roots[t]
        depth = 0
        seq = self.hash_seqs[t]
        vec = self.data[idx]
        while True:
            # add pivots if useful
            if len(node.pivot_ids) < self.k and idx not in node.pivot_ids:
                # ensure “(c-1)r-separated” heuristic – we approximate via uniqueness.
                node.pivot_ids.append(idx)
            if depth == self.km:  # reached max label length
                return
            hval = seq[depth](vec)
            if hval not in node.children:
                node.children[hval] = TrieNode(depth + 1, node)
            node = node.children[hval]
            depth += 1

    def _descend(self, t: int, q: np.ndarray) -> Tuple[TrieNode, int]:
        """Figure 2 DESCEND algorithm :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}."""
        node = self.roots[t]
        depth = 0
        seq = self.hash_seqs[t]
        while depth < self.km and node.children:
            hval = seq[depth](q)
            if hval not in node.children:  # mismatch – best-match leaf reached
                break
            node = node.children[hval]
            depth += 1
        return node, depth


class MultiDocTrieNode(TrieNode):

    def __init__(self, depth: int, parent: Optional["TrieNode"]):
        super().__init__(depth, parent)
        self.pivot_ids: dict[List[int]] = {}  # document: (pivots, id)


class MultiDocLSHForest(LSHForest):

    def __init__(self, lsh_family: LSHFamily, l: int = 10, k: int = 4, km: int = 64):
        super().__init__(lsh_family, l, k, km)
        self.roots: List[MultiDocTrieNode] = [MultiDocTrieNode(0, None) for _ in range(l)]
        self.data: List[List[np.ndarray]] = (
            []
        )
        # (document, vectors, elements): the points themselves
    
    def num_docs(self):
        return len(self.data)

    def insert(self, vec: np.ndarray, doc: int):
        if len(self.data) == doc:
            self.data.append([vec])
        else:
            self.data[doc].append(vec)

        idx = len(self.data[doc]) - 1
        for tree in range(self.l):
            self._insert_into_tree(tree, doc, idx)

    def batch_insert(self, data: Sequence[np.ndarray]):
        for document, vecs in enumerate(data):
            for vec in vecs:
                self.insert(vec, document)

    def query(
        self,
        q: np.ndarray,
        m: int = 1,
        dist: Callable[[np.ndarray, np.ndarray], float] = None,
    ) -> List[List[Tuple[int, float]]]:
        """Return the m nearest ids (and distances) for each document using the synchronous two-phase
        DESCEND + SYNCHASCEND procedure.

        Returns a lists of m tuples grouped by document
        """
        if dist is None:
            dist = lambda a, b: np.count_nonzero(a != b)  # Hamming

        # ---- top-down phase ----
        leaves, depths = [], []
        for tree in range(self.l):
            node, depth = self._descend(tree, q)
            leaves.append(node)
            depths.append(depth)

        # ---- bottom-up phase ----
        remaining = {d for d in range(self.num_docs())}
        candidates = np.empty(self.num_docs(), dtype=object)
        for doc in range(len(candidates)):
            candidates[doc] = set()

        level = max(depths)
        while level >= 0 and remaining:
            for tree in range(self.l):
                if depths[tree] != level:
                    continue

                node = leaves[tree]

                # Add pivots from this level to documents that still need matches
                for doc in list(remaining):
                    need = m - len(candidates[doc])
                    new_pivots = node.pivot_ids.get(doc, [])[:need]
                    candidates[doc].update(new_pivots)
                    if len(candidates[doc]) >= m:
                        remaining.remove(doc)
                
                if not remaining:
                    break

                # move pointer one level up
                node = node.parent if node.parent else node
                depths[tree] -= 1
            level -= 1

        # rank and return m best
        results = []
        for doc, doc_candidates in enumerate(candidates):
            scored = [(i, dist(q, self.data[doc][i])) for i in doc_candidates]
            scored.sort(key=lambda x: x[1])
            results.append(scored[:m])
        return results

    def _insert_into_tree(self, t: int, doc: int, idx: int):
        node = self.roots[t]
        depth = 0
        seq = self.hash_seqs[t]
        vec = self.data[doc][idx]
        while True:
            # add pivots
            if doc not in node.pivot_ids:
                node.pivot_ids[doc] = [idx]
            elif len(node.pivot_ids[doc]) < self.k:
                node.pivot_ids[doc].append(idx)

            if depth == self.km:  # reached max label length
                return

            hash_val = seq[depth](vec)
            if hash_val not in node.children:
                node.children[hash_val] = MultiDocTrieNode(depth + 1, node)
            node = node.children[hash_val]
            depth += 1
