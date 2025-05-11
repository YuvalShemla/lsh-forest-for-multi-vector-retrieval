
import numpy as np

def collect_matches(forests, queries, distance, a):
        multi = not isinstance(forests, list)

        n = forests.num_docs() if multi else len(forests)
        q = len(queries)
        d = len(queries[0])
        matches = np.empty((n, q, d), dtype=np.float32)
        if multi:
                for i, query in enumerate(queries):
                        results = forests.query(query, a, dist=distance)
                        for document, result in enumerate(results):
                                idx, _ = result[0]
                                matches[document, i] = forests.data[document][idx]
        else:
                for document, forest in enumerate(forests):
                        for i, query in enumerate(queries):
                                idx = forest.query(query, a, dist=distance)[0][0]
                                matches[document, i] = forest.data[idx] 

        return matches

        

def sim_scores(matches: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """
    Dot-product similarity for every document.
    Returns (D,) vector of scores.
    """
    return np.tensordot(matches, queries, axes=([1, 2], [0, 1]))
        
def rank_documents(forests, queries, dist, a=1) -> list[(int, float)]:
        scores = sim_scores(collect_matches(forests, queries, dist, a), queries)
        order  = np.argsort(-scores)
        return order

def best_document(forests, queries, dist, a=1) -> tuple[float, int]:
    """
    Convenience wrapper that returns only the best matching document.
    """
    scores = sim_scores(collect_matches(forests, queries, dist, a), queries)
    best   = int(np.argmax(scores))
    return float(scores[best]), best
