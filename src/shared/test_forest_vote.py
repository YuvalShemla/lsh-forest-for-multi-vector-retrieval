"""
Test script for the Forest Vote implementation.
This script tests the performance of the Forest Vote algorithm on synthetic data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from forest_vote import ForestVote, ScorerConfig, DepthWeightScheme
from lsh_family import LSHFamily


@dataclass
class TestConfig:
    """Configuration parameters for the test."""
    n_docs: int = 100
    vectors_per_doc: int = 10
    vector_dim: int = 4
    noise_std: float = 0.01
    n_trees: int = 5
    max_depth: int = 15
    max_hash_attempts: int = 1000
    max_split_ratio: float = 2.0
    results_dir: str = "results"


def generate_document_vectors(
    n_docs: int,
    vectors_per_doc: int,
    vector_dim: int,
    noise_std: float
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Generate synthetic document vectors.
    
    Args:
        n_docs: Number of documents to generate
        vectors_per_doc: Number of vectors per document
        vector_dim: Dimension of each vector
        noise_std: Standard deviation of noise to add
        
    Returns:
        Tuple of (vectors, doc_ids) where vectors is a list of normalized vectors
        and doc_ids is a list of document IDs corresponding to each vector
    """
    # Generate document centers
    doc_centers = [np.random.randn(vector_dim) for _ in range(n_docs)]
    doc_centers = [center / np.linalg.norm(center) for center in doc_centers]
    
    # Generate vectors for each document
    vectors = []
    doc_ids = []
    
    for doc_id, center in enumerate(doc_centers):
        for _ in range(vectors_per_doc):
            vector = center + np.random.normal(0, noise_std, vector_dim)
            vector = vector / np.linalg.norm(vector)
            vectors.append(vector)
            doc_ids.append(doc_id)
            
    return vectors, doc_ids


def generate_query_document(
    vector_dim: int,
    n_vectors: int,
    noise_std: float
) -> List[np.ndarray]:
    """
    Generate a query document with random center and vectors.
    
    Args:
        vector_dim: Dimension of each vector
        n_vectors: Number of vectors to generate
        noise_std: Standard deviation of noise to add
        
    Returns:
        List of normalized query vectors
    """
    center = np.random.randn(vector_dim)
    center = center / np.linalg.norm(center)
    
    query_vectors = []
    for _ in range(n_vectors):
        vector = center + np.random.normal(0, noise_std, vector_dim)
        vector = vector / np.linalg.norm(vector)
        query_vectors.append(vector)
        
    return query_vectors


def chamfer_distance(A: List[np.ndarray], B: List[np.ndarray]) -> float:
    """
    Compute the (one-sided) Chamfer distance from A to B.
    
    Args:
        A: List of source vectors
        B: List of target vectors
        
    Returns:
        Mean of minimum distances from each vector in A to B
    """
    dists = []
    for a in A:
        dists.append(np.min([np.linalg.norm(a - b) for b in B]))
    return np.mean(dists)


def calculate_ground_truth(
    vectors: List[np.ndarray],
    doc_ids: List[int],
    query_vectors: List[np.ndarray]
) -> Tuple[List[float], List[float], int]:
    """
    Calculate ground truth similarities and Chamfer distances.
    
    Args:
        vectors: List of all document vectors
        doc_ids: Document IDs corresponding to vectors
        query_vectors: List of query vectors
        
    Returns:
        Tuple of (true_similarities, chamfer_dists, true_doc_id)
    """
    n_docs = max(doc_ids) + 1
    true_similarities = []
    chamfer_dists = []
    
    for doc_id in range(n_docs):
        doc_vectors = [v for v, d in zip(vectors, doc_ids) if d == doc_id]
        similarities = []
        for q_vec in query_vectors:
            for d_vec in doc_vectors:
                sim = np.dot(q_vec, d_vec)
                similarities.append(sim)
        true_similarities.append(np.mean(similarities))
        chamfer_dists.append(chamfer_distance(query_vectors, doc_vectors))
    
    true_doc_id = np.argmax(true_similarities)
    return true_similarities, chamfer_dists, true_doc_id


def analyze_results(
    true_doc_id: int,
    scores: Dict[int, float],
    vectors: List[np.ndarray],
    doc_ids: List[int],
    query_vectors: List[np.ndarray],
    config: ScorerConfig,
    results_dir: str = "results"
) -> Dict:
    """
    Analyze and visualize the results.
    
    Args:
        true_doc_id: ID of the true matching document
        scores: Dictionary of document scores
        vectors: List of all document vectors
        doc_ids: Document IDs corresponding to vectors
        query_vectors: List of query vectors
        config: Scoring configuration
        results_dir: Directory to save results
        
    Returns:
        Dictionary containing analysis metrics
    """
    # Ensure results directory exists
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert scores to DataFrame
    score_df = pd.DataFrame([
        {'doc_id': doc_id, 'score': score}
        for doc_id, score in scores.items()
    ])
    
    # Calculate ground truth
    true_similarities, chamfer_dists, _ = calculate_ground_truth(vectors, doc_ids, query_vectors)
    
    # Add ground truth to DataFrame
    score_df['true_similarity'] = [true_similarities[d] for d in score_df['doc_id']]
    score_df['chamfer'] = [chamfer_dists[d] for d in score_df['doc_id']]
    score_df['is_true_doc'] = score_df['doc_id'] == true_doc_id
    
    # Compute ranks
    score_df['chamfer_rank'] = score_df['chamfer'].rank(method='min')
    score_df['score_rank'] = score_df['score'].rank(ascending=False, method='min')
    
    # Save results
    csv_filename = f"lsh_results_g{config.gamma}_d{config.depth_scheme.value}_p{int(config.popularity)}_b{config.beta}.csv"
    score_df.to_csv(os.path.join(results_dir, csv_filename), index=False)
    
    # Create visualizations
    create_analysis_plots(score_df, config, results_dir)
    create_chamfer_plot(score_df, config, results_dir)
    
    # Calculate metrics
    metrics = calculate_metrics(score_df, true_doc_id)
    metrics['score_df'] = score_df
    
    return metrics


def create_analysis_plots(score_df: pd.DataFrame, config: ScorerConfig, results_dir: str) -> None:
    """Create and save the main analysis plots."""
    plt.figure(figsize=(18, 12))
    
    # 1. Score vs True Similarity
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=score_df, x='true_similarity', y='score', hue='is_true_doc', legend='brief')
    plt.title('Forest Score vs True Similarity')
    plt.xlabel('True Cosine Similarity')
    plt.ylabel('Forest Score')
    
    # 2. Score Distribution
    plt.subplot(2, 2, 2)
    sns.histplot(data=score_df, x='score', hue='is_true_doc', bins=20, legend='brief')
    plt.title('Score Distribution')
    plt.xlabel('Forest Score')
    
    # 3. Chamfer Distance Distribution
    plt.subplot(2, 2, 3)
    sns.histplot(data=score_df, x='chamfer', hue='is_true_doc', bins=20, legend='brief')
    plt.title('Chamfer Distance Distribution')
    plt.xlabel('Chamfer Distance (lower is better)')
    
    # 4. Score vs Chamfer (sorted)
    plt.subplot(2, 2, 4)
    score_df_sorted = score_df.sort_values('chamfer')
    plt.plot(score_df_sorted['chamfer'], score_df_sorted['score'], 'o-')
    plt.title('Sorted Scores vs Chamfer Distances')
    plt.xlabel('Chamfer Distance')
    plt.ylabel('Forest Score')
    
    plt.tight_layout()
    analysis_plot_filename = f"lsh_analysis_g{config.gamma}_d{config.depth_scheme.value}_p{int(config.popularity)}_b{config.beta}.png"
    plt.savefig(os.path.join(results_dir, analysis_plot_filename))
    plt.close()


def create_chamfer_plot(score_df: pd.DataFrame, config: ScorerConfig, results_dir: str) -> None:
    """Create and save the Chamfer distance plot."""
    # Calculate correlations
    score_df['log_score'] = np.log1p(score_df['score'])
    score_chamfer_corr = score_df['log_score'].corr(score_df['chamfer'])
    score_similarity_corr = score_df['score'].corr(score_df['true_similarity'])
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=score_df, x='chamfer', y='log_score', label=f'Points (n={len(score_df)})')
    sns.regplot(data=score_df, x='chamfer', y='log_score', scatter=False, color='red', 
                label=f'Best-fit line (r={score_chamfer_corr:.4f})')
    plt.title(f'Chamfer Distance vs Log(Forest Score)\n'
              f'gamma={config.gamma}, depth_scheme={config.depth_scheme.value}\n'
              f'Score-Similarity Corr: {score_similarity_corr:.4f}')
    plt.xlabel('Chamfer Distance')
    plt.ylabel('Log(Forest Score)')
    plt.legend()
    
    chamfer_plot_filename = f"chamfer_vs_score_g{config.gamma}_d{config.depth_scheme.value}_p{int(config.popularity)}_b{config.beta}.png"
    plt.savefig(os.path.join(results_dir, chamfer_plot_filename))
    plt.close()


def calculate_metrics(score_df: pd.DataFrame, true_doc_id: int) -> Dict:
    """Calculate and return analysis metrics."""
    metrics = {}
    if true_doc_id in score_df['doc_id'].values:
        metrics['true_doc_rank'] = score_df[score_df['is_true_doc']].index[0] + 1
        metrics['true_doc_score'] = score_df[score_df['is_true_doc']]['score'].iloc[0]
        metrics['top_k_accuracy'] = {
            k: true_doc_id in score_df.nlargest(k, 'score')['doc_id'].values
            for k in [1, 3, 5]
        }
    else:
        metrics['true_doc_rank'] = None
        metrics['true_doc_score'] = None
        metrics['top_k_accuracy'] = {k: False for k in [1, 3, 5]}
    
    metrics['score_correlation'] = score_df['score'].corr(score_df['true_similarity'])
    return metrics


def main():
    """Main function to run the test."""
    # Load configuration
    config = TestConfig()
    
    # Generate data
    print("Generating data...")
    vectors, doc_ids = generate_document_vectors(
        n_docs=config.n_docs,
        vectors_per_doc=config.vectors_per_doc,
        vector_dim=config.vector_dim,
        noise_std=config.noise_std
    )
    
    # Generate query
    print("Generating query...")
    query_vectors = generate_query_document(
        vector_dim=config.vector_dim,
        n_vectors=10,
        noise_std=config.noise_std
    )
    
    # Create and build forest
    print("Building forest...")
    lsh_family = LSHFamily(config.vector_dim)
    forest = ForestVote(
        lsh_family=lsh_family,
        l=config.n_trees,
        km=config.max_depth,
        max_hash_attempts=config.max_hash_attempts,
        max_split_ratio=config.max_split_ratio
    )
    forest.build_forest(vectors, doc_ids)
    
    # Run analysis
    print("Running analysis...")
    scorer_config = ScorerConfig(
        depth_scheme=DepthWeightScheme.LINEAR,
        gamma=0.3, 
        popularity=True,
        beta=0.7
    )
    
    # Find true document ID
    _, _, true_doc_id = calculate_ground_truth(vectors, doc_ids, query_vectors)
    
    # Query forest
    scores = forest.query(query_vectors, scorer_config)
    
    # Analyze results
    metrics = analyze_results(
        true_doc_id, 
        scores, 
        vectors, 
        doc_ids, 
        query_vectors, 
        scorer_config, 
        results_dir=config.results_dir
    )
    
    # Print metrics
    print("\nResults:")
    print(f"True document ID: {true_doc_id}")
    print(f"True document rank: {metrics['true_doc_rank']}")
    print(f"True document score: {metrics['true_doc_score']:.4f}")
    print(f"Score correlation with true similarity: {metrics['score_correlation']:.4f}")
    print("\nTop-k accuracy:")
    for k, is_correct in metrics['top_k_accuracy'].items():
        print(f"Top-{k}: {'✓' if is_correct else '✗'}")


if __name__ == "__main__":
    main() 