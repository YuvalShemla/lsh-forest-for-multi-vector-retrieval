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
    # ScorerConfig parameters
    gamma: float = 0.3
    depth_scheme: DepthWeightScheme = DepthWeightScheme.LINEAR
    popularity: bool = True
    beta: float = 0.7
    lin_clip: bool = True
    skip_root: bool = True
    weight_floor: float = 1e-6

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


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
    # Calculate correlations using raw scores
    score_chamfer_corr = score_df['score'].corr(score_df['chamfer'])
    score_similarity_corr = score_df['score'].corr(score_df['true_similarity'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Raw scores vs Chamfer
    sns.scatterplot(data=score_df, x='chamfer', y='score', ax=ax1, label=f'Points (n={len(score_df)})')
    sns.regplot(data=score_df, x='chamfer', y='score', scatter=False, color='red', ax=ax1,
                label=f'Best-fit line (r={score_chamfer_corr:.4f})')
    ax1.set_title('Raw Scores vs Chamfer Distance')
    ax1.set_xlabel('Chamfer Distance')
    ax1.set_ylabel('Raw Score')
    ax1.legend()
    
    # Plot 2: Log scores vs Chamfer
    score_df['log_score'] = np.log1p(score_df['score'])
    log_score_chamfer_corr = score_df['log_score'].corr(score_df['chamfer'])
    
    sns.scatterplot(data=score_df, x='chamfer', y='log_score', ax=ax2, label=f'Points (n={len(score_df)})')
    sns.regplot(data=score_df, x='chamfer', y='log_score', scatter=False, color='red', ax=ax2,
                label=f'Best-fit line (r={log_score_chamfer_corr:.4f})')
    ax2.set_title('Log Scores vs Chamfer Distance')
    ax2.set_xlabel('Chamfer Distance')
    ax2.set_ylabel('Log(Score + 1)')
    ax2.legend()
    
    # Add explanation of correlation difference
    correlation_explanation = (
        "Note: The correlation coefficient (r) shown is Pearson's correlation, "
        "while the best-fit line uses linear regression. They may differ because:\n"
        "1. Pearson's correlation measures linear relationship strength\n"
        "2. The best-fit line minimizes squared errors\n"
        "3. Outliers can affect them differently"
    )
    
    plt.suptitle(f'Score Analysis (Vectors per Doc: {config.vectors_per_doc})\n'
                 f'gamma={config.gamma}, depth_scheme={config.depth_scheme.value}\n'
                 f'Raw Score-Similarity Corr: {score_similarity_corr:.4f}\n'
                 f'{correlation_explanation}')
    plt.tight_layout()
    
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


def run_experiments(base_config: TestConfig) -> None:
    """
    Run experiments with different parameter combinations.
    
    Args:
        base_config: Base configuration to use for all experiments
    """
    # Parameter ranges to sweep
    depth_schemes = [
        DepthWeightScheme.LINEAR,
        DepthWeightScheme.LOGISTIC,
        DepthWeightScheme.EXPONENTIAL
    ]
    gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    beta_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    popularity_values = [True, False]
    max_depth_values = [10, 12, 15, 18, 20]
    
    # Create results directory for experiments
    experiments_dir = os.path.join(base_config.results_dir, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Generate data once for all experiments
    print("Generating data for all experiments...")
    vectors, doc_ids = generate_document_vectors(
        n_docs=base_config.n_docs,
        vectors_per_doc=base_config.vectors_per_doc,
        vector_dim=base_config.vector_dim,
        noise_std=base_config.noise_std
    )
    
    # Generate query once for all experiments
    print("Generating query for all experiments...")
    query_vectors = generate_query_document(
        vector_dim=base_config.vector_dim,
        n_vectors=base_config.vectors_per_doc,
        noise_std=base_config.noise_std
    )
    
    # Calculate ground truth once
    _, _, true_doc_id = calculate_ground_truth(vectors, doc_ids, query_vectors)
    
    # Create LSH family once
    lsh_family = LSHFamily(base_config.vector_dim)
    
    # Track experiment results
    experiment_results = []
    
    # Run experiments
    total_experiments = (len(depth_schemes) * len(gamma_values) * len(beta_values) * 
                        len(popularity_values) * len(max_depth_values))
    experiment_count = 0
    
    for depth_scheme in depth_schemes:
        for gamma in gamma_values:
            for beta in beta_values:
                for popularity in popularity_values:
                    for max_depth in max_depth_values:
                        experiment_count += 1
                        print(f"\nRunning experiment {experiment_count}/{total_experiments}")
                        print(f"Parameters: depth_scheme={depth_scheme.value}, gamma={gamma}, "
                              f"beta={beta}, popularity={popularity}, max_depth={max_depth}")
                        
                        # Create experiment-specific config
                        exp_config = TestConfig(
                            **{k: v for k, v in base_config.__dict__.items() 
                               if k not in ['depth_scheme', 'gamma', 'beta', 'popularity', 'max_depth']},
                            depth_scheme=depth_scheme,
                            gamma=gamma,
                            beta=beta,
                            popularity=popularity,
                            max_depth=max_depth,
                            results_dir=experiments_dir
                        )
                        
                        # Create scorer config
                        scorer_config = ScorerConfig(
                            depth_scheme=depth_scheme,
                            gamma=gamma,
                            beta=beta,
                            popularity=popularity,
                            lin_clip=base_config.lin_clip,
                            skip_root=base_config.skip_root,
                            weight_floor=base_config.weight_floor
                        )
                        
                        # Create and build forest
                        forest = ForestVote(
                            lsh_family=lsh_family,
                            l=base_config.n_trees,
                            km=max_depth,
                            max_hash_attempts=base_config.max_hash_attempts,
                            max_split_ratio=base_config.max_split_ratio
                        )
                        forest.build_forest(vectors, doc_ids)
                        
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
                            results_dir=experiments_dir
                        )
                        
                        # Store experiment results
                        experiment_results.append({
                            'depth_scheme': depth_scheme.value,
                            'gamma': gamma,
                            'beta': beta,
                            'popularity': popularity,
                            'max_depth': max_depth,
                            'true_doc_rank': metrics['true_doc_rank'],
                            'true_doc_score': metrics['true_doc_score'],
                            'score_correlation': metrics['score_correlation'],
                            'top_1_accuracy': metrics['top_k_accuracy'][1],
                            'top_3_accuracy': metrics['top_k_accuracy'][3],
                            'top_5_accuracy': metrics['top_k_accuracy'][5]
                        })
    
    # Save all experiment results to CSV
    results_df = pd.DataFrame(experiment_results)
    results_df.to_csv(os.path.join(experiments_dir, 'all_experiment_results.csv'), index=False)
    
    # Create summary plots
    create_experiment_summary_plots(results_df, experiments_dir)
    
    print("\nAll experiments completed!")
    print(f"Results saved in: {experiments_dir}")


def create_experiment_summary_plots(results_df: pd.DataFrame, results_dir: str) -> None:
    """Create summary plots for all experiments."""
    plt.figure(figsize=(15, 10))
    
    # 1. Top-1 Accuracy by Depth Scheme and Popularity
    plt.subplot(2, 2, 1)
    sns.boxplot(data=results_df, x='depth_scheme', y='top_1_accuracy', hue='popularity')
    plt.title('Top-1 Accuracy by Depth Scheme and Popularity')
    plt.xticks(rotation=45)
    
    # 2. Score Correlation by Gamma and Beta
    plt.subplot(2, 2, 2)
    pivot_corr = results_df.pivot_table(
        values='score_correlation',
        index='gamma',
        columns='beta',
        aggfunc='mean'
    )
    sns.heatmap(pivot_corr, annot=True, cmap='YlOrRd')
    plt.title('Average Score Correlation by Gamma and Beta')
    
    # 3. True Document Rank by Max Depth
    plt.subplot(2, 2, 3)
    sns.boxplot(data=results_df, x='max_depth', y='true_doc_rank')
    plt.title('True Document Rank by Max Depth')
    
    # 4. Top-k Accuracy Distribution
    plt.subplot(2, 2, 4)
    accuracy_data = pd.melt(
        results_df,
        value_vars=['top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy'],
        var_name='k',
        value_name='accuracy'
    )
    sns.boxplot(data=accuracy_data, x='k', y='accuracy')
    plt.title('Top-k Accuracy Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'experiment_summary.png'))
    plt.close()


def main():
    """Main function to run the test."""
    # Load configuration with custom values
    config = TestConfig(
        n_docs=1000,
        vectors_per_doc=50,
        vector_dim=5,
        noise_std=0.02,
        n_trees=10,
        max_depth=15,
        max_hash_attempts=100,
        max_split_ratio=2.5,
        results_dir="test_results",
        gamma=0.3,
        depth_scheme=DepthWeightScheme.LINEAR,
        popularity=True,
        beta=0.8,
        lin_clip=False,
        skip_root=True,
        weight_floor=1e-5
    )
    
    # Run experiments
    run_experiments(config)


if __name__ == "__main__":
    main() 