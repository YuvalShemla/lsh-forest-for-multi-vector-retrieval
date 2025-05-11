import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple
from collections import defaultdict
from recursive_lsh_forest import RecursiveLSHForest, Node
from lsh_forest import BitSamplingLSH, RandomHyperplaneLSH
import os


def generate_vectors(
    n_vectors: int,
    dim: int,
    cluster_centers: int = 5,
    cluster_std: float = 1,
    binary: bool = False
) -> np.ndarray:
    """
    Generate synthetic vectors with tunable parameters.
    
    Parameters:
    -----------
    n_vectors : int
        Number of vectors to generate
    dim : int
        Dimension of each vector
    cluster_centers : int
        Number of cluster centers to generate
    cluster_std : float
        Standard deviation of clusters
    binary : bool
        If True, generate binary vectors
    """
    # Generate cluster centers
    centers = np.random.randn(cluster_centers, dim)
    
    # Assign vectors to clusters
    cluster_sizes = np.random.multinomial(n_vectors, [1/cluster_centers] * cluster_centers)
    vectors = []
    
    for i, size in enumerate(cluster_sizes):
        cluster_vectors = centers[i] + np.random.randn(size, dim) * cluster_std
        vectors.append(cluster_vectors)
    
    vectors = np.vstack(vectors)
    
    if binary:
        vectors = (vectors > 0).astype(int)
    
    return vectors


class ForestAnalyzer:
    """Analyzer for LSH Forest structure and statistics."""
    
    def __init__(self, n_vectors=100, n_trees=1, max_depth=10, cluster_centers=15, cluster_std=0.5, max_split_ratio=1.2, max_hash_attempts=1000, dim=64):
        self.n_vectors = n_vectors
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.cluster_centers = cluster_centers
        self.cluster_std = cluster_std
        self.max_split_ratio = max_split_ratio
        self.max_hash_attempts = max_hash_attempts
        self.dim = dim
        self.vectors = self._generate_vectors()
        self.forest = self._build_forest()
        
    def _generate_vectors(self):
        """Generate synthetic vectors for testing."""
        return generate_vectors(
            n_vectors=self.n_vectors,
            dim=self.dim,
            cluster_centers=self.cluster_centers,
            cluster_std=self.cluster_std,
            binary=True
        )
        
    def _build_forest(self):
        """Build the LSH forest with the generated vectors."""
        lsh_family = RandomHyperplaneLSH(dim=self.dim)
        forest = RecursiveLSHForest(
            lsh_family=lsh_family,
            l=self.n_trees,
            km=self.max_depth,
            max_split_ratio=self.max_split_ratio,
            max_hash_attempts=self.max_hash_attempts
        )
        forest.build_forest(self.vectors)
        return forest
    
    def get_tree_stats(self, tree_idx: int = 0) -> Dict:
        """Get statistics for a single tree."""
        root = self.forest.roots[tree_idx]
        stats = {
            'total_nodes': 0,
            'leaf_nodes': 0,
            'max_depth': 0,
            'node_sizes': [],  # List of (depth, size) tuples
            'leaf_sizes': [],  # List of leaf node sizes
            'depth_distribution': defaultdict(int)  # Count of nodes at each depth
        }
        
        def traverse(node: Node, depth: int):
            if not node:
                return
                
            stats['total_nodes'] += 1
            stats['max_depth'] = max(stats['max_depth'], depth)
            stats['depth_distribution'][depth] += 1
            stats['node_sizes'].append((depth, len(node.vector_ids)))
            
            if not node.left and not node.right:
                stats['leaf_nodes'] += 1
                stats['leaf_sizes'].append(len(node.vector_ids))
            else:
                traverse(node.left, depth + 1)
                traverse(node.right, depth + 1)
                
        traverse(root, 0)
        return stats
    
    def plot_tree(self, tree_idx: int = 0, figsize: Tuple[int, int] = (30, 15), save=True):
        """Plot the tree structure with node sizes, using graphviz_layout for better spacing. Optionally save the plot."""
        G = nx.DiGraph()
        root = self.forest.roots[tree_idx]
        n_vectors = self.n_vectors

        def add_nodes(node):
            if not node:
                return
            node_label = f"{len(node.passed_vectors)}"
            G.add_node(
                id(node),
                size=len(node.vector_ids),
                passed=len(node.passed_vectors),
                label=node_label
            )
            if node.left:
                G.add_edge(id(node), id(node.left))
                add_nodes(node.left)
            if node.right:
                G.add_edge(id(node), id(node.right))
                add_nodes(node.right)
        add_nodes(root)

        # Use graphviz_layout for better tree layout
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB')

        # Create the plot with a much larger figure size
        fig, ax = plt.subplots(figsize=figsize)
        sizes = [G.nodes[n]['size'] for n in G.nodes()]
        passed = [G.nodes[n]['passed'] for n in G.nodes()]
        labels = {n: G.nodes[n]['label'] for n in G.nodes()}

        # Create a scatter plot for the nodes
        scatter = ax.scatter(
            [pos[n][0] for n in G.nodes()],
            [pos[n][1] for n in G.nodes()],
            c=passed,
            s=[(s + 1) * 500 for s in sizes],
            cmap=plt.cm.viridis,
            alpha=0.7
        )
        # Draw the edges
        for edge in G.edges():
            start_pos = pos[edge[0]]
            end_pos = pos[edge[1]]
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                   'k-', alpha=0.3, linewidth=1)
        # Add labels (just the number, in bold)
        for node in G.nodes():
            ax.text(pos[node][0], pos[node][1], labels[node],
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=10,
                   fontweight='bold')
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Number of passed vectors')
        # Remove axes
        ax.set_axis_off()
        plt.title(f"Tree Structure (n={n_vectors})\nNode label shows number of passed vectors", 
                 pad=20, fontsize=12)
        # Save the plot
        if save:
            results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))
            os.makedirs(results_dir, exist_ok=True)
            plt.savefig(os.path.join(results_dir, f'tree_plot_n{n_vectors}.png'), bbox_inches='tight')
        plt.show()
        
    def plot_statistics(self, tree_idx: int = 0, save=True):
        """Plot various statistics about the tree. Optionally save the plot."""
        stats = self.get_tree_stats(tree_idx)
        root = self.forest.roots[tree_idx]
        nodes = self.forest.nodes[tree_idx]
        n_vectors = self.n_vectors
        
        # Gather additional statistics
        leaf_depths = []
        def traverse(node, depth):
            if not node:
                return
            if not node.left and not node.right:
                leaf_depths.append(depth)
            traverse(node.left, depth + 1)
            traverse(node.right, depth + 1)
        traverse(root, 0)
        
        # Prepare data for average node size plots
        depth_sizes = defaultdict(list)
        depth_trials = defaultdict(list)
        all_trials = []
        for depth, size in stats['node_sizes']:
            depth_sizes[depth].append(size)
        for node in nodes:
            depth_trials[node.depth].append(node.trial_attempts)
            all_trials.append(node.trial_attempts)
        avg_sizes = [np.mean(depth_sizes[d]) for d in range(stats['max_depth'] + 1)]
        avg_trials = [np.mean(depth_trials[d]) if depth_trials[d] else 0 for d in range(stats['max_depth'] + 1)]
        
        # Create subplots (2x2 grid)
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])  # Leaf size distribution
        ax2 = fig.add_subplot(gs[0, 1])  # Number of nodes at each depth
        ax3 = fig.add_subplot(gs[1, 0])  # Average node size by depth
        ax4 = fig.add_subplot(gs[1, 1])  # Average trial attempts by depth
        
        # 1. Leaf size distribution
        ax1.hist(stats['leaf_sizes'], bins=30, color='lightgreen', edgecolor='black')
        ax1.set_title('Distribution of Leaf Node Sizes', pad=20, fontsize=12)
        ax1.set_xlabel('Number of Vectors', fontsize=10)
        ax1.set_ylabel('Count', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Number of nodes at each depth
        depths = list(stats['depth_distribution'].keys())
        counts = list(stats['depth_distribution'].values())
        ax2.bar(depths, counts, color='salmon', edgecolor='black')
        ax2.set_title('Number of Nodes at Each Depth', pad=20, fontsize=12)
        ax2.set_xlabel('Depth', fontsize=10)
        ax2.set_ylabel('Count', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Average node size by depth (all vectors)
        ax3.plot(range(stats['max_depth'] + 1), avg_sizes, 'o-', color='blue')
        ax3.set_title('Average Node Size by Depth', pad=20, fontsize=12)
        ax3.set_xlabel('Depth', fontsize=10)
        ax3.set_ylabel('Average Number of Vectors', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. Average trial attempts by depth
        ax4.plot(range(stats['max_depth'] + 1), avg_trials, 'd-', color='purple')
        ax4.set_title('Average Trial Attempts by Depth', pad=20, fontsize=12)
        ax4.set_xlabel('Depth', fontsize=10)
        ax4.set_ylabel('Average Trial Attempts', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'LSH Forest Statistics (n={n_vectors})', fontsize=16, y=0.98)
        # Save the plot
        if save:
            results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))
            os.makedirs(results_dir, exist_ok=True)
            plt.savefig(os.path.join(results_dir, f'statistics_plot_n{n_vectors}.png'), bbox_inches='tight')
        plt.show()

        # Separate histogram of trial attempts
        plt.figure(figsize=(8, 6))
        plt.hist(all_trials, bins=30, color='mediumpurple', edgecolor='black')
        plt.title('Histogram of Trial Attempts per Node', fontsize=14)
        plt.xlabel('Number of Trial Attempts', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.grid(True, alpha=0.3)
        if save:
            plt.savefig(os.path.join(results_dir, f'trial_attempts_hist_n{n_vectors}.png'), bbox_inches='tight')
        plt.show()
        
    def print_summary(self, tree_idx: int = 0):
        """Print a summary of the tree statistics."""
        stats = self.get_tree_stats(tree_idx)
        
        print(f"\nTree {tree_idx} Summary:")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Leaf nodes: {stats['leaf_nodes']}")
        print(f"Maximum depth: {stats['max_depth']}")
        print(f"Average leaf size: {np.mean(stats['leaf_sizes']):.2f}")
        print(f"Standard deviation of leaf sizes: {np.std(stats['leaf_sizes']):.2f}")
        print(f"Tree balance ratio: {stats['leaf_nodes'] / (2 ** stats['max_depth']):.2f}")
        
        # Calculate per-level statistics for both final and passed vectors
        depth_sizes = defaultdict(list)
        depth_passed = defaultdict(list)
        
        # First collect all node sizes
        for depth, size in stats['node_sizes']:
            depth_sizes[depth].append(size)
            
        # Then collect passed vectors for all nodes at each depth
        for node in self.forest.nodes[tree_idx]:
            depth_passed[node.depth].append(len(node.vector_ids))
            
        print("\nPer-Level Statistics:")
        print("Depth | Nodes | Final Avg | Final Std | Passed Avg | Passed Std")
        print("-" * 65)
        for depth in range(stats['max_depth'] + 1):
            sizes = depth_sizes[depth]
            passed = depth_passed[depth]
            if sizes:  # Only print if we have nodes at this depth
                print(f"{depth:5d} | {len(sizes):5d} | {np.mean(sizes):9.2f} | {np.std(sizes):9.2f} | {np.mean(passed):10.2f} | {np.std(passed):10.2f}")


def main():
    # Create analyzer with max_split_ratio=1.2
    analyzer = ForestAnalyzer(n_vectors=100, cluster_centers=5, dim=128, max_depth=20, max_split_ratio=2,
                               n_trees=1, cluster_std=1, max_hash_attempts=1000)
    
    # Print summary
    # analyzer.print_summary()
    
    # # Plot tree
    # analyzer.plot_tree()
    
    # Plot statistics
    analyzer.plot_statistics()


if __name__ == "__main__":
    main() 