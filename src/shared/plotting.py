import matplotlib.pyplot as plt
import numpy as np


def plot(x, y, xlabel):
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel('Similarity Score')
    plt.title(f'Convergence of LSH approximation as {xlabel} grows')
    plt.show()

def plot_similarity_heatmap(query_vecs, doc_vecs, title="Similarity Heatmap"):
    sim_matrix = np.dot(query_vecs, doc_vecs.T)  # shape (q, m)

    gamma = 2  # adjust for more/less suppression
    transformed = np.sign(sim_matrix) * (np.abs(sim_matrix) ** gamma)

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("bwr")
    plt.imshow(transformed, aspect='auto', cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(label=f'Power-Law Dot Product (γ={gamma})')
    plt.xlabel("Document Vector Index")
    plt.ylabel("Query Vector Index")
    plt.title(title)
    plt.tight_layout()
    plt.show()