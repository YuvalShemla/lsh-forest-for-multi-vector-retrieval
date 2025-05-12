import matplotlib.pyplot as plt
import numpy as np


def plot(x, y, xlabel, ylabel, title):
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_with_identity_line(x, y, xlabel, ylabel, title):
    x = np.asarray(x)
    y = np.asarray(y)

    plt.figure()
    plt.scatter(x, y, marker='o')

    # ─── y = x identity line in red ───
    x_line = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), 200)
    plt.plot(x_line, x_line, color='red', linewidth=2, linestyle='--', label='y = x')

    # ─── cosmetics ───
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_with_trendline(x, y, xlabel, ylabel, title, deg: int = 1):
    x = np.asarray(x)
    y = np.asarray(y)

    plt.figure()
    plt.scatter(x, y, marker='o')

    coeffs   = np.polyfit(x, y, deg=deg)       # least‑squares fit
    poly     = np.poly1d(coeffs)
    x_line   = np.linspace(x.min(), x.max(), 200)
    plt.plot(x_line, poly(x_line), linewidth=2, linestyle='--')  # trend only

    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_similarity_heatmap(query_vecs, doc_vecs, title="Similarity Heatmap"):
    query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
    doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
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