import numpy as np
from typing import Callable

class LSHFamily:
    """
    A family of LSH functions using random projections.
    Each hash function is a random hyperplane that splits the space into two halves.
    """
    
    def __init__(self, vector_dim: int):
        """
        Initialize the LSH family.
        
        Args:
            vector_dim: Dimension of the vectors to be hashed
        """
        self.vector_dim = vector_dim
        
    def sample(self) -> Callable[[np.ndarray], int]:
        """
        Sample a random hash function from the family.
        
        Returns:
            A function that takes a vector and returns 0 or 1
        """
        # Generate random projection vector
        projection = np.random.randn(self.vector_dim)
        projection = projection / np.linalg.norm(projection)  # Normalize
        
        def hash_func(vector: np.ndarray) -> int:
            """
            Hash a vector using the random projection.
            
            Args:
                vector: Input vector to hash
                
            Returns:
                0 if vector is on one side of the hyperplane, 1 if on the other
            """
            return 1 if np.dot(vector, projection) >= 0 else 0
            
        return hash_func 