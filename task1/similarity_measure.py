import numpy as np


def cosine_similarity(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a) + 1e-10
    B_norm = np.linalg.norm(B, axis=1) + 1e-10
    return np.dot(B, a.T) / (B_norm * a_norm)


def dot_product(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.dot(B, a.T)


def manhattan_distance(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(B - a), axis=1)


def euclidean_distance(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.linalg.norm(B - a, axis=1)


def random_similarity(_a: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.random.uniform(0, 1, size=B.shape[0])
