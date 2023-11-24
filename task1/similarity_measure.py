import numpy as np
import random


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b)


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.sum(np.abs(a - b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)


def random_similarity(_a: np.ndarray, _b: np.ndarray) -> float:
    return random.uniform(0, 1)
