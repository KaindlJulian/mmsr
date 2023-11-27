import pandas as pd
import numpy as np


def sorensen_dice_genres(genre1: set, genre2: set) -> float:
    """Calculate the Sorensen-Dice coefficient for two sets of genres.
    Args:
        genre1 (set): First set of genres.
        genre2 (set): Second set of genres.
    Returns:
        float: Sorensen-Dice coefficient.
    """
    return 2 * len(genre1 & genre2) / (len(genre1) + len(genre2))


def calculate_dcg(
    sample_query: pd.Series, retrieved_tracks: pd.DataFrame, k: int
) -> float:
    """Calculate DCG for a single query song.
    Args:
        sample_query (pd.Series): Query song.
        retrieved_tracks (pd.DataFrame): Retrieved tracks.
        k (int): Number of retrieved tracks to consider.
    Returns:
        float: DCG@k
    """
    # Vectorized calculation of relevance scores
    relevance_scores = retrieved_tracks["genre"].apply(
        lambda x: sorensen_dice_genres(sample_query["genre"], x)
    )
    top_k_scores = relevance_scores.head(k).values

    denominators = np.log2(
        np.arange(2, k + 1)
    )  # Starts from 2 since log2(1) is 0, and it's used for the first item
    return top_k_scores[0] + sum(top_k_scores[1:] / denominators)


def calculate_idcg(sample_query: pd.Series, k: int, genres: pd.DataFrame) -> float:
    """Calculate IDCG for a single query song.
    Args:
        sample_query (pd.Series): Query song.
        k (int): Number of retrieved tracks to consider.
        genres (pd.DataFrame): Genres for every track.

    Returns:
        float: IDCG@k
    """
    # Vectorized calculation of relevance scores
    relevance_scores = genres["genre"].apply(
        lambda x: sorensen_dice_genres(sample_query["genre"], x)
    )
    # sort and take top k
    sorted_scores = relevance_scores.sort_values(ascending=False).head(k).values

    # normalize by log2 of rank
    denominators = np.log2(np.arange(2, k + 1))
    return sorted_scores[0] + sum(sorted_scores[1:] / denominators)


# different versions for usage in pipeline


def calculate_idcg_2(sample_query: str, k: int, genres: dict) -> float:
    relevance_scores = [
        sorensen_dice_genres(genres[sample_query], genres[x])
        for x in genres.keys()
        if x != sample_query
    ]
    relevance_scores.sort(reverse=True)
    sorted_scores = relevance_scores[:k]
    denominators = np.log2(np.arange(2, k + 1))
    return sorted_scores[0] + sum(sorted_scores[1:] / denominators)


def calculate_dcg_2(
    sample_query: str, retrieved_k_tracks: pd.DataFrame, k: int, genres: dict
) -> float:
    relevance_scores = [
        sorensen_dice_genres(genres[sample_query], genres[x])
        for x in retrieved_k_tracks["id"]
        if x != sample_query
    ]
    top_k_scores = relevance_scores[:k]
    denominators = np.log2(
        np.arange(2, k + 1)
    )  # Starts from 2 since log2(1) is 0, and it's used for the first item
    return top_k_scores[0] + sum(top_k_scores[1:] / denominators)
