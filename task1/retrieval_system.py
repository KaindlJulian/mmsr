from typing import Union, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from task1.similarity_measure import cosine_similarity


@dataclass
class SongInfo:
    title: str
    artist: str


class RetrievalSystem:
    def __init__(
            self,
            df: pd.DataFrame,
            sim_metric: Callable = cosine_similarity,
            sim_feature: str = "bert_embedding",
            enable_cache: bool = True
    ):
        self.df = df
        self.sim_metric = sim_metric
        self.sim_feature = sim_feature

        if self.sim_feature not in self.df.columns:
            raise ValueError(
                f"'{self.sim_feature}' not found in the dataframe columns."
            )

        # Precompute the stacked version of the feature
        # array(list(array)) -> 2d array
        self.all_songs_stacked = np.vstack(self.df[self.sim_feature].values)

        self.cache_enabled = enable_cache
        self.cache = {}

    def _calc_similarity(self, song: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        Calculate the similarity of the given song with all songs in the dataset.

        Parameters:
        - song: DataFrame row representing the song.
        - n: Number of top similar songs to retrieve.

        Returns:
        - DataFrame of top n similar songs.
        """
        song_vector = song[self.sim_feature]

        # Compute similarity for each song in the dataset, ensuring each song_vec is reshaped to 2D
        similarity = np.array(
            [
                # TODO this sets similarity with self as -1, do we want this?
                self.sim_metric(song_vector, song_vec) if not np.array_equal(song_vector, song_vec) else -1
                for song_vec in self.all_songs_stacked
            ]
        )

        top_n_indices = np.argsort(similarity)[::-1][:n]
        top_n = self.df.iloc[top_n_indices]
        # make pandas happy: no in-place modification
        top_n = self.df.iloc[top_n_indices].copy()
        top_n["similarity"] = similarity[top_n_indices]
        return top_n

    def random_baseline(self, query: Union[int, str], n: int = 5) -> pd.DataFrame:
        """
        Retrieve random songs from the dataset.

        Parameters:
        - query: If int, row of df. If str, song_id. If SongInfo, title and artist.- song_id: ID of the song.
            Not used in this method.
        - n: Number of songs to retrieve.

        Returns:
        - DataFrame of n random songs.
        """
        rand_n = self.df.sample(n=n)
        return self._remove_embeddings(rand_n)

    @staticmethod
    def _remove_embeddings(df: pd.DataFrame) -> pd.DataFrame:
        """
        Do not return columns with "embedding" or "tf-idf" in the name


        Args:
            df (pd.DataFrame): Dataframe to remove columns from

        Returns:
            pd.DataFrame: Dataframe without embedding and tf-idf columns
        """
        return df.loc[:, ~df.columns.str.contains("embedding|tf-idf")].reset_index(
            drop=True
        )

    def retrieve(self, query: Union[int, str, SongInfo], n: int = 5) -> pd.DataFrame:
        """
        Retrieve the top n songs similar to the given song_id.

        Parameters:
        - query: If int, row of df. If str, song_id. If SongInfo, title and artist.
        - n: Number of songs to retrieve.

        Returns:
        - DataFrame of top n similar songs.
        """
        if isinstance(query, (int, str)):
            song_id = query
            if song_id not in self.df["id"].values and song_id not in self.df.index:
                raise ValueError(f"Song id {song_id} not in the dataset.")
            song = (
                self.df.loc[song_id]
                if isinstance(song_id, int)
                else self.df[self.df["id"] == song_id].iloc[0]
            )
        elif isinstance(query, SongInfo):
            title, artist = query.title, query.artist
            song = self.df[(self.df["song"] == title) & (self.df["artist"] == artist)]
            if song.empty:
                raise ValueError(
                    f"Song with title '{title}' and artist '{artist}' not found in the dataset."
                )
            song = song.iloc[0]
        else:
            raise ValueError(
                "Invalid query type. Provide either song_id (int/str) or an instance of SongInfo."
            )

        if self.cache_enabled and song['id'] in self.cache:
            cached_result = self.cache[song['id']]
            if len(cached_result) >= n:
                return cached_result.head(n)

        top_n = self._calc_similarity(song, n=n)
        # from the assignment, it is not 100% clear what to return --> return all except embeddings
        result = self._remove_embeddings(top_n)

        self.cache[song['id']] = result
        return result
