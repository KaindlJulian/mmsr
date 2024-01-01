import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from typing import Any, Callable, Dict, Tuple
from io import StringIO

from task2.dcg import calculate_dcg_2, calculate_idcg_2

DEFAULT_K = 10


class Pipeline:
    def __init__(self, config: pd.DataFrame, genres: pd.DataFrame, k=100):
        """
        Initialises the pipeline by precomputing all results for all systems. The results are then saved to disk.
        :param config: Dataframe with retrieval systems ["rs_object", "metric", "feature"]
        :param genres: Genres for every track
        :param k: Number of results to precompute for every query
        """
        self.eval = config
        self.results = {}
        self.max_k = k
        self.genres = genres.copy()
        self.genre_overlap_matrix = self.__create_genre_overlap_matrix(genres)
        self.id_2_genres = genres.set_index("id")["genre"].to_dict()

        for system in tqdm(
            config.itertuples(),
            total=len(config),
            desc=f"Creating result lists for every rs (max_k={self.max_k})",
        ):
            rs = system.rs_object
            file_path = (
                f"results/{system.metric}_{system.feature}_results_{self.max_k}.npy"
            )

            r = {}
            try:
                r = np.load(file_path, allow_pickle=True).item()
                print(
                    f'loaded results for {system.metric=}, {system.feature=} from "{file_path}"'
                )
            except OSError:
                for query in tqdm(
                    rs.df.itertuples(),
                    total=len(rs.df),
                    desc=f"calculating results for {system.metric=} {system.feature=}",
                ):
                    r[query.id] = rs.retrieve(query.id, self.max_k)[
                        ["id", "similarity", "genre"]
                    ]
                np.save(file_path, r)
            self.results[rs] = r

    def run(self, steps: Tuple[Callable, Dict[str, Any]]) -> pd.DataFrame:
        for func, kwargs in tqdm(steps, desc="running pipeline"):
            col = []
            for system in tqdm(
                self.eval.itertuples(),
                total=len(self.eval),
                desc=f"Calculating '{func.__name__}' with {kwargs}",
            ):
                col.append(func(self, system, **kwargs))
            self.eval[func.__name__] = col
        return self.eval

    def load_results_csv(self, file_name):
        loaded = pd.read_csv(file_name)
        # parse nested dataframe
        if "precision_and_recall_interval" in loaded.columns:
            loaded["precision_and_recall_interval"] = loaded[
                "precision_and_recall_interval"
            ].apply(
                lambda x: pd.read_csv(
                    StringIO(x),
                    delim_whitespace=True,
                    header=None,
                    names=["k", "recall", "precision"],
                    skiprows=2,
                ).set_index("k")
            )
        if (
            loaded["metric"].eq(self.eval["metric"]).all()
            and loaded["feature"].eq(self.eval["feature"]).all()
        ):
            loaded["rs_object"] = self.eval["rs_object"]
            self.eval = loaded

    @staticmethod
    def __create_genre_overlap_matrix(genres_df):
        genre_dict = genres_df.set_index("id")["genre"].to_dict()
        overlap_matrix = pd.DataFrame(
            index=genres_df["id"], columns=genres_df["id"], dtype=bool
        )
        for song_id, genres in tqdm(
            genre_dict.items(),
            total=len(genres_df),
            desc="Creating genre overlap matrix",
        ):
            overlap_matrix.loc[song_id] = [
                bool(genres & genre_dict[other_id]) for other_id in genres_df["id"]
            ]
        return overlap_matrix

    def __get_full_results(self, rs) -> dict:
        return self.results[rs]

    def mean_precision_at_k(self, system, **kwargs):
        k = kwargs.get("k", DEFAULT_K)
        rs = system.rs_object
        results = self.__get_full_results(rs)
        precision = 0
        for query in rs.df.itertuples():
            retrieved = results[query.id][:k]
            relevant_items_retrieved = self.genre_overlap_matrix.loc[
                query.id, retrieved["id"]
            ].sum()
            precision += relevant_items_retrieved / k
        return precision / len(rs.df)

    def mean_recall_at_k(self, system, **kwargs):
        k = kwargs.get("k", DEFAULT_K)
        rs = system.rs_object
        results = self.__get_full_results(rs)
        recall = 0
        for query in rs.df.itertuples():
            retrieved = results[query.id][:k]
            relevant_items_retrieved = self.genre_overlap_matrix.loc[
                query.id, retrieved["id"]
            ].sum()
            relevant_items_total = self.genre_overlap_matrix.loc[query.id].sum()
            recall += (
                relevant_items_retrieved / relevant_items_total
                if relevant_items_total > 0
                else 0.0
            )
        return recall / len(rs.df)

    def precision_and_recall_interval(self, system, **kwargs):
        k1 = kwargs.get("k_min", 0)
        k2 = kwargs.get("k_max", DEFAULT_K)
        assert k1 < k2
        step_size = kwargs.get("step_size", 1)
        k_values = list(range(k2, k1 - 1, -step_size))
        if k1 not in k_values:
            k_values.append(k1)

        rs = system.rs_object
        metric_name = system.metric
        feature_name = system.feature
        results = self.__get_full_results(rs)
        num_queries = len(rs.df)

        recall_array = np.zeros(len(k_values))
        precision_array = np.zeros(len(k_values))
        for query in tqdm(
            rs.df.itertuples(),
            total=num_queries,
            desc=f"... for {metric_name=}, {feature_name=}",
        ):
            for i, k in enumerate(k_values):
                retrieved = results[query.id][:k]

                relevant_items_retrieved = self.genre_overlap_matrix.loc[
                    query.id, retrieved["id"]
                ].sum()
                relevant_items_total = self.genre_overlap_matrix.loc[query.id].sum()
                precision = relevant_items_retrieved / k
                recall = (
                    relevant_items_retrieved / relevant_items_total
                    if relevant_items_total > 0
                    else 0.0
                )

                precision_array[i] += precision
                recall_array[i] += recall

        recall_array /= num_queries
        precision_array /= num_queries
        return (
            pd.DataFrame(
                {"k": k_values, "recall": recall_array, "precision": precision_array}
            )
            .set_index("k")
            # .to_numpy()
        )

    def mean_ndcg_at_k(self, system, **kwargs):
        k = kwargs.get("k", DEFAULT_K)
        rs = system.rs_object
        metric_name = system.metric
        feature_name = system.feature
        results = self.__get_full_results(rs)

        ndcg = 0
        for query in tqdm(
            rs.df.itertuples(),
            total=len(rs.df),
            desc=f"... for {metric_name=}, {feature_name=}",
        ):
            retrieved = results[query.id]
            idcg = calculate_idcg_2(query.id, k, self.id_2_genres)
            ndcg += calculate_dcg_2(query.id, retrieved, k, self.id_2_genres) / idcg
        return ndcg / len(rs.df)

    def genre_coverage_at_k(self, system, **kwargs):
        k = kwargs.get("k", DEFAULT_K)
        all_genres = set()
        for g in self.id_2_genres.values():
            all_genres.update(g)

        rs = system.rs_object
        results = self.__get_full_results(rs)
        result_genres = set()
        for query in rs.df.itertuples():
            retrieved = results[query.id][:k]
            for r in retrieved.itertuples():
                result_genres.update(self.id_2_genres[r.id])
        return len(result_genres) / len(all_genres)

    def mean_genre_diversity_at_k(self, system, **kwargs):
        k = kwargs.get("k", DEFAULT_K)
        all_genres = set()
        for g in self.id_2_genres.values():
            all_genres.update(g)
        genre_2_index = {g: i for i, g in enumerate(all_genres)}

        rs = system.rs_object
        results = self.__get_full_results(rs)

        entropy_sum = 0.0
        for query in rs.df.itertuples():
            retrieved = results[query.id][:k]
            distribution = np.zeros(len(all_genres), dtype=float)
            for r in retrieved.itertuples():
                genres = self.id_2_genres[r.id]
                for g in genres:
                    distribution[genre_2_index[g]] += 1 / len(genres)
            distribution /= k
            # https://en.wikipedia.org/wiki/Diversity_index#Shannon_index
            entropy_per_query = -np.sum(
                distribution * np.log2(distribution, where=(distribution != 0))
            )
            entropy_sum += entropy_per_query

        return entropy_sum / len(rs.df)

    def save_to_csv(self, _, **kwargs):
        file_name = kwargs.get("file_name", "task3_pipeline.csv")
        self.eval.to_csv(file_name, index=False)
