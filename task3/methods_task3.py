import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
from sklearn.decomposition import PCA
from task1.retrieval_system import RetrievalSystem
from task1.similarity_measure import (
    cosine_similarity,
    dot_product,
    manhattan_distance,
    euclidean_distance,
    random_similarity,
)


def scale_feature(feature):
    arr = []
    for row in feature:
        arr.append(row)
    arr = np.array(arr)

    # fir scaler to whole feature
    scaler = StandardScaler()
    scaler.fit(arr)

    for idx, row in tqdm(enumerate(feature), total=len(feature), desc="Scaling rows"):
        # scale row
        transformed_row = scaler.transform(row.reshape(1, -1))
        # Update the copy of the column
        feature.iloc[idx] = transformed_row[0]
    return feature


def concat_scale_features(first_feature, second_feature, df):
    # Concat features to form aggregated feature
    first = df[first_feature]
    second = df[second_feature]

    # no scaling yields better results
    # first = scale_feature(first.copy())
    # second = scale_feature(second.copy())

    combined_features = pd.concat([first, second], axis=1)
    combined_features['aggr_feature'] = combined_features.apply(lambda row: np.concatenate(row), axis=1)

    print(f"Number of columns in the first feature: {len(combined_features.iloc[0, 0])}")
    print(f"Number of columns in the second feature: {len(combined_features.iloc[0, 1])}")
    print(f"Number of columns in the combined features: {len(combined_features.iloc[0, 2])}")

    # returns dataframe with first, second and combined feature
    return combined_features


def pca_feature(combined_features):
    # reduce columns to 20%
    pca_components = int(len(combined_features.iloc[0, 2]) * 0.2)
    print(f"Reducing aggregated feature to {pca_components} components")
    pca = PCA(n_components=pca_components)
    arr = []

    # convert to arr where feature values are columns and rows are samples
    for row in combined_features["aggr_feature"]:
        arr.append(row)
    arr = np.array(arr)

    # fit pca to whole whole arr
    pca.fit(arr)

    aggr_feature_copy = combined_features["aggr_feature"].copy()

    for idx, row in tqdm(enumerate(aggr_feature_copy), total=len(aggr_feature_copy), desc="Transforming rows"):
        # apply pca to rows
        transformed_row = pca.transform(row.reshape(1, -1))
        # Update the copy of the column
        aggr_feature_copy.iloc[idx] = transformed_row[0]

    # returns dataframe containing only the reduced feature
    return aggr_feature_copy


def early_fusion(first_feature, second_feature, df):
    # concat
    features = concat_scale_features(first_feature, second_feature, df)
    # scale
    aggr_feature = pca_feature(features)
    name = f"ef_{first_feature}_{second_feature}"
    # Add aggregated feature to the dataframe
    df[name] = aggr_feature

    # And define new retrieval system instance for aggregated feature
    new_rs = RetrievalSystem(
        df=df,
        sim_metric=cosine_similarity,
        sim_feature=name,
    )
    # returns retrieval system instance with early fusion
    return new_rs, name, df

# example use
# define new RS, feature name needed for evaluation
# rs_cos_early_fusion_1, feature_name1, df = early_fusion("bert", "musicnn", df)

# sample_song = SongInfo(title="Always", artist="Bon Jovi")
# rs_cos_early_fusion_1.retrieve(sample_song)
