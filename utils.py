import numpy as np
import pandas as pd
import os


def read(feature, h=0):
    return pd.read_csv(f"{os.getcwd()}/../data/id_{feature}_mmsr.tsv", delimiter="\t", header=h)


def embed_and_merge(df1, df2, col_name):
    embedding = df2.columns.difference(["id"], sort=False)
    df2[col_name] = df2[embedding].apply(lambda x: np.array(x, dtype=float), axis=1)
    df2.drop(embedding, inplace=True, axis=1)
    return pd.merge(df1, df2, left_on="id", right_on="id", how="left")
