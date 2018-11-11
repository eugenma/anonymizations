from pprint import pformat
from typing import Callable, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from pyrobdean.base import MatchingResult


def series_to_df(src: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(src).T


def concat_any(**kwargs: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """Concat DataFrames with Series into a single DataFrame.

    The index is creates as multilevel index.
    """
    def as_df(src):
        if isinstance(src, pd.Series):
            return series_to_df(src)
        return src

    exclude_none = {k: v for k, v in kwargs.items() if v is not None}
    data_frames = map(as_df, exclude_none.values())
    return pd.concat(data_frames, keys=exclude_none.keys())


def pdisplay(src: pd.DataFrame, caption: str="", head: bool=True) -> None:
    df = src.fillna("")
    if head:
        df = df.head()

    if caption:
        display(df.style.set_caption(caption))
    else:
        display(df)


def show_matching(
        data: pd.DataFrame, aux_info: pd.Series,
        match_algorithm: Callable[[pd.DataFrame, pd.Series], MatchingResult],
        plot: bool=True) -> MatchingResult:
    matched = match_algorithm(data, aux_info)

    if not plot:
        return matched

    if matched.info:
        display("Info from the algorithms")
        print(pformat(matched.info, indent=4))

    if not matched.has_match:
        display("No match found")
        return matched

    if matched.pr.count() > 1:
        pdisplay(
            series_to_df(matched.pr.sort_values(ascending=False).head()),
            f"Probability distribution (head of {matched.pr.count()} total)")
        matched.pr.plot(title="Probability distribution", )
        plt.xlabel("Record index")
        plt.ylabel("Probability")

    aux_in_data = aux_info.name in data.index
    anonymized_true = data.loc[aux_info.name] if aux_in_data else None

    if not matched.has_match:
        pdisplay(concat_any(auxiliary=aux_info,
                            anonymized_true=anonymized_true),
                 "No match found", head=False)
    else:
        pdisplay(concat_any(auxiliary=aux_info,
                            anonymized_true=anonymized_true,
                            match=matched.match),
                 "With matches", head=False)

    return matched


def show_scatter(orig: pd.DataFrame, anon: pd.DataFrame, npics: int=4):
    _, num_data_cols = anon.shape
    num_rows = int(np.ceil(npics/2))
    plt.subplots(num_rows, 2, figsize=(10, 10))

    # Since there are too many combinations for each column,
    # we just print scatter of neighbor columns
    for i in range(1, num_rows * 2 + 1):
        plt.subplot(num_rows, 2, i)
        plt.scatter(orig[i - 1], orig[i], marker='.', color='b', alpha=0.3)
        plt.scatter(anon[i - 1], anon[i], marker='.', color='r', alpha=0.3)
        plt.legend(labels=["Original", "Anonymized", ])


def extract_anon(anon, aux):
    return anon.loc[aux.name, :]
