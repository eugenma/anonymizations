from functools import partial
from typing import Sequence, Optional

from joblib import Memory

import numpy as np
import pandas as pd
import pandas_datareader as pdr


mem = Memory(location='.tmp-joblib')


def generate_uniform(size: Sequence[int]) -> pd.DataFrame:
    """Generate sample data from uniform integers."""
    return pd.DataFrame(np.random.randint(0, 100, size=size))


def generate_tickers(tickers: Optional[Sequence[str]]=None) -> pd.DataFrame:
    """Create a sample data from asset tickers.

    :param tickers: The NASDAQ Tickers one wants to use. Has a default of 12
                    tickers.
    """
    def fit(asset: pd.DataFrame, nrows) -> pd.DataFrame:
        """Return first `nrows` 'Open' ticks."""
        column = asset.columns.get_loc('Open')
        asset.reset_index(drop=True, inplace=True)
        return asset.iloc[:nrows, column]

    if not tickers:
        # Just some default tickers. There is no meaning behind the list and
        # are chosen without any order or relation.
        tickers = ['AAPL', 'ABC', 'IBM',
                   'MSFT', 'FB', 'AMZN',
                   'SAP', 'VLKAY', 'BASFY',
                   'PG', 'NSRGF', 'DB']

    assets = list(pdr.get_data_morningstar(t) for t in tickers)
    min_shape = min([a.shape[0] for a in assets])
    raw_data = map(partial(fit, nrows=min_shape), assets)

    combine = pd.concat(list(raw_data), axis=1)
    combine.columns = list(range(combine.shape[1]))

    get_combine = mem.cache(lambda: combine)

    df = get_combine()

    return df


def impute_missings(src: pd.DataFrame, p_missings: float) -> pd.DataFrame:
    missings = np.random.binomial(1, p_missings, size=src.shape)
    missings = pd.DataFrame(missings.astype(bool))

    return src.where(~missings, np.nan)


def anonymize(orig: pd.DataFrame, sd: float=1.0) -> pd.DataFrame:
    errors = np.random.normal(0.0, scale=sd, size=orig.shape)
    result = orig + errors  # type: pd.DataFrame
    return result
