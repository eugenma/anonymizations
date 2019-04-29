from functools import partial
import itertools
from typing import Sequence, Optional, Callable

from joblib import Memory

import numpy as np
import pandas as pd
import pandas_datareader as pdr


mem = Memory(location='.tmp-joblib')


def generate_uniform(size: Sequence[int]) -> pd.DataFrame:
    """Generate sample data from uniform integers."""
    return pd.DataFrame(np.random.randint(0, 100, size=size))


def build_accessor_alphavantage(key):
    from alpha_vantage.timeseries import TimeSeries
    
    @mem.cache(ignore=['ts', ])
    def single_accessor(ts: TimeSeries, ticker: str) -> pd.Series:
        data, meta = ts.get_daily(ticker)
        return data['1. open'].reset_index(drop=True)
        
    def accessor(tickers: Sequence[str]) -> pd.DataFrame:
        ts = TimeSeries(key, output_format='pandas')

        data, rows_calc_data = itertools.tee(map(partial(single_accessor, ts), tickers))
        min_rows = min(map(lambda d: d.shape[0], rows_calc_data))
        same_size_data = map(lambda d: d.iloc[:min_rows,], data) 
        combine = pd.concat(list(same_size_data), axis=1)
        combine.columns = list(range(combine.shape[1]))
        return combine

    return accessor


def generate_tickers(tickers: Optional[Sequence[str]], accessor: Callable[[Sequence[str],], pd.DataFrame]) -> pd.DataFrame:
    """Create a sample data from asset tickers.

    :param tickers: The NASDAQ Tickers one wants to use. Has a default of 12
                    tickers.
    """
    if not tickers:
        # Just some default tickers. There is no meaning behind the list and
        # are chosen without any order or relation.
        tickers = ['AAPL', 'ABC', 'IBM',
                   'MSFT', 'FB', 'AMZN',
                   'SAP', 'VLKAY', 'BASFY',
                   'PG', 
                   # 'NSRGF', 
                   'DB']

    df = accessor(tickers)
    return df


def impute_missings(src: pd.DataFrame, p_missings: float) -> pd.DataFrame:
    missings = np.random.binomial(1, p_missings, size=src.shape)
    missings = pd.DataFrame(missings.astype(bool))

    return src.where(~missings, np.nan)


def anonymize(orig: pd.DataFrame, sd: float=1.0) -> pd.DataFrame:
    errors = np.random.normal(0.0, scale=sd, size=orig.shape)
    result = orig + errors  # type: pd.DataFrame
    return result
