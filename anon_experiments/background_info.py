"""For Details check the jupyter notebook."""
from typing import Union

import pandas as pd
import numpy as np

from pyrobdean.base import supp


def __disclose(orig: pd.DataFrame, num_attribs: int) -> pd.DataFrame:
    """Simulated auxiliary variables from the database.
    
    :remark: Cannot be applied on matrices. Use `auxiliary()` instead.
    """
    support = supp(orig)
    amount = min(num_attribs, support.size)
    indices = np.random.choice(support.index.size, amount, False)
    selected = support.iloc[indices]

    res = pd.Series(np.empty(orig.size))
    res.loc[:] = np.nan
    res.loc[selected.index] = selected

    return res


def disclose(orig: pd.DataFrame, rows: Union[float, int]=0.5, num_attribs: int=3) \
        -> pd.DataFrame:
    """"Generate auxiliary background knowledge out of source database.
    
    :param num_attribs: Amount of attributes one want to disclose. 
    :param rows: Either percentage (if `rows < 1.0`) or absolute number of 
            rows one wants to disclose.
    :return: The dataframe with `rows` amount of rows an same number of 
            attributes as `orig`. In each record there are `num_attribs` 
            disclosed attributes. Other remain as `NaN`.
    """
    if rows >= 1.0:
        num_rows = rows
    else:
        num_rows = int(orig.index.size*rows)

    irows = np.random.randint(0, orig.index.size, num_rows)
    sub = orig.iloc[irows]
    known = sub.apply(lambda r: __disclose(r, num_attribs), axis=1)
    return known
