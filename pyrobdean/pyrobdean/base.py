from typing import Callable, Any

import pandas as pd

SimScalar = Callable[[Any, Any], Any]
Support = Callable[[pd.Series, ], pd.Series]


class MatchingResult:
    def __init__(self, match, pr, scores, info):
        self.pr = pr
        self.scores = scores
        self.info = info
        self.match = match

    @property
    def has_match(self) -> bool:
        return self.match is not None and not self.match.empty


def supp(x: pd.Series) -> pd.Series:
    """return sub series of nonnull attributes of a series.

    Note:
        Cannot be applied to matrices. Since each axis may have different
        number of na's.
    """
    if isinstance(x, pd.DataFrame):
        msg = f"'x' should be of type '{pd.Series}'. Currently is '{type(x)}'."
        raise ValueError(msg)

    return x.dropna()


def sim(x: pd.Series, y: pd.Series, sim_scalar: SimScalar) -> float:
    """Calculate similarity measure of two vectors.

    The parameter order is interchangeable, i.e. the function is symmetric.

    Note:
        Cannot be applied to matrices, due to usage of `supp()`.

    :param x: The first vector.
    :param y: The second vector.
    :param sim_scalar: The scalar similarity function.
    """
    indices_x = supp(x).index
    indices_y = supp(y).index

    denom = indices_x.union(indices_y).size

    nom = sum(sim_scalar(x, y))

    return nom/denom


def sim_aux_rdash(rdash: pd.DataFrame, aux: pd.Series, sim_scalar: SimScalar) \
        -> pd.DataFrame:
    """Calculate similarity measure on support of aux.

    Since the support is taken from the first parameter, the order of `aux` and
    `rdash` should be preserved. i.e. this function is _not_ symmetric.

    :param aux: Is the auxiliary information.
    :param rdash: The anonymized database one wants to disclose.
    """
    if isinstance(aux, pd.DataFrame):
        msg = " ".join([
            f"'aux' should be of type '{pd.Series}'.",
            f"Currently is '{type(aux)}'."])
        raise ValueError(msg)

    aux_support = supp(aux)
    rdash_on_aux = rdash.iloc[:, aux_support.index]

    result = rdash_on_aux \
        .apply(lambda r: sim_scalar(r, aux_support), axis=1)
    return result
