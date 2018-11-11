import numpy as np
import pandas as pd

from ..base import sim_aux_rdash, MatchingResult, supp
from ..sim_measures import sim_zero_one


def _score_weighted(rdash: pd.DataFrame, aux: pd.Series, sim_scalar):
    aux_supp = supp(aux)

    # in the line .count(axis=0) we count the size of the support of the column
    weights = rdash.iloc[:, aux_supp.index] \
        .count(axis=0) \
        .apply(lambda a_supp_i: 1.0 / np.log(a_supp_i))

    sim_res = sim_aux_rdash(rdash, aux, sim_scalar)

    result = (weights * sim_res).sum(axis=1)
    return result


def _select_different(scores):
    unique_scores = scores.sort_values(ascending=False).unique()

    # we want to ignore numerical errors, hence unique() is not enough.
    # we need to consider approximately close values as same.
    unique_mask = ~np.isclose(np.diff(unique_scores), 0.0)
    # the first one is always included
    unique_mask = np.insert(unique_mask, 0, True)
    unique_scores = unique_scores[unique_mask]

    two_highest = unique_scores[:2]
    return two_highest


def _select_two_highest(scores):
    sorted = scores.sort_values(ascending=False)
    return sorted[:2]


def algorithm1b(
        rdash, aux,
        sim_scalar=sim_zero_one,
        eccentricity=0.1) -> MatchingResult:

    def score(rdash, aux):
        return _score_weighted(rdash, aux, sim_scalar)

    s = score(rdash, aux)
    # two_highest = _select_different(s)
    two_highest = _select_two_highest(s)

    max_s = two_highest.iloc[0]
    max_s_2 = two_highest.iloc[1]
    s_var = s.var()

    std = (max_s - max_s_2) / s_var

    info = {'max_s': max_s, 'max_s_2': max_s_2, 's_var': s.var(),
            'std': std, 'eccentricity': eccentricity}

    pr_distribution_nn = s.apply(lambda v: np.exp(v / s_var))
    c = pr_distribution_nn.sum()
    pr_distribution = (1.0 / c) * pr_distribution_nn

    has_match = std > eccentricity
    if has_match:
        match = rdash[s == two_highest[0]]
        return MatchingResult(match=match, pr=pr_distribution, scores=s,
                              info=info)

    return MatchingResult(match=None, pr=pr_distribution, scores=s,
                          info=info)
