import numpy as np
import pandas as pd

from ..base import sim_aux_rdash, MatchingResult
from ..sim_measures import sim_zero_one


def _score_min(rdash, aux, sim_scalar):
    return sim_aux_rdash(rdash, aux, sim_scalar).min(axis=1)


def algorithm1a(rdash, aux,
                sim_scalar=sim_zero_one,
                alpha=0.01) -> MatchingResult:
    def score(r, a):
        return _score_min(r, a, sim_scalar)

    scores = score(rdash, aux)
    matches = rdash[scores > alpha]

    if matches.empty:
        return MatchingResult(match=matches, pr=None, scores=scores, info=None)

    num_matches = len(matches.index)
    pr_distribution = pd.Series(
        np.repeat(1.0/num_matches, num_matches),
        index=matches.index)

    return MatchingResult(match=matches, pr=pr_distribution, scores=scores,
                          info=None)
