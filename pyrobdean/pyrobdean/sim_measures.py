def sim_zero_one(x, y, eps: float=1.0) -> bool:
    """Zero One similarity measure.

    :return: `1` if `x` and `y` are within `eps`, otherwise `0`.
    """
    return (x-y).abs() < eps
