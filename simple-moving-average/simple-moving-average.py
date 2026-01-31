def simple_moving_average(values: list[int | float], window_size: int) -> list[float]:
    """Calculates the Simple Moving Average of the given values.

    Parameters
    ----------
    values: list[int | float]
        A list of numerical values.
    window_size: int
        The size of the moving window.

    Returns
    -------
    list[float]
        A list containing the SMA values.
    """
    smas = []
    # Calc initial window sum:
    running_sum = sum(values[:window_size])
    smas.append(running_sum / window_size)
    # Slide the window:
    for i in range(window_size, len(values)):
        running_sum += values[i] - values[i - window_size]
        smas.append(running_sum / window_size)

    return smas