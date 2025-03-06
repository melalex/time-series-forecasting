import numpy as np
import pandas as pd


def series_to_sliding_window(series: pd.Series, window_size: int) -> np.ndarray:
    """
    Convert a Pandas Series into a NumPy array with a sliding window transformation.

    Parameters:
        series (pd.Series): Input time series.
        window_size (int): The size of the sliding window.

    Returns:
        np.ndarray: A 2D NumPy array where each row is a window.
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a Pandas Series.")

    if window_size <= 0:
        raise ValueError("Window size must be greater than 0.")

    if len(series) < window_size:
        raise ValueError(
            "Window size must be smaller than or equal to the series length."
        )

    # Convert Series to NumPy array
    series_array = series.to_numpy()

    # Apply sliding window using NumPy's stride_tricks
    shape = (len(series) - window_size + 1, window_size)
    strides = (series_array.strides[0], series_array.strides[0])

    x = np.lib.stride_tricks.as_strided(series_array, shape=shape, strides=strides)
    y = np.array([it[0] for it in x[1:]])

    return x[:-1].astype(np.float32), y[..., None].astype(np.float32)
