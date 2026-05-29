"""The axis-aligned slice extraction primitive (internal).

A single pure function that slices a 2-D array by half-open index bounds,
padding any pixel that has no counterpart in the source. Kept separate from
:mod:`noobfriend.extraction.cutout._core` so the slice and reproject paths
stay independent.
"""

import numpy as np


def slice_with_fill(
    data: np.ndarray,
    x_bounds: tuple[int, int],
    y_bounds: tuple[int, int],
    fill_value: float = np.nan,
) -> np.ndarray:
    """Slice ``data`` by half-open index bounds, padding out-of-range pixels.

    The requested window may extend past the edges of (or fall entirely
    outside) ``data``; any pixel with no counterpart in ``data`` takes
    ``fill_value``. The output dtype is promoted as needed so that, for example,
    a NaN fill on integer data yields a float result.

    Parameters
    ----------
    data : numpy.ndarray
        The 2-D source array, indexed ``[row, col]`` i.e. ``[y, x]``.
    x_bounds : tuple[int, int]
        ``(x_start, x_end)`` half-open column bounds.
    y_bounds : tuple[int, int]
        ``(y_start, y_end)`` half-open row bounds.
    fill_value : float, optional
        Value for pixels outside ``data``, by default :data:`numpy.nan`.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(y_end - y_start, x_end - x_start)``.
    """
    out = np.full(
        (y_bounds[1] - y_bounds[0], x_bounds[1] - x_bounds[0]),
        fill_value,
        dtype=np.result_type(data.dtype, fill_value),
    )
    data_y_size, data_x_size = data.shape

    x_start_in_data = max(x_bounds[0], 0)
    x_end_in_data = min(x_bounds[1], data_x_size)
    y_start_in_data = max(y_bounds[0], 0)
    y_end_in_data = min(y_bounds[1], data_y_size)

    x_out_start = x_start_in_data - x_bounds[0]
    x_out_end = x_end_in_data - x_bounds[0]
    y_out_start = y_start_in_data - y_bounds[0]
    y_out_end = y_end_in_data - y_bounds[0]

    if x_out_end > x_out_start and y_out_end > y_out_start:
        out[y_out_start:y_out_end, x_out_start:x_out_end] = data[
            y_start_in_data:y_end_in_data, x_start_in_data:x_end_in_data
        ]
    return out
