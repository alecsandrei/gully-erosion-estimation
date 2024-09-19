from __future__ import annotations

from pathlib import Path
import collections.abc as c

import numpy as np
import ruptures as rpt
import geopandas as gpd
from scipy.interpolate import UnivariateSpline

from gully_erosion_estimation import DEBUG


def find_changepoints(values: np.ndarray, penalty: int):
    algorithm = rpt.Pelt(model='rbf').fit(values)
    return algorithm.predict(pen=penalty)


def estimate_gully(
    values_before: np.ndarray,
    values_after: np.ndarray,
    changepoints: c.Sequence[int],
):
    # Only the first changepoint is considered.
    # The others are used, for now, for plotting purposes (debug)

    def spline_fit(x, y):
        return UnivariateSpline(x, y)(x)

    def fill_head_with_nan(y: np.ndarray, changepoint: int):
        y = y.copy()
        y[:changepoint] = np.nan
        return y

    def pad(y1: np.ndarray, y2: np.ndarray):
        y1 = y1.copy()
        nans = np.empty(y2.shape[0])
        nans[:] = np.nan
        return np.concatenate([nans, y1])

    def normalize(array, min_, max_):
        y_min = min(array)
        y_max = max(array)
        return min_ + (array - y_min) * (max_ - min_) / (y_max - y_min)

    def exponential(y):
        return 5**y

    def estimate_nan(y: np.ndarray):
        y = y.copy()
        nan_indices = np.argwhere(np.isnan(y))
        nan_len = nan_indices.shape[0]
        max_, min_ = y[nan_indices[0] - 1], y[nan_indices[-1] + 1]
        estimated = normalize(exponential(np.linspace(max_, min_, nan_len)), min_, max_)
        y[nan_indices] = estimated
        return y

    def fill_polyfit(padded: np.ndarray, y_poly: np.ndarray, y2: np.ndarray):
        padded = padded.copy()
        y_poly += np.abs(y2[0] - y_poly[0])
        padded[:y_poly.shape[0]] = y_poly
        return padded

    def debug_estimation(before: np.ndarray, estimation: np.ndarray):
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=(15, 5))

        x = range(before.shape[0] - estimation.shape[0], before.shape[0])
        ax.plot(x, estimation, c='orange')
        ax.plot(before)
        for changepoint in changepoints:
            ax.axvline(changepoint, color='r', ls='--', linewidth=0.4)
        plt.ylabel('Elevation')
        # plt.show()

    y1 = values_before.copy()
    y1_head = y1[:changepoints[0]]
    y1_poly = spline_fit(np.arange(y1_head.shape[0]), y1_head)
    y2 = values_after.copy()
    no_head = fill_head_with_nan(y1, changepoints[0])
    padded = pad(no_head, y2)
    with_head = fill_polyfit(padded, y1_poly, y2)
    estimation = estimate_nan(with_head)
    if DEBUG >= 2:
        debug_estimation(y1, estimation)
    return estimation
