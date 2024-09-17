from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import ruptures as rpt
import geopandas as gpd

from gully_automation import DEBUG


def find_changepoints(values: np.ndarray):
    algorithm = rpt.Pelt(model='rbf').fit(values)
    return algorithm.predict(pen=25)


def plot_changepoints(
    values: np.ndarray,
    changepoints: Sequence[int],
    out_file: Path
):
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(12, 5))

    ax.plot(values)
    plt.xticks(np.arange(0, values.shape[0], 5))
    for change_point in changepoints:
        ax.axvline(change_point, color='b', ls='--', linewidth=0.4)
    plt.xticks(rotation=90)
    plt.savefig(out_file)
    plt.close()


def estimate_gully(
    values_before: np.ndarray,
    values_after: np.ndarray,
    changepoint: int,
):

    def poly_fit(x, y, d):
        if x.shape[0] <= d:
            return y
        return np.polynomial.polynomial.Polynomial.fit(x, y, d)(x)

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
        plt.axvline(changepoint, c='red')
        plt.ylabel('Altitudine')
        # plt.show()

    y1 = values_before.copy()
    y1_head = y1[:changepoint]
    y1_poly = poly_fit(np.arange(y1_head.shape[0]), y1_head, 5)
    y2 = values_after.copy()
    no_head = fill_head_with_nan(y1, changepoint)
    padded = pad(no_head, y2)
    with_head = fill_polyfit(padded, y1_poly, y2)
    estimation = estimate_nan(with_head)
    if DEBUG >= 1:
        debug_estimation(y1, estimation)
    return estimation
