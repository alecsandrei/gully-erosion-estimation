from __future__ import annotations

from pathlib import Path
import collections.abc as c

import numpy as np
import ruptures as rpt
import geopandas as gpd

from gully_erosion_estimation import DEBUG


def find_changepoints(values: np.ndarray, penalty: int):
    algorithm = rpt.Pelt(model='rbf').fit(values)
    # the last value should not be returned.
    return algorithm.predict(pen=penalty)[:-1]


def estimate_gully(
    values_before: np.ndarray,
    values_after: np.ndarray,
    changepoints: c.Sequence[int],
    debug_out_file: Path | None = None
):
    # Only the first changepoint is considered.
    # The others are used, for now, for plotting purposes (debug)

    def poly_fit(x, y, d=5):
        if x.shape[0] <= 5:
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
        # estimated = normalize(exponential(np.linspace(max_, min_, nan_len)), min_, max_)
        # dont include max min
        linear_sequence = np.linspace(max_, min_, nan_len + 2)[1:-1]
        min_, max_ = linear_sequence[0], linear_sequence[-1]
        estimated = normalize(exponential(linear_sequence), max_, min_)
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
        ax.plot(
            x,
            estimation,
            c='#00ff00',
            label='2019 estimated channel',
            linewidth=2,
        )
        ax.plot(
            before,
            c='#5795dc',
            label='2012 flow path profile',
            linewidth=2
        )
        for i, changepoint in enumerate(changepoints):
            ax.axvline(
                changepoint,
                color='r',
                ls='--',
                linewidth=2,
                label='changepoint' if i == 0 else None
            )
        ax.legend(prop={'size': 15})
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylabel('Elevation (m)', size=15)
        plt.tight_layout()
        plt.savefig(debug_out_file, dpi=300)
        plt.close()
        # plt.show()

    def debug_poly_fit(
        head: np.ndarray,
        head_splline_fitted: np.ndarray
    ):
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=(5, 5))

        x = range(head.shape[0])
        ax.plot(x, head, c='#5795dc')
        ax.plot(head_splline_fitted, c='#00ff00')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylabel('Elevation (m)', size=15)
        plt.tight_layout()
        plt.savefig(debug_out_file.with_name(
            f'{debug_out_file.stem}_poly_fit.png'
        ), dpi=300)
        plt.close()
        # plt.show()

    y1 = values_before.copy()
    y1_head = y1[:changepoints[0]]
    y1_poly = poly_fit(np.arange(y1_head.shape[0]), y1_head)
    if DEBUG >= 2 and debug_out_file is not None:
        debug_poly_fit(y1_head, y1_poly)
    y2 = values_after.copy()
    no_head = fill_head_with_nan(y1, changepoints[0])
    padded = pad(no_head, y2)
    with_head = fill_polyfit(padded, y1_poly, y2)
    estimation = estimate_nan(with_head)
    # estimation[:changepoints[0]] = poly_fit(np.arange(estimation[:changepoints[0]].shape[0]), estimation[:changepoints[0]])
    # estimation = poly_fit(np.arange(estimation.shape[0]), estimation)
    if DEBUG >= 2 and debug_out_file is not None:
        debug_estimation(y1, estimation)
    return estimation
