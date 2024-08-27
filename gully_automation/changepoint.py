from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import ruptures as rpt
import geopandas as gpd


def find_changepoints(values: np.ndarray):
    algorithm = rpt.Pelt(model='rbf').fit(values)
    return algorithm.predict(pen=15)


def plot_changepoints(values: np.ndarray, changepoints: Sequence[float]):
    import matplotlib.pyplot as plt

    for changepoint in changepoints:
        _, ax = plt.subplots(figsize=(12, 5))

        plt.plot(values, ax=ax)
        plt.xticks(np.arange(0, values.shape[0], 5))
        for change_point in changepoints:
            ax.axvline(change_point, color='b', ls='--', linewidth=0.4)
        plt.xticks(rotation=90)
        plt.show()
