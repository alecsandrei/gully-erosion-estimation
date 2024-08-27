from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import random
import itertools

import matplotlib.colors
import matplotlib.pyplot as plt
import shapely.geometry
import geopandas as gpd
import pandas as pd

from gully_automation import EPS

if TYPE_CHECKING:
    from qgis.core import QgsVectorLayer


def vector_layers_to_geodataframe(
    vector_layers: list[Path]
) -> gpd.GeoDataFrame | None:
    if not vector_layers:
        return None
    df = pd.DataFrame()
    for layer in vector_layers:
        gdf = gpd.read_file(layer)
        gdf['_layer'] = layer.stem
        df = pd.concat([df, gdf], ignore_index=True)
    assert isinstance(df, gpd.GeoDataFrame)
    return df


def qgis_vector_to_geodataframe(
    vector_layer: QgsVectorLayer
):
    return gpd.GeoDataFrame.from_features(vector_layer.getFeatures())


def plot(**series: gpd.GeoSeries):
    fig = plt.figure()
    ax = fig.add_subplot()
    cnames = list(matplotlib.colors.cnames)
    random.shuffle(cnames)
    for (name, serie), color in zip(
        series.items(), itertools.cycle(cnames)
    ):
        if (serie.geom_type == 'Polygon').all():
            serie.boundary.plot(
                ax=ax, color=color, label=name, alpha=1, linewidth=3
            )
        else:
            serie.plot(ax=ax, color=color, label=name, alpha=1)
    plt.legend()
    plt.show()
