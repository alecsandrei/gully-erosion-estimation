from __future__ import annotations

import shapely.geometry
from pathlib import Path
import geopandas as gpd
import pandas as pd

from gully_automation import EPS


def vector_layers_to_geodataframe(
    vector_layers: list[Path]
) -> gpd.GeoDataFrame | None:
    if not vector_layers:
        return None
    df = pd.DataFrame()
    for layer in vector_layers:
        gdf = gpd.read_file(layer)
        gdf['_layer'] = layer.stem
        df = pd.concat([df, gdf])
    assert isinstance(df, gpd.GeoDataFrame)
    return df


def intersects(g1, g2) -> bool:
    """A wrapper intersects which instead uses the distance."""
    return g1.distance(g2) < EPS
