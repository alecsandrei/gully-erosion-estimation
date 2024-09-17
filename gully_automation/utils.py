from __future__ import annotations

from pathlib import Path
import typing as t

import geopandas as gpd
import pandas as pd

from gully_automation import DEBUG

if t.TYPE_CHECKING:
    from qgis.core import QgsVectorLayer


def vector_layers_to_geodataframe(
    vector_layers: list[Path]
) -> gpd.GeoDataFrame | None:
    if not vector_layers:
        return None
    df = pd.DataFrame()
    for i, layer in enumerate(vector_layers, start=1):
        if DEBUG >= 1:
            print(f'Reading layer {layer.name}')
        gdf = gpd.read_file(layer)
        gdf['_layer'] = layer.stem
        df = pd.concat([df, gdf], ignore_index=True)
    assert isinstance(df, gpd.GeoDataFrame)
    return df


def qgis_vector_to_geodataframe(vector_layer: QgsVectorLayer):
    return gpd.GeoDataFrame.from_features(vector_layer.getFeatures())
