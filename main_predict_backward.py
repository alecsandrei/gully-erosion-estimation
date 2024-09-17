from __future__ import annotations

from pathlib import Path
import typing as t
import concurrent.futures

import shapely
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from gully_automation import DEBUG, CACHE, MODEL
from gully_automation.geometry import (
    get_centerline,
    merge_linestrings,
    CenterlineTypes,
    get_pour_points,
    merge_downstream,
    is_orphaned,
    dangles_gen,
    snap_to_nearest,
    aggregate_overlapping_points,
    estimate_gully_beds,
    map_centerlines_and_profiles,
    extend_line_to_geom,
    Endpoints
)
from gully_automation.raster import DEM, multilevel_b_spline, Evaluator, inverse_distance_weighted, align_rasters
from gully_automation.changepoint import find_changepoints, plot_changepoints


def run(gpkg: Path, dem: Path, out_folder: Path):
    model = gpkg.stem
    print('Estimating for model', model)
    out_folder = out_folder / f'{model}_2003'
    if not out_folder.exists():
        out_folder.mkdir()
    _2003_gdf = gpd.read_file(gpkg, layer='2003')
    _2012_gdf = gpd.read_file(gpkg, layer='2012', engine='fiona')
    epsg = _2012_gdf.crs.to_epsg()
    _2003_polygon = _2003_gdf.geometry[0]
    _2012_polygon = _2012_gdf.geometry[0]
    _2012_2003_diff = _2012_polygon.difference(_2003_polygon)

    dem_2012 = DEM(dem, epsg=epsg)
    _2003_centerline = CenterlineTypes.from_linestrings(
        get_centerline(_2003_polygon, crs=epsg)
    )
    _2003_centerline.clean_orphaned()
    _2003_merged = merge_linestrings(
        _2003_centerline.contiguous,
        _2003_centerline.discontiguous
    )

    _2012_centerline = CenterlineTypes.from_linestrings(
        get_centerline(_2012_polygon, crs=epsg)
    )
    _2012_centerline.clean_orphaned()
    _2012_merged = merge_linestrings(
        _2012_centerline.contiguous,
        _2012_centerline.discontiguous
    )

    _2003_merged.to_file(out_folder / f'{model}_2003_centerline.shp')
    _2012_merged.to_file(out_folder / f'{model}_2012_centerline.shp')
    _2012_dangles = list(dangles_gen(_2012_merged))

    gpd.GeoSeries(_2012_dangles, crs=epsg).to_file(
        out_folder / f'{model}_2012_centerline_dangles.shp'
    )
    _2012_profiles = dem_2012.line_profiles(
        _2012_dangles,
        epsg,
        out_file=None
    ).clip(_2012_polygon)
    _2012_profiles = _2012_profiles[
        ~_2012_profiles.apply(lambda line: is_orphaned(line, _2012_profiles))
    ]

    _2012_profiles.to_file(out_folder / f'{model}_2012_profiles_from_dangles.shp')
    _2003_dangles = gpd.GeoSeries(list(dangles_gen(_2003_merged)), crs=epsg)
    _2003_dangles.to_file(
        out_folder / f'{model}_2003_centerline_dangles.shp'
    )
    _2003_profiles = dem_2012.line_profiles(
        _2003_dangles,
        epsg,
        out_file=out_folder / f'{model}_2003_profiles_from_dangles.shp'
    )
    pour_points = (
        gpd.GeoSeries(
            [Endpoints.from_linestring(line).end for line in _2012_profiles], crs=epsg
        )
        .drop_duplicates()
    )
    assert pour_points.shape[0] == 1, 'Expected a single pour point'
    pour_points_snapped = gpd.GeoSeries([
        snap_to_nearest(pour_point, _2003_dangles)
        for pour_point in pour_points
    ], crs=epsg)
    pour_points_snapped.to_file(
        out_folder / 'pour_points_snapped.shp'
    )
    _2003_merged = gpd.GeoSeries(
        merge_downstream(
            _2003_centerline,
            pour_points_snapped,
            _2003_polygon
        ),
        crs=epsg
    )
    _2003_merged.to_file(out_folder / f'{model}_2003_merged_downstream.shp')


Models = t.Literal[
    'saveni_aval',
]

MODELS = ['saveni_aval']


def main(model: Models):
    dem = Path(f'./data/{model}_2012.tif')
    gpkg = Path(f'./data/{model}.gpkg')
    out_folder = Path('./data/derived')
    out_folder.mkdir(exist_ok=True)
    run(gpkg, dem, out_folder)


if __name__ == '__main__':
    if MODEL is not None:
        main(model=MODEL)
    else:
        for model in MODELS[::-1]:
            main(model=model)
