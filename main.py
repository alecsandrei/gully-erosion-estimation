from __future__ import annotations

from pathlib import Path
import typing as t
import concurrent.futures

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from gully_erosion_estimation import DEBUG, CACHE, MODEL, EVAL, PENALTY, MULTIPROCESSING
from gully_erosion_estimation.geometry import (
    get_centerline,
    merge_linestrings,
    CenterlineTypes,
    get_pour_points,
    merge_downstream,
    aggregate_overlapping_points,
    estimate_gully_beds,
    map_centerlines_and_profiles,
    extend_line_to_geom
)
from gully_erosion_estimation.raster import DEM, multilevel_b_spline, Evaluator, inverse_distance_weighted, align_rasters
from gully_erosion_estimation.changepoint import find_changepoints, plot_changepoints


def debug_profiles(model: Models, profile_samples):
    profiles = Path('./data/derived/profiles')
    if not profiles.exists():
        profiles.mkdir()
    model_profiles = profiles / model
    if not model_profiles.exists():
        model_profiles.mkdir()
    for id_ in profile_samples['ID_LINE'].unique():
        out_file = model_profiles / f'{id_}.png'
        profile = profile_samples.loc[profile_samples['ID_LINE'] == id_, 'Z']
        plot_changepoints(profile.values, find_changepoints(profile.values), out_file=out_file)


def run(gpkg: Path, dem: Path, truth_dem: Path, out_folder: Path):
    model = gpkg.stem
    print('Estimating for model', model)
    out_folder = out_folder / model
    if not out_folder.exists():
        out_folder.mkdir()
    try:
        _2003_gdf = gpd.read_file(gpkg, layer='2003')
    except:
        ...
    _2012_gdf = gpd.read_file(gpkg, layer='2012', engine='pyogrio')
    EPSG = _2012_gdf.crs.to_epsg()
    _2019_gdf = gpd.read_file(gpkg, layer='2019', engine='pyogrio')
    assert _2012_gdf.shape[0] == 1
    assert _2019_gdf.shape[0] == 1
    _2012_polygon = _2012_gdf.geometry[0]
    _2019_polygon = _2019_gdf.geometry[0]
    _2012_2019_diff = _2019_polygon.difference(_2012_polygon)
    estimation_surface = gpd.read_file(
        gpkg, layer='estimation_surfaces', engine='pyogrio'
    ).geometry.union_all().intersection(_2019_polygon)
    dem_processor = DEM(dem, epsg=EPSG)
    dem_truth_processor = DEM(truth_dem, epsg=EPSG)
    if CACHE:
        print('reading cached features...')

        def read(filenames: list[str]):
            for filename in filenames:
                geoseries = gpd.read_file(
                    out_folder / filename,
                    engine='pyogrio',
                ).geometry
                # print('read', filename, geoseries)
                yield geoseries

        jobs = (
            f'{model}_2019_merged.shp',
            f'{model}_pour_points.shp',
            f'{model}_merged_downstream.shp',
            f'{model}_profiles_2012.shp',
            f'{model}_profiles.shp',
            f'{model}_centerlines.shp',
            f'{model}_centerlines_snapped.shp',
        )

        results = read(jobs)
        _2019_merged = next(results)
        pour_points = next(results)
        merged_downstream = next(results)
        profiles_2012 = next(results)
        profiles = next(results)
        centerlines = next(results)
        centerlines_snapped = next(results)
        print('finished reading cached features')
    else:
        _2012_centerline = get_centerline(_2012_gdf.geometry[0], _2012_gdf.crs)
        _2012_centerline.to_file(out_folder / f'{model}_2012_centerline.shp')
        _2012_centerline_types = CenterlineTypes.from_linestrings(
            _2012_centerline
        )
        _2019_centerline = get_centerline(_2019_gdf.geometry[0], _2019_gdf.crs)
        _2019_centerline.to_file(out_folder / f'{model}_2019_centerline.shp')
        _2019_centerline_types = CenterlineTypes.from_linestrings(
            _2019_centerline
        )

        _2012_centerline_types.clean_orphaned()
        _2019_centerline_types.clean_orphaned()

        _2019_merged = merge_linestrings(*_2019_centerline_types)
        _2019_merged.to_file(out_folder / f'{model}_2019_merged.shp')
        pour_points = get_pour_points(_2019_merged, _2012_2019_diff)
        pour_points.to_file(out_folder / f'{model}_pour_points.shp')
        merged_downstream = gpd.GeoSeries(  # type: ignore
            merge_downstream(
                _2019_centerline_types, pour_points,
                _2012_2019_diff
            ),
            crs=_2012_gdf.crs
        )
        merged_downstream.to_file(out_folder / f'{model}_merged_downstream.shp')
        profiles_2012 = dem_processor.line_profiles(
            pour_points,
            out_file=out_folder / f'{model}_profiles_2012.shp'
        ).geometry
        # profiles_2012 = profiles_2012.intersection(_2012_polygon)
        gully_bed = map_centerlines_and_profiles(
            merged_downstream,
            profiles_2012,
            pour_points,
            dem_processor.size_x
        )
        gully_beds = list(gully_bed)
        centerlines_snapped = [
            extend_line_to_geom(gully_bed.centerline, _2019_polygon.boundary)
            for gully_bed in gully_beds
        ]
        centerlines = [gully_bed.centerline for gully_bed in gully_beds]
        profiles = [gully_bed.profile for gully_bed in gully_beds]

        gpd.GeoSeries(centerlines_snapped, crs=EPSG).to_file(
            out_folder / f'{model}_centerlines.shp'
        )
        gpd.GeoSeries(centerlines_snapped, crs=EPSG).to_file(
            out_folder / f'{model}_centerlines_snapped.shp'
        )
        gpd.GeoSeries(profiles, crs=EPSG).to_file(
            out_folder / f'{model}_profiles.shp'
        )
    profile_out_dir = None
    if DEBUG >= 1:
        profile_out_dir = out_folder / 'estimation'
        profile_out_dir.mkdir(exist_ok=True)

    estimations, _ = estimate_gully_beds(
        profiles,
        centerlines_snapped,
        dem_processor,
        EPSG,
        penalty=PENALTY,
        profile_out_dir=profile_out_dir
    )
    # estimations.to_file(out_folder / 'estimations.shp')
    estimations_agg = t.cast(gpd.GeoDataFrame, aggregate_overlapping_points(estimations, 'Z', 'min'))
    # estimations_agg.to_file(out_folder / 'estimations_agg.shp')
    limit_sample = dem_processor.sample([_2019_polygon.boundary], epsg=EPSG)
    interpolation = DEM.from_raster(
        multilevel_b_spline(
            pd.concat([estimations_agg, limit_sample], ignore_index=True)[['Z', 'geometry']],
            dem_processor.size_x,
            elevation_field='Z'
        ))
    interpolation_filtered = interpolation.gaussian_filter()
    gully_cover = DEM.from_raster(
        inverse_distance_weighted(
            limit_sample,
            cell_size=dem_processor.size_x,
            power=2,
            elevation_field='Z'
        ))
    dems = align_rasters(
        [dem_processor, interpolation_filtered,
         dem_truth_processor, gully_cover],
        reference_raster=dem_processor
    )
    masked_dems = [
        DEM.from_raster(dem.apply_mask(estimation_surface)) for dem in dems
    ]
    if EVAL:
        evaluator = Evaluator(
            *masked_dems,
            estimation_surface=estimation_surface
        )
        evaluator.evaluate()
        print('Estimated for', model)
    # breakpoint()
    if DEBUG >= 1:
        import shutil
        limit_sample.to_file(out_folder / 'limit_sample.shp')
        estimations.to_file(out_folder / 'estimations.shp')
        estimations_agg.to_file(out_folder / 'estimations_agg.shp')

        if (out_folder / 'interpolation').exists():
            for file in (out_folder / 'interpolation').iterdir():
                file.unlink()
            (out_folder / 'interpolation').rmdir()
        interpolation_folder = Path(shutil.move(
            interpolation.path.parent,
            out_folder.resolve()
        ))
        interpolation_folder.rename(
            interpolation_folder.with_name('interpolation')
        )

        if (out_folder / 'cover').exists():
            for file in (out_folder / 'cover').iterdir():
                file.unlink()
            (out_folder / 'cover').rmdir()
        interpolation_folder = Path(shutil.move(
            gully_cover.path.parent,
            out_folder.resolve()
        ))
        interpolation_folder.rename(
            interpolation_folder.with_name('cover')
        )

        print(estimations)
        print(interpolation)


Models = t.Literal[
    'soldanesti_aval',
    'soldanesti_amonte',
    'saveni_aval',
    'saveni_amonte'
]

MODELS = [
    'soldanesti_aval', 'saveni_amonte', 'saveni_aval', 'soldanesti_amonte'
]


def main(model: Models):
    dem = Path(f'./data/{model}_2012.tif')
    if (path := Path(f'./data/{model}_2019_mbs.asc')).exists():
        dem_truth = path
    else:
        dem_truth = path.with_suffix('.tif')
    gpkg = Path(f'./data/{model}.gpkg')
    out_folder = Path('./data/derived')
    run(gpkg, dem, dem_truth, out_folder)


if __name__ == '__main__':
    if MODEL is not None:
        main(model=MODEL)
    else:
        if MULTIPROCESSING:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(MODELS)
            ) as executor:
                results = executor.map(main, MODELS)
                for result in results:
                    print(result)
        else:
            for model in MODELS:
                main(model=model)
