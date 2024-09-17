from __future__ import annotations

from pathlib import Path
import typing as t
import concurrent.futures

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from gully_automation import DEBUG, CACHE, MODEL, EVAL
from gully_automation.geometry import (
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
from gully_automation.raster import DEM, multilevel_b_spline, Evaluator, inverse_distance_weighted, align_rasters
from gully_automation.changepoint import find_changepoints, plot_changepoints


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
    _2012_gdf = gpd.read_file(gpkg, layer='2012', engine='fiona')
    EPSG = _2012_gdf.crs.to_epsg()
    _2019_gdf = gpd.read_file(gpkg, layer='2019', engine='fiona')
    _2012_polygon = _2012_gdf.geometry[0]
    _2019_polygon = _2019_gdf.geometry[0]
    _2012_2019_diff = _2019_polygon.difference(_2012_polygon)
    dem_processor = DEM(dem, epsg=EPSG)
    dem_truth_processor = DEM(truth_dem, epsg=EPSG)
    if CACHE:
        _2012_centerline_types = CenterlineTypes.from_linestrings(
            gpd.read_file(out_folder / f'{model}_2012_centerline.shp')
        )
        _2019_centerline_types = CenterlineTypes.from_linestrings(
            gpd.read_file(out_folder / f'{model}_2019_centerline.shp')
        )
        _2019_merged = gpd.read_file(out_folder / f'{model}_2019_merged.shp')
        pour_points = gpd.read_file(out_folder / f'{model}_pour_points.shp').geometry
        merged_downstream = gpd.read_file(out_folder / f'{model}_merged_downstream.shp').geometry
        profiles_2012 = gpd.read_file(out_folder / f'{model}_profiles_2012.shp').geometry
        # profile_sample = gpd.read_file(out_folder / f'{model}_profile_samples.shp')
        # centerline_sample = gpd.read_file(out_folder / f'{model}_centerline_samples.shp')
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
    gully_bed = map_centerlines_and_profiles(
        merged_downstream,
        profiles_2012,
        pour_points,
        dem_processor.size_x
    )
    gully_beds = list(gully_bed)
    centerlines_snapped = [
        extend_line_to_geom(gully_bed.centerline, _2019_polygon)
        for gully_bed in gully_beds
    ]
    profiles_snapped = [
        extend_line_to_geom(gully_bed.profile, _2019_polygon)
        for gully_bed in gully_beds
    ]
    centerline_sample = dem_processor.sample(
        centerlines_snapped, EPSG
    )
    centerline_sample.to_file(
        out_folder / f'{model}_centerline_samples.shp'
    )
    profile_sample = dem_processor.sample(
        profiles_snapped, EPSG
    )
    profile_sample.to_file(
        out_folder / f'{model}_profile_samples.shp'
    )

    def debug_profiles():
        profiles_dir = out_folder / 'profiles'
        profiles_dir.mkdir(exist_ok=True)
        for id_ in profile_sample['ID_LINE'].unique():
            fig, ax = plt.subplots(figsize=(12, 5))
            subset = profile_sample.loc[profile_sample['ID_LINE'] == id_, 'Z'].reset_index(drop=True)
            subset_before = centerline_sample.loc[centerline_sample['ID_LINE'] == id_, 'Z'].reset_index(drop=True)
            subset_before.index = range(subset.index[-1] + 1, subset_before.shape[0] + subset.index[-1] + 1)
            subset_before.plot(ax=ax)
            subset.plot(ax=ax)
            plt.savefig(profiles_dir / f'{id_}.png')
            plt.close()

    profile_out_dir = None
    if DEBUG >= 1:
        profile_out_dir = out_folder / 'estimation'
        profile_out_dir.mkdir(exist_ok=True)

    estimations, changepoints = estimate_gully_beds(
        gully_beds,
        dem_processor,
        EPSG,
        profile_out_dir=profile_out_dir
    )
    gpd.GeoSeries(changepoints).to_file(out_folder / f'{model}_changepoints.shp')
    estimations_agg = t.cast(gpd.GeoDataFrame, aggregate_overlapping_points(estimations, 'Z', 'min'))
    limit_sample = dem_processor.sample([_2019_polygon.boundary], epsg=EPSG)
    interpolation = DEM.from_raster(
        multilevel_b_spline(
            pd.concat([estimations_agg, limit_sample], ignore_index=True)[['Z', 'geometry']],
            dem_processor.size_x,
            elevation_field='Z'
        ))
    gully_cover = DEM.from_raster(
        inverse_distance_weighted(
            limit_sample,
            cell_size=dem_processor.size_x,
            power=2,
            elevation_field='Z'
        ))
    dems = align_rasters(
        [dem_processor, interpolation, dem_truth_processor, gully_cover],
        reference_raster=dem_processor
    )

    estimation_surface = _2012_2019_diff
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
    dem_truth = Path(f'./data/{model}_2019_mbs.asc')
    gpkg = Path(f'./data/{model}.gpkg')
    out_folder = Path('./data/derived')
    run(gpkg, dem, dem_truth, out_folder)


if __name__ == '__main__':
    if MODEL is not None:
        main(model=MODEL)
    else:
        # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        #     executor.map(main, MODELS)
        for model in MODELS[::-1]:
            main(model=model)
