from pathlib import Path
from typing import Literal

import geopandas as gpd
import shapely
from gully_automation.centerline import (
    get_centerline,
    merge_linestrings,
    CenterlineTypes,
    plot,
    endpoints_intersecting_boundary,
    merge_downstream
)
from gully_automation.dem import DEM
from gully_automation.utils import vector_layers_to_geodataframe


def run(gpkg: Path, dem: Path, out_folder: Path, use_cached=True):
    try:
        _2003_gdf = gpd.read_file(gpkg, layer='2003')
    except:
        ...
    _2012_gdf = gpd.read_file(gpkg, layer='2012')
    _2019_gdf = gpd.read_file(gpkg, layer='2019')
    _2012_polygon = _2012_gdf.geometry[0]
    _2019_polygon = _2019_gdf.geometry[0]
    _2012_2019_diff = _2019_polygon.difference(_2012_polygon)

    if use_cached:
        _2012_centerline_types = CenterlineTypes.from_linestrings(
            gpd.read_file(out_folder / '2012_centerline.shp')
        )
        _2019_centerline_types = CenterlineTypes.from_linestrings(
            gpd.read_file(out_folder / '2019_centerline.shp')
        )
        _2019_merged = gpd.read_file(out_folder / '2019_merged.shp')
        pour_points = gpd.read_file(out_folder / 'pour_points.shp')
        merged_downstream = gpd.read_file(out_folder / 'merged_downstream.shp')
    else:
        _2012_centerline = get_centerline(_2012_gdf.geometry[0], _2012_gdf.crs)
        _2012_centerline.to_file(out_folder / '2012_centerline.shp')
        _2012_centerline_types = CenterlineTypes.from_linestrings(
            _2012_centerline
        )

        _2019_centerline = get_centerline(_2019_gdf.geometry[0], _2019_gdf.crs)
        _2019_centerline.to_file(out_folder / '2019_centerline.shp')
        _2019_centerline_types = CenterlineTypes.from_linestrings(
            _2019_centerline
        )

        _2012_centerline_types.clean_orphaned()
        _2012_centerline_types.contiguous.to_file(out_folder / '2012_centerlines_contiguous.shp')
        _2012_centerline_types.discontiguous.to_file(out_folder / '2012_centerlines_discontiguous.shp')

        _2019_centerline_types.clean_orphaned()
        _2019_centerline_types.contiguous.to_file(out_folder / '2019_centerlines_contiguous.shp')
        _2019_centerline_types.discontiguous.to_file(out_folder / '2019_centerlines_discontiguous.shp')

        _2019_merged = merge_linestrings(*_2019_centerline_types)
        _2019_merged.to_file(out_folder / '2019_merged.shp')
        pour_points = endpoints_intersecting_boundary(
                _2019_merged, _2012_gdf.geometry[0]
            )
        pour_points.to_file(out_folder / 'pour_points.shp')
        # profiles_2012_path = DEM(dem).profiles(pour_points)
        # profiles_2012 = vector_layers_to_geodataframe(profiles_2012_path)

        merged_downstream = merge_downstream(
            _2019_centerline_types, pour_points, _2012_2019_diff
        )
        gpd.GeoSeries(merged_downstream).to_file(out_folder / 'merged_downstream.shp')

    # plot(
    #     discontiguous_2012=_2012_centerline_types.discontiguous,
    #     contiguous_2012=_2012_centerline_types.contiguous,
    #     merged_2019=_2019_merged,
    #     _2019_limit=_2019_gdf,
    #     _2012_limit=_2012_gdf,
    #     pour_points=endpoints_intersecting_boundary(
    #         _2019_merged, _2012_gdf.geometry[0]
    #     ),
    # )

    # _2012_centerline_types.extract_discontiguous_endpoints().plot()

    # merge_linestrings(erosion_centerlines.contiguous, erosion_centerlines.discontiguous).to_file('./data/erosion_centerlines_merged.shp')
    # print(difference_d)
    # get_pour_points(erosion_centerlines)


Models = Literal['soldanesti_aval', 'soldanesti_amonte',
                 'saveni_aval', 'saveni_amonte']


def main(model: Models, use_cached=False):
    dem = Path(f'./data/{model}_2012.tif')
    gpkg = Path(f'./data/{model}.gpkg')
    out_folder = Path('./data/derived')
    run(gpkg, dem, out_folder, use_cached=use_cached)


if __name__ == '__main__':
    main(model='soldanesti_amonte', use_cached=False)
