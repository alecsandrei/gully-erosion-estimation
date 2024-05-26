from pathlib import Path

import geopandas as gpd

from gully_automation.centerline import (
    get_centerline,
    geometry_difference,
    merge_linestrings,
    CenterlineTypes,
    plot,
    endpoints_intersecting_boundary
)
from gully_automation.dem import DEM


def run(_2003, _2012, _2019):
    dem = Path('/media/alex/alex/gullies/saveni_aval_2012_mbs.asc')
    _2003_gdf = gpd.read_file(_2003)
    _2012_gdf = gpd.read_file(_2012)
    _2019_gdf = gpd.read_file(_2019)

    _2012_centerline = get_centerline(_2012_gdf.geometry[0], _2012_gdf.crs)
    _2012_centerline_types = CenterlineTypes.from_linestrings(
        _2012_centerline
    )
    _2019_centerline = get_centerline(_2019_gdf.geometry[0], _2019_gdf.crs)
    _2019_centerline_types = CenterlineTypes.from_linestrings(
        _2019_centerline
    )
    difference_c = geometry_difference(
        _2019_centerline_types.contiguous, _2012_gdf.geometry[0]
    )
    difference_d = geometry_difference(
        _2019_centerline_types.discontiguous, _2012_gdf.geometry[0]
    )
    _2019_erosion_centerlines = CenterlineTypes(
        difference_c.explode(ignore_index=True),
        difference_d.explode(ignore_index=True)
    )
    _2019_erosion_centerlines.clean_orphaned()
    _2019_erosion_centerlines.contiguous.to_file('./data/2019_erosion_centerlines_continuous.shp')
    _2019_erosion_centerlines.discontiguous.to_file('./data/2019_erosion_centerlines_discontinuous.shp')

    _2012_centerline_types.clean_orphaned()
    _2012_centerline_types.contiguous.to_file('./data/2012_centerlines_continuous.shp')
    _2012_centerline_types.discontiguous.to_file('./data/2012_centerlines_discontiguous.shp')

    _2019_centerline_types.clean_orphaned()
    _2019_centerline_types.contiguous.to_file('./data/2019_centerlines_continuous.shp')
    _2019_centerline_types.discontiguous.to_file('./data/2019_centerlines_discontiguous.shp')

    _2019_merged = merge_linestrings(*_2019_centerline_types)
    pour_points = endpoints_intersecting_boundary(
            _2019_merged, _2012_gdf.geometry[0]
        )
    dem = DEM(dem).profiles(pour_points)

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


if __name__ == '__main__':
    _2003 = Path('data', '2003.shp')
    _2012 = Path('data', '2012.shp')
    _2019 = Path('data', '2019.shp')

    run(_2003, _2012, _2019)
