import itertools
from pathlib import Path
from functools import partial
from typing import NamedTuple

import shapely.geometry
import shapely.ops
import geopandas as gpd
import centerline.geometry
import matplotlib.pyplot as plt


class CenterlineTypes(NamedTuple):
    continuous: gpd.GeoSeries
    discontinuous: gpd.GeoSeries


def read(file) -> gpd.GeoDataFrame:
    return gpd.read_file(file)


def get_centerline(gdf: gpd.GeoSeries) -> gpd.GeoSeries:
    return gpd.GeoSeries(
        centerline.geometry.Centerline(gdf.geometry[0]).geometry,
        crs=gdf.crs
    )  # type: ignore


def preprocess_multi_line(multi_line: gpd.GeoSeries):
    merged = shapely.ops.linemerge(multi_line.geometry[0])
    exploded = (
        gpd.GeoSeries(merged, crs=multi_line.crs)  # type: ignore
        .explode(ignore_index=True)
    )
    return exploded


def get_difference(polygon: shapely.geometry.Polygon, geom: gpd.GeoSeries):
    """Gets the difference between a polygon and a geometry."""
    diff = geom.difference(polygon)
    return diff[~diff.is_empty]


def plot(*series: gpd.GeoSeries):
    fig = plt.figure()
    ax = fig.add_subplot()
    for serie, color in zip(series, itertools.cycle(['b', 'r', 'y', 'g'])):
        serie.plot(ax=ax, color=color)
    plt.show()
    plt.savefig('test.png')


def get_pour_points(erosion_centerlines: CenterlineTypes):
    for linestring in erosion_centerlines.discontinuous:
        if is_orphaned(linestring, *erosion_centerlines):
            print(linestring)


def is_orphaned(
    linestring: shapely.geometry.LineString, *other_linestrings: gpd.GeoSeries
):
    """Whether a linestring is orphaned (has no common vertex with others)."""
    for other_ls in itertools.chain(other_linestrings):
        if linestring == other_ls:
            continue  # Skip self-comparison
        assert isinstance(other_ls, shapely.geometry.LineString)
        if linestring.intersects(other_ls):
            return False
    return True


def has_continuity(
    linestring: shapely.geometry.LineString, *other_linestrings: gpd.GeoSeries
):
    coords = shapely.geometry.mapping(linestring)['coordinates']
    first, last = (shapely.geometry.Point(coords[0]),
                   shapely.geometry.Point(coords[-1]))
    first_overlaps = 0
    last_overlaps = 0
    for other_ls in itertools.chain(other_linestrings):
        if linestring == other_ls:
            continue  # Skip self-comparison
        assert isinstance(other_ls, shapely.geometry.LineString)
        if other_ls.intersects(first):
            first_overlaps += 1
        if other_ls.intersects(last):
            last_overlaps += 1
    if all([first_overlaps, last_overlaps]):
        return True
    return False


def split_lines_by_type(linestrings: gpd.GeoSeries) -> CenterlineTypes:
    bool_ = linestrings.geometry.map(
        lambda feature: has_continuity(feature, linestrings)
    )
    return CenterlineTypes(
        linestrings[bool_], linestrings[~bool_]  # type: ignore
    )


def run(_2003, _2012, _2019):
    _2003_gdf = read(_2003)
    _2012_gdf = read(_2012)
    _2019_gdf = read(_2019)

    _2012_centerline = get_centerline(_2012_gdf.geometry)
    _2019_centerline = get_centerline(_2019_gdf.geometry)
    continuous, discontinuous = split_lines_by_type(
        preprocess_multi_line(_2019_centerline)
    )
    difference_c = get_difference(_2012_gdf.geometry[0], continuous)
    difference_d = get_difference(_2012_gdf.geometry[0], discontinuous)
    erosion_centerlines = CenterlineTypes(difference_c, difference_d)
    # plot(difference_d, difference_c, _2019_gdf)
    get_pour_points(erosion_centerlines)


if __name__ == '__main__':
    _2003 = Path('data', '2003.shp')
    _2012 = Path('data', '2012.shp')
    _2019 = Path('data', '2019.shp')

    run(_2003, _2012, _2019)
