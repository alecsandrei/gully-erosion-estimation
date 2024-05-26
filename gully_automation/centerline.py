import itertools
from typing import NamedTuple, Union, Tuple, TypeAlias
import random

import shapely.algorithms
import shapely.algorithms.cga
import shapely.algorithms.polylabel
import shapely.geometry
import shapely.geometry.geo
import shapely.geos
import shapely.ops
import geopandas as gpd
import centerline.geometry
import matplotlib.colors
import matplotlib.pyplot as plt


class Endpoints(NamedTuple):

    start: shapely.Point
    end: shapely.Point

    @staticmethod
    def from_linestring(linestring: shapely.LineString):
        coords = linestring.coords
        return Endpoints(
            shapely.Point(coords[0]), shapely.Point(coords[-1])
        )


class CenterlineTypes(NamedTuple):
    """Stores two types of centerlines.

    Contiguous lines are lines that have the first and last vertex
    overlapping with other lines.
    Discontiguous lines are lines that have at most one vertex overlapping
    with other lines.
    This object does not allow multipart geometries.
    """
    contiguous: gpd.GeoSeries
    discontiguous: gpd.GeoSeries

    def clean_orphaned(self):
        """Cleans orphaned contiguous lines."""
        bool_ = self.contiguous.apply(
            lambda feature: is_orphaned(
                feature, self.contiguous, self.discontiguous
            )
        )
        self._replace(contiguous=self.contiguous[~bool_])

        bool_ = self.discontiguous.apply(
            lambda feature: is_orphaned(
                feature, self.discontiguous, self.discontiguous
            )
        )
        self._replace(discontiguous=self.discontiguous[~bool_])

    def merge_downstream(self):
        """Merges downstream centerlines."""

        def compute_discontiguous(linestring: shapely.LineString):
            ...

    def extract_discontiguous_endpoints(self) -> gpd.GeoSeries:
        points = []
        for linestring in self.discontiguous:
            coords = linestring.coords
            start, end = shapely.Point(coords[0]), shapely.Point(coords[-1])
            if not self.contiguous.intersects(start).any():
                points.append(start)
            elif not self.contiguous.intersects(end).any():
                points.append(end)
        return gpd.GeoSeries(points, crs=self.discontiguous.crs)

    @staticmethod
    def from_linestrings(*linestrings: gpd.GeoSeries):
        linestrings = merge_then_explode(*linestrings)
        assert isinstance(linestrings, gpd.GeoSeries)
        bool_ = linestrings.geometry.map(
            lambda feature: is_contiguous(feature, linestrings)
        )
        return CenterlineTypes(
            linestrings[bool_],
            linestrings[~bool_]
        )  # type: ignore


def is_contiguous(
    linestring: shapely.LineString, *other_linestrings: gpd.GeoSeries
):
    coords = shapely.geometry.mapping(linestring)['coordinates']
    first, last = (shapely.Point(coords[0]),
                   shapely.Point(coords[-1]))
    first_overlaps = 0
    last_overlaps = 0
    for other_ls in itertools.chain(*other_linestrings):
        if linestring == other_ls:
            continue  # Skip self-comparison
        assert isinstance(other_ls, shapely.LineString)
        if other_ls.intersects(first):
            first_overlaps += 1
        if other_ls.intersects(last):
            last_overlaps += 1
    if all([first_overlaps, last_overlaps]):
        return True
    return False


def get_centerline(geoms: shapely.Polygon, crs) -> gpd.GeoSeries:
    return gpd.GeoSeries(
        centerline.geometry.Centerline(geoms).geometry,
        crs=crs
    )  # type: ignore


def merge_linestrings(*linestrings: gpd.GeoSeries) -> gpd.GeoSeries:
    assert len(set(linestring.crs for linestring in linestrings)) == 1
    lines = itertools.chain.from_iterable(
        [
            linestring.geometry.explode(ignore_index=True).values
            for linestring in linestrings
        ]
    )
    return gpd.GeoSeries(
        shapely.ops.linemerge(lines).geoms, crs=linestrings[0].crs
    )  # type: ignore


def merge_then_explode(*linestrings: gpd.GeoSeries):
    merged = merge_linestrings(*linestrings)
    return merged.explode(ignore_index=True)


def endpoints_intersecting_boundary(
    linestrings: gpd.GeoSeries,
    polygon: shapely.Polygon,
) -> gpd.GeoSeries:
    """The points that intersect with a given polygon boundary."""
    points = []
    for linestring in linestrings:
        intersection = shapely.intersection(linestring, polygon.boundary)
        if not isinstance(intersection, shapely.Point):
            continue
        points.append(intersection)
    return gpd.GeoSeries(points, crs=linestrings.crs)


def get_linestring_endpoints(
    linestrings: gpd.GeoSeries
) -> list[Endpoints]:
    endpoints = []
    for linestring in linestrings:
        endpoints.append(Endpoints.from_linestring(linestring))
    return endpoints


def geometry_difference(
    geom: gpd.GeoSeries,
    polygon: shapely.Polygon,
) -> gpd.GeoSeries:
    """Gets the difference between a polygon and a geometry."""
    diff = geom.difference(polygon)
    return diff[~diff.is_empty]


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
    plt.savefig('test.png')


def is_orphaned(
    linestring: shapely.LineString, *other_linestrings: gpd.GeoSeries
):
    """Whether a linestring is orphaned (has no common vertex with others)."""
    for other_ls in itertools.chain(*other_linestrings):
        if linestring == other_ls:
            continue  # Skip self-comparison
        assert isinstance(other_ls, shapely.LineString)
        if linestring.intersects(other_ls):
            return False
    return True
