from __future__ import annotations

import itertools
from typing import (
    NamedTuple,
    Literal,
    cast,
    Generator
)
from dataclasses import dataclass

import pandas as pd
import shapely.algorithms
import shapely.algorithms.cga
import shapely.algorithms.polylabel
import shapely.geometry
import shapely.geometry.geo
import shapely.geos
import shapely.ops
import shapely
import geopandas as gpd
import centerline.geometry

from gully_automation import EPS


def intersects(g1, g2) -> bool:
    """A wrapper 'intersects' which instead uses the distance."""
    return g1.distance(g2) < EPS


class Endpoints(NamedTuple):

    start: shapely.Point
    end: shapely.Point

    @staticmethod
    def from_linestring(linestring: shapely.LineString) -> Endpoints:
        points = linestring.boundary.geoms
        return Endpoints(
            shapely.Point(points[0]), shapely.Point(points[-1])
        )

    def intersects(
        self,
        other: shapely.Geometry, how: Literal['any', 'all'] = 'any'
    ) -> bool:
        if how == 'any':
            if any(intersects(endpoint, other) for endpoint in self):
                return True
        elif how == 'all':
            if all(intersects(endpoint, other) for endpoint in self):
                return True
        return False


@dataclass
class CenterlineTypes:
    """Stores two types of centerlines.

    Contiguous lines are lines that have the first and last vertex
    overlapping with other lines.
    Discontiguous lines are lines that have at most one vertex overlapping
    with other lines.
    This object does not allow multipart geometries.
    """
    contiguous: gpd.GeoSeries
    discontiguous: gpd.GeoSeries

    def __iter__(self):
        for linestrings in (self.contiguous, self.discontiguous):
            yield linestrings

    def clean_orphaned(self):
        """Cleans orphaned contiguous lines."""
        bool_ = self.contiguous.apply(
            lambda feature: is_orphaned(
                feature, self.contiguous, self.discontiguous
            )
        )
        self.contiguous = cast(gpd.GeoSeries, self.contiguous[~bool_])

        bool_ = self.discontiguous.apply(
            lambda feature: is_orphaned(
                feature, self.contiguous, self.discontiguous
            )
        )
        self.discontiguous = cast(gpd.GeoSeries, self.discontiguous[~bool_])

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
    def from_linestrings(
        *linestrings: gpd.GeoSeries,
        merge=True,
        explode=True
    ):
        assert (
            len(set(linestring.crs for linestring in linestrings)) == 1
            and linestrings[0].crs is not None
        )
        linestrings = pd.concat(
            linestrings, ignore_index=True
        ).reset_index().geometry  # type: ignore
        if merge:
            linestrings = merge_linestrings(linestrings)  # type: ignore
        if explode:
            linestrings = linestrings.explode(  # type: ignore
                ignore_index=True, index_parts=False
            ) 
        assert (
            isinstance(linestrings, gpd.GeoSeries)
        ), f'Found {type(linestrings)}'
        bool_ = linestrings.geometry.map(  # type: ignore
            lambda feature: is_contiguous(  # type: ignore
                feature,
                linestrings
            )
        )
        return CenterlineTypes(
            linestrings[bool_],
            linestrings[~bool_]
        )  # type: ignore


def merge_downstream(
    centerline_types: CenterlineTypes,
    pour_points: gpd.GeoSeries,
    split_limit: shapely.Polygon | shapely.MultiPolygon,
    within_limit:  shapely.Polygon | shapely.MultiPolygon
) -> list[shapely.LineString]:
    # NOTE: Works but should be optimized (use graphs?)

    pour_points_multipoint = shapely.MultiPoint(pour_points)

    if split_limit.has_z:
        split_limit = to_2d(split_limit)

    if within_limit.has_z:
        within_limit = to_2d(within_limit)

    def endpoints_intersect_limit(linestring: shapely.LineString):
        """True if both endpoints intersect the limit."""
        endpoints = Endpoints.from_linestring(linestring)
        return endpoints.intersects(split_limit, how='all')

    # def preprocess_inputs(lines: gpd.GeoSeries):
    #     lines = (
    #         lines
    #         .clip(split_limit)
    #         .explode(ignore_index=True, index_parts=False)
    #     )
    #     lines = lines[
    #         ~lines.apply(endpoints_intersect_limit)  # type: ignore
    #     ]
    #     return lines

    def process_inputs_2(lines: gpd.GeoSeries):
        intersecting_centerlines_bool = lines.apply(
            lambda line: intersects(line, pour_points_multipoint)
        )
        split_lines = (
            lines[intersecting_centerlines_bool]
            .apply(lambda line: shapely.ops.split(line, split_limit))
            .explode(index_parts=False, ignore_index=True)
        )
        split_lines = pd.concat(
            [lines[~intersecting_centerlines_bool], split_lines],
            ignore_index=True
        )
        return split_lines[split_lines.within(within_limit)]

    assert (
        (crs1 := centerline_types.contiguous.crs)
        == (crs2 := centerline_types.discontiguous.crs)
        == (crs3 := pour_points.crs)
    ), f"CRS mismatch from input: '{crs1}', '{crs2}', '{crs3}'"

    crs = centerline_types.contiguous.crs
    contiguous_lines = process_inputs_2(centerline_types.contiguous)
    discontiguous_lines = process_inputs_2(centerline_types.discontiguous)
    first_lines = pd.concat(
        [contiguous_lines[contiguous_lines.intersects(pour_points_multipoint)],
         discontiguous_lines[discontiguous_lines.intersects(
             pour_points_multipoint)]],
        ignore_index=True
    )

    merged = []

    # 'i' for debugging
    for i, discontiguous_line in enumerate(discontiguous_lines):
        if discontiguous_line in first_lines:
            merged.append(discontiguous_line)
            continue
        types = CenterlineTypes.from_linestrings(
            contiguous_lines,
            discontiguous_lines,
            merge=False,
            explode=True
        )
        types.clean_orphaned()
        contiguous_type = types.contiguous.reset_index(drop=True)
        # Add back the first lines + the discontiguous line itself
        contiguous_type.loc[
            contiguous_type.shape[0]] = discontiguous_line
        # appending a shapely object makes the geoseries lose the crs
        contiguous_type.set_crs(crs, inplace=True)
        contiguous_first_lines = contiguous_lines[
            contiguous_lines.intersects(pour_points_multipoint)
        ]
        discontiguous_first_lines = discontiguous_lines[
            discontiguous_lines.intersects(pour_points_multipoint)
        ]
        assert (
            (crs1 := contiguous_first_lines.crs)
            == (crs2 := discontiguous_first_lines.crs)
        ), f'{crs1.epsg} not equal to {crs2.epsg}'
        first_lines = pd.concat(
            [contiguous_first_lines, discontiguous_first_lines],
            ignore_index=True
        )
        assert (
            (crs1 := contiguous_type.crs)
            == (crs2 := first_lines.crs)
        ), f'{crs1} not equal to {crs2}'
        contiguous_type = pd.concat(
            [contiguous_type, first_lines],
            ignore_index=True
        )
        assert contiguous_type.crs is not None
        last = None
        while True:
            assert contiguous_type.crs is not None
            types = CenterlineTypes.from_linestrings(
                contiguous_type,
                merge=False,
                explode=True
            )
            contiguous_type = types.contiguous
            current = contiguous_type.shape[0]
            if last is None:
                last = current
            else:
                if last == current:
                    break
                else:
                    last = current
            assert (
                (crs1 := contiguous_type.crs)
                == (crs2 := first_lines.crs)
            ), f'{crs1.epsg} not equal to {crs2.epsg}'
            contiguous_type = pd.concat(
                [contiguous_type, first_lines],
                ignore_index=True
            )
            contiguous_type.loc[
                contiguous_type.shape[0]] = discontiguous_line
            # appending a shapely object makes the geoseries lose the crs
            contiguous_type.set_crs(crs, inplace=True)

        merged_line = [
            discontiguous_line,
        ]
        if not contiguous_type.empty:
            merged_line.extend(contiguous_type.geometry)
        merged_line = shapely.ops.linemerge(merged_line)
        first_line = first_lines[first_lines.intersects(merged_line)]
        assert first_line.shape[0] in (1, 0)
        if not first_line.empty:
            merged_line = shapely.ops.linemerge(
                [merged_line, *first_line.geometry.tolist()]
            )
        assert isinstance(merged_line, shapely.LineString)
        merged.append(merged_line)
    return merged


def is_contiguous(
    linestring: shapely.LineString,
    *other_linestrings: gpd.GeoSeries
):
    if linestring.is_closed:
        return True
    # coords = shapely.geometry.mapping(linestring)['coordinates']
    coords = linestring.boundary.geoms
    first, last = (shapely.Point(coords[0]),
                   shapely.Point(coords[-1]))
    first_overlaps = 0
    last_overlaps = 0
    for other_ls in itertools.chain.from_iterable(other_linestrings):
        if linestring == other_ls:
            continue  # Skip self-comparison
        assert isinstance(other_ls, shapely.LineString)
        if first.intersects(other_ls):
            first_overlaps += 1
        if last.intersects(other_ls):
            last_overlaps += 1
        if first_overlaps != 0 and last_overlaps != 0:
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
            linestring.geometry.explode(ignore_index=True, index_parts=False).values
            for linestring in linestrings
        ]
    )
    merged = shapely.ops.linemerge(lines)
    if isinstance(merged, shapely.MultiLineString):
        return gpd.GeoSeries(
            merged.geoms, crs=linestrings[0].crs
        )  # type: ignore
    elif isinstance(merged, shapely.LineString):
        return gpd.GeoSeries(
            shapely.MultiLineString([merged]).geoms, crs=linestrings[0].crs
        )  # type: ignore
    assert False, 'should not have reached here'


def merge_then_explode(*linestrings: gpd.GeoSeries):
    merged = merge_linestrings(*linestrings)
    return merged.explode(ignore_index=True, index_parts=False)


def to_2d(shape):
    """Converts a three dimensional shape to a 2D."""
    return shapely.ops.transform(lambda x, y, z: (x, y), shape)


def get_pour_points(
    linestrings: gpd.GeoSeries,
    polygon: shapely.Polygon,
) -> gpd.GeoSeries:
    """The points that intersect with a given polygon boundary."""

    points = []
    intersections_poly = linestrings.intersection(polygon)
    intersections_bounds = linestrings.intersection(polygon.boundary)

    for linestring, intersection_poly, intersection_bounds in zip(
        linestrings, intersections_poly, intersections_bounds
    ):
        if intersection_poly.is_empty:
            continue

        # if len(points) == 8 and linestrings[linestrings == linestring].index[0] == 113:
        #     breakpoint()

        disallowed_points: list[shapely.Point] = []
        if isinstance(intersection_poly, shapely.MultiLineString):
            differences = linestring.difference(polygon)
            if isinstance(differences, shapely.LineString):
                differences = shapely.MultiLineString([differences])
            for intersection in differences.geoms:
                endpoints = Endpoints.from_linestring(intersection)
                if endpoints.intersects(polygon.boundary, how='all'):
                    disallowed_points.extend(endpoints)
        elif isinstance(intersection_poly, shapely.LineString):
            endpoints = Endpoints.from_linestring(intersection_poly)
            if (
                endpoints.intersects(polygon.boundary, how='all')
                or is_orphaned(intersection_poly, intersections_poly)
            ):
                disallowed_points.extend(endpoints)
        disallowed_points = shapely.MultiPoint(
            disallowed_points
        )  # type: ignore
        if isinstance(intersection_bounds, shapely.MultiPoint):
            for point in intersection_bounds.geoms:
                if not intersects(point, disallowed_points):
                    points.append(point)
        elif isinstance(intersection_bounds, shapely.Point):
            if not intersects(intersection_bounds, disallowed_points):
                points.append(intersection_bounds)
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
    diff = diff[~diff.is_empty]
    return diff.explode(ignore_index=True, index_parts=False)


def is_orphaned(
    linestring: shapely.LineString, *other_linestrings: gpd.GeoSeries
):
    """Whether a linestring is orphaned (has no common vertex with others)."""
    for other_ls in itertools.chain(*other_linestrings):
        if linestring == other_ls:
            continue  # Skip self-comparison
        # assert isinstance(other_ls, shapely.LineString)
        if linestring.intersects(other_ls):
            return False
    return True


class GullyBed(NamedTuple):
    centerline: shapely.LineString
    profile: shapely.LineString


def map_centerlines_and_profiles(
    centerlines: gpd.GeoSeries,
    profiles: gpd.GeoSeries,
    pour_points: gpd.GeoSeries,
    grid_size: float
) -> Generator[GullyBed, None, None]:
    # The distance between the centerline and the profile
    # should not be bigger than the pixel size of the DEM.
    for centerline_ in centerlines:
        pour_point = pour_points[pour_points.intersects(centerline_)]
        assert (
            pour_point.shape[0] == 1
        ), 'Expected zero or one pour points.'
        pour_point = pour_point.iloc[0]
        # Drop duplicates because it's possible there are duplicate profiles
        # E.g when there are two pour points close, within grid_size
        # distance of each other
        gdf: gpd.GeoDataFrame = (
            profiles
            .to_frame()
            .reset_index(drop=True)
        )  # type: ignore
        gdf['distance'] = gdf.distance(pour_point)
        gdf.sort_values(by='distance', inplace=True)
        profile = gdf.iloc[0]
        # profile = (
        #     profiles[ < grid_size]
        #     .drop_duplicates()
        # )
        # assert (
        #     (profile_shape := profile.shape[0]) in (0, 1)
        # ), 'Expected zero or one profiles.'
        if profile['distance'] > grid_size:
            # It's possible that some pour points do not have profiles.
            # E.g when the user does not correctly define
            # the boundary of the gully.
            continue
        profile = profile.iloc[0]
        assert isinstance(profile, shapely.LineString)
        assert isinstance(centerline_, shapely.LineString)
        yield GullyBed(centerline_, profile)
