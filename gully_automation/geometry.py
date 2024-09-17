from __future__ import annotations

import itertools
import typing as t
import collections.abc as c
from dataclasses import (
    dataclass,
    field
)
from pathlib import Path

import pandas as pd
from shapely import (
    MultiPolygon,
    MultiLineString,
    Polygon,
    LineString,
    Point,
    Geometry,
    MultiPoint,
    shortest_line
)
from shapely.ops import (
    linemerge,
    transform,
    snap,
    split
)
from shapely.geometry.base import BaseGeometry
import geopandas as gpd
import centerline.geometry
from qgis.PyQt.QtCore import QMetaType

from gully_automation import (
    EPS,
    DEBUG
)
from gully_automation.changepoint import (
    estimate_gully,
    find_changepoints
)

if t.TYPE_CHECKING:
    from gully_automation.raster import DEM


AnyLine: t.TypeAlias = LineString | MultiLineString


def intersects(g1: BaseGeometry, g2: BaseGeometry) -> bool:
    """A wrapper 'intersects' which instead uses the distance."""
    return g1.distance(g2) < EPS


def within(g1: BaseGeometry, g2: BaseGeometry) -> bool:
    """A wrapper 'within' which instead uses the EPS overlap."""
    return g1.intersection(g2).length > EPS


def disjoint(g1: BaseGeometry, g2: BaseGeometry) -> bool:
    """A wrapper 'disjoint' which instead uses the % overlap."""
    return g1.intersection(g2).length < EPS


class Endpoints(t.NamedTuple):

    start: Point
    end: Point

    @staticmethod
    def from_linestring(linestring: LineString) -> Endpoints:
        points = linestring.boundary.geoms
        return Endpoints(
            Point(points[0]), Point(points[-1])
        )

    def intersects(
        self,
        other: Geometry, how: t.Literal['any', 'all'] = 'any'
    ) -> bool:
        if how == 'any':
            if any(intersects(endpoint, other) for endpoint in self):
                return True
        elif how == 'all':
            if all(intersects(endpoint, other) for endpoint in self):
                return True
        return False

    def shortest_line(self, other: Geometry):
        l1 = shortest_line(self.start, other)
        l2 = shortest_line(self.end, other)
        return l1 if l1.length <= l2.length else l2


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
        self.contiguous = t.cast(gpd.GeoSeries, self.contiguous[~bool_])

        bool_ = self.discontiguous.apply(
            lambda feature: is_orphaned(
                feature, self.contiguous, self.discontiguous
            )
        )
        self.discontiguous = t.cast(gpd.GeoSeries, self.discontiguous[~bool_])

    def extract_discontiguous_endpoints(self) -> gpd.GeoSeries:
        points = []
        for linestring in self.discontiguous:
            coords = linestring.coords
            start, end = Point(coords[0]), Point(coords[-1])
            if not self.contiguous.intersects(start).any():
                points.append(start)
            elif not self.contiguous.intersects(end).any():
                points.append(end)
        return gpd.GeoSeries(
            data=points, crs=self.discontiguous.crs
        )  # type: ignore

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
        ), f'Found {type(linestrings)}, expected GeoSeries.'
        bool_ = linestrings.geometry.map(  # type: ignore
            lambda feature: is_contiguous(  # type: ignore
                feature,
                linestrings
            )
        )
        return CenterlineTypes(
            t.cast(gpd.GeoSeries, linestrings[bool_]),
            t.cast(gpd.GeoSeries, linestrings[~bool_])
        )  # type: ignore


def merge_downstream(
    centerline_types: CenterlineTypes,
    pour_points: gpd.GeoSeries,
    split_limit: Polygon | MultiPolygon,
) -> list[LineString]:
    # NOTE: Works but should be optimized!!!

    pour_points_multipoint = MultiPoint(pour_points)

    if split_limit.has_z:
        split_limit = to_2d(split_limit)

    # if within_limit.has_z:
    #     within_limit = to_2d(within_limit)

    def process_inputs(lines: gpd.GeoSeries):
        intersecting_centerlines_bool = lines.apply(
            lambda line: intersects(line, pour_points_multipoint)
        )
        split_lines = (
            lines[intersecting_centerlines_bool]
            .apply(lambda line: split(line, split_limit))
            .explode(index_parts=False, ignore_index=True)
        )
        split_lines = pd.concat(
            [lines[~intersecting_centerlines_bool], split_lines],
            ignore_index=True
        )
        intersection = (
            split_lines
            .intersection(split_limit)
            .explode(index_parts=False, ignore_index=True)
        )
        line_type = (LineString, MultiLineString)
        is_line = intersection.apply(
            lambda geometry: isinstance(geometry, line_type)
        )
        not_empty = ~intersection.is_empty
        is_starting_line = intersection.apply(
            lambda line: intersects(line, pour_points_multipoint)
        )
        intersects_limit = intersection.apply(
            lambda line: intersects(line, split_limit.boundary)
        )
        longer_than_epsg = intersection.length > EPS
        # is_starting_line == intersects_limit means that the lines
        # which intersect the boundary but do not intersects the
        # pour points are filtered out
        return intersection[
            is_line
            & not_empty
            & (is_starting_line == intersects_limit)
            & longer_than_epsg
        ]

    assert (
        (crs1 := centerline_types.contiguous.crs)
        == (crs2 := centerline_types.discontiguous.crs)
        == (crs3 := pour_points.crs)
    ), f"CRS mismatch from input: '{crs1}', '{crs2}', '{crs3}'"

    crs = centerline_types.contiguous.crs
    contiguous_lines = process_inputs(centerline_types.contiguous)
    discontiguous_lines = process_inputs(centerline_types.discontiguous)
    first_lines = pd.concat(
        [contiguous_lines[contiguous_lines.intersects(pour_points_multipoint)],
         discontiguous_lines[discontiguous_lines.intersects(
             pour_points_multipoint)]],
        ignore_index=True
    )

    merged = []

    # 'i' for debugging
    for discontiguous_line in discontiguous_lines:
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
            contiguous_lines.apply(lambda line: intersects(
                line, pour_points_multipoint))
        ]
        discontiguous_first_lines = discontiguous_lines[
            discontiguous_lines.apply(
                lambda line: intersects(line, pour_points_multipoint))
        ]
        assert (
            (crs1 := contiguous_first_lines.crs)
            == (crs2 := discontiguous_first_lines.crs)
        ), f'{crs1.epsg} not equal to {crs2.epsg}'
        first_lines = pd.concat(
            [contiguous_first_lines, discontiguous_first_lines],
            ignore_index=True
        )
        assert isinstance(first_lines, gpd.GeoSeries)
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
            assert isinstance(contiguous_type, gpd.GeoSeries)
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
            assert isinstance(contiguous_type, gpd.GeoSeries)
            contiguous_type.loc[
                contiguous_type.shape[0]] = discontiguous_line
            # appending a shapely object makes the geoseries lose the crs
            contiguous_type.set_crs(crs, inplace=True)

        merged_line = [
            discontiguous_line,
        ]
        if not contiguous_type.empty:
            merged_line.extend(contiguous_type.geometry)
        merged_line = linemerge(merged_line)
        first_line = first_lines[first_lines.intersects(merged_line)]
        assert first_line.shape[0] in (1, 0)
        if not first_line.empty:
            merged_line = linemerge(
                [merged_line, *first_line.geometry.tolist()]
            )
        assert isinstance(merged_line, LineString)
        if intersects(merged_line, split_limit.boundary):
            merged.append(merged_line)
    return merged


def is_contiguous(
    linestring: LineString,
    *other_linestrings: gpd.GeoSeries
):
    if linestring.is_closed:
        return True
    # coords = Geometry.mapping(linestring)['coordinates']
    coords = linestring.boundary.geoms
    first, last = (Point(coords[0]),
                   Point(coords[-1]))
    first_overlaps = 0
    last_overlaps = 0
    for other_ls in itertools.chain.from_iterable(other_linestrings):
        if linestring == other_ls:
            continue  # Skip self-comparison
        assert isinstance(other_ls, LineString)
        if first.intersects(other_ls):
            first_overlaps += 1
        if last.intersects(other_ls):
            last_overlaps += 1
        if first_overlaps != 0 and last_overlaps != 0:
            return True
    return False


def get_centerline(geoms: Polygon, crs) -> gpd.GeoSeries:
    return gpd.GeoSeries(
        centerline.geometry.Centerline(
            geoms, interpolation_distance=0.25).geometry,
        crs=crs
    )  # type: ignore


def merge_linestrings(*linestrings: gpd.GeoSeries) -> gpd.GeoSeries:
    assert len(set(linestring.crs for linestring in linestrings)) == 1
    lines = itertools.chain.from_iterable(
        [
            linestring.geometry.explode(
                ignore_index=True, index_parts=False).values
            for linestring in linestrings
        ]
    )
    merged = linemerge(lines)
    if isinstance(merged, MultiLineString):
        return gpd.GeoSeries(
            merged.geoms, crs=linestrings[0].crs
        )  # type: ignore
    elif isinstance(merged, LineString):
        return gpd.GeoSeries(
            MultiLineString([merged]).geoms, crs=linestrings[0].crs
        )  # type: ignore
    assert False, 'should not have reached here'


def merge_then_explode(*linestrings: gpd.GeoSeries):
    merged = merge_linestrings(*linestrings)
    return merged.explode(ignore_index=True, index_parts=False)


T = t.TypeVar('T', bound=BaseGeometry)


def to_2d(shape: T) -> T:
    """Converts a three dimensional shape to a 2D."""
    if not shape.has_z:
        return shape
    return transform(lambda x, y, _: (x, y), shape)  # type: ignore


def merge_intersecting_lines(linestrings: gpd.GeoSeries) -> gpd.GeoSeries:
    """Merges lines that intersect."""
    count = linestrings.shape[0]
    _linestrings: list[AnyLine] = []
    while True:
        _linestrings = []
        for linestring in linestrings:
            consumed = gpd.GeoSeries(_linestrings, crs=linestrings.crs)
            not_consumed = linestrings[linestrings.apply(
                lambda line: consumed[consumed.contains(line)].empty)]
            if not_consumed.empty:
                continue
            intersecting = not_consumed[not_consumed.intersects(linestring)]
            if intersecting.empty:
                continue
            merged = linemerge(intersecting.explode().tolist())
            _linestrings.append(merged)
        linestrings = gpd.GeoSeries(_linestrings, crs=linestrings.crs)
        if linestrings.shape[0] == count:
            break
        count = linestrings.shape[0]
    return linestrings


def dangles_gen(
    linestrings: gpd.GeoSeries | c.Sequence[AnyLine]
) -> t.Generator[Point, None, None]:
    if not isinstance(linestrings, gpd.GeoSeries):
        linestrings = gpd.GeoSeries(linestrings)
    linestrings_exploded = linestrings.explode(
        index_parts=False, ignore_index=True
    )
    for linestring in linestrings_exploded:
        endpoints = Endpoints.from_linestring(linestring)
        for point in endpoints:
            if linestrings.intersects(point).sum() == 1:
                yield point


def get_pour_points(
    linestrings: gpd.GeoSeries,
    polygon: Polygon,
) -> gpd.GeoSeries:
    """The points that intersect with a given polygon boundary."""

    intersections_poly = (
        linestrings
        .intersection(polygon)
        .explode(index_parts=False, ignore_index=True)
    )
    intersections_poly = intersections_poly[~intersections_poly.is_empty]

    def pour_point_gen() -> t.Generator[Point, None, None]:
        nonlocal intersections_poly
        intersections_poly = t.cast(gpd.GeoSeries, intersections_poly)
        disallowed_points = MultiPoint(get_disallowed_points())
        intersections_poly = intersections_poly[~intersections_poly.intersects(
            disallowed_points)]
        assert isinstance(intersections_poly, gpd.GeoSeries)
        linestrings_preprocessed = merge_intersecting_lines(intersections_poly)
        for linestring in linestrings_preprocessed.geometry:
            dangles_intersecting = []
            for dangle in dangles_gen([linestring]):
                if intersects(dangle, polygon.boundary):
                    dangles_intersecting.append(dangle)
            n_dangles = len(dangles_intersecting)
            assert n_dangles != 0
            if n_dangles == 1:
                yield dangles_intersecting[0]

    def get_disallowed_points() -> list[Point]:
        """The endpoints of the lines that should be omitted."""
        diff = linestrings.difference(polygon)
        diff = diff[~diff.is_empty]
        orphaned = diff[diff.apply(lambda line: is_orphaned(line, diff))]
        dangles = []
        for dangle in dangles_gen(orphaned):
            if intersects(dangle, polygon.boundary):
                dangles.append(dangle)
        for intersection in intersections_poly:
            endpoints = Endpoints.from_linestring(intersection)
            if endpoints.intersects(polygon.boundary, how='all'):
                for endpoint in endpoints:
                    try:
                        dangles.remove(endpoint)
                    except ValueError:
                        continue
        return dangles

    return gpd.GeoSeries(pour_point_gen(), crs=linestrings.crs)


def is_orphaned(
    linestring: LineString,
    *other_linestrings: gpd.GeoSeries
) -> bool:
    """Whether a linestring is orphaned (has no common vertex with others)."""
    for other_ls in itertools.chain(*other_linestrings):
        if linestring == other_ls:
            continue  # Skip self-comparison
        if linestring.intersects(other_ls):
            return False
    return True


@dataclass
class GullyBed:
    centerline: LineString
    profile: LineString
    shortest_line: LineString = field(init=False)

    def __post_init__(self):
        # By doing this, we make sure that the
        # centerline and the profile are continuous.
        centerline_endpoints = Endpoints.from_linestring(self.centerline)
        profile_endpoints = Endpoints.from_linestring(self.profile)
        l1 = profile_endpoints.shortest_line(centerline_endpoints.start)
        l2 = profile_endpoints.shortest_line(centerline_endpoints.end)
        if l1.length <= l2.length:
            self.centerline = self.centerline.reverse()
        self.shortest_line = l1 if l1.length <= l2.length else l2

    @property
    def merged(self) -> LineString:
        return self.merge()

    def merge(self) -> LineString:
        merged = linemerge(
            [self.profile, self.shortest_line, self.centerline]
        ).reverse()
        assert isinstance(merged, LineString), 'Expected linestring.'
        return merged


def estimate_gully_beds(
    gully_beds: list[GullyBed],
    dem: DEM,
    epsg: str,
    profile_out_dir: Path | None = None
) -> tuple[gpd.GeoDataFrame, list[Point]]:
    profiles = [bed.profile for bed in gully_beds]
    centerlines = [bed.centerline for bed in gully_beds]
    changepoints: list[Point] = []
    profile_sample = dem.sample(profiles, epsg)
    centerline_sample = dem.sample(centerlines, epsg)

    line_id_col = (
        'LINE_ID' if 'LINE_ID' in profile_sample
        else 'ID_LINE' if 'ID_LINE' in profile_sample
        else None
    )

    def estimate(id_) -> gpd.GeoDataFrame:
        import matplotlib.pyplot as plt
        profile: gpd.GeoDataFrame = profile_sample.loc[
            profile_sample[line_id_col] == id_
        ]
        centerline: gpd.GeoDataFrame = centerline_sample.loc[
            centerline_sample[line_id_col] == id_
        ]
        changepoint = find_changepoints(profile['Z'].values)[0]
        changepoints.append(profile.geometry.iloc[changepoint])
        estimation = estimate_gully(
            profile['Z'].values,
            centerline['Z'].values,
            changepoint
        )
        if profile_out_dir is not None and DEBUG >= 1:
            print('Saving profile', id_, 'in', profile_out_dir, end='\r')
            plt.savefig(profile_out_dir / f'{id_}.png')
            plt.close()
        estimation_profile = pd.concat(
            [centerline.geometry, profile.geometry],
            ignore_index=True
        )
        estimation_profile['Z'] = estimation
        estimation_profile[line_id_col] = id_
        return estimation_profile  # type: ignore

    estimations = None
    for id_ in profile_sample[line_id_col].unique():  # type: ignore
        estimation = estimate(id_)
        if estimations is None:
            estimations = estimation
        else:
            estimations = pd.concat(
                [estimations, estimation],
                ignore_index=True
            )
    return (estimations, changepoints)  # type: ignore


def snap_to_nearest(
    geom: Geometry,
    other: gpd.GeoSeries
):
    idx = other.distance(geom).sort_values().index[0]
    return snap(
        geom,
        other.loc[idx],
        tolerance=EPS
    )


def extend_line_to_geom(
    line: LineString,
    geom: Geometry,
    endpoint: t.Literal['start', 'end'] = 'start'
) -> LineString:
    point = getattr(Endpoints.from_linestring(line), endpoint)
    return linemerge(
        [shortest_line(point, geom), line]
    )  # type: ignore


def map_centerlines_and_profiles(
    centerlines: gpd.GeoSeries,
    profiles: gpd.GeoSeries,
    pour_points: gpd.GeoSeries,
    grid_size: float
) -> t.Generator[GullyBed, None, None]:
    # The distance between the centerline and the profile
    # should not be bigger than the pixel size of the DEM.
    for centerline_ in centerlines:
        pour_point = pour_points[
            pour_points.apply(lambda point: intersects(point, centerline_))
        ]
        assert (
            pour_point.shape[0] in (0, 1)
        ), f'Expected zero or one pour points, found {pour_point.shape[0]}.'
        if pour_point.shape[0] == 0:
            print(pour_point)
            continue
        pour_point = pour_point.iloc[0]
        # Drop duplicates because it's possible there are duplicates
        gdf: gpd.GeoDataFrame = (
            profiles
            .to_frame()
            .reset_index(drop=True)
        )  # type: ignore
        gdf['distance'] = gdf.distance(pour_point)
        gdf.sort_values(by='distance', inplace=True)
        profile = gdf.iloc[0]
        if profile['distance'] > grid_size * 2:
            # It's possible that some pour points do not have profiles.
            # E.g when the user does not correctly define
            # the boundary of the gully, and the profile falls
            # outside the boundary.
            continue
        profile = profile.iloc[0]
        assert isinstance(profile, LineString)
        assert isinstance(centerline_, LineString)
        yield GullyBed(centerline_, profile)


def aggregate_overlapping_points(
    points: gpd.GeoDataFrame,
    value_field: str,
    method: t.Literal['min', 'max', 'mean', 'median'] = 'max'
):
    df = (
        points[[value_field, points.active_geometry_name]]
        .groupby(points.active_geometry_name)  # type: ignore
        .agg(method)
        .reset_index()
    )
    return gpd.GeoDataFrame(
        df,
        geometry=points.active_geometry_name,
        crs=points.crs
    )  # type: ignore
