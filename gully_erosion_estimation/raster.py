"""Used for DEM preprocessing and analysis."""

from __future__ import annotations

import os
import typing as t
from dataclasses import dataclass
from pathlib import Path
from functools import cached_property
import itertools

import geopandas as gpd
from qgis.core import (
    QgsField,
    QgsRasterLayer,
    QgsRectangle,
    QgsCoordinateReferenceSystem,
    QgsProcessingFeedback,
)
from qgis.PyQt.QtCore import QMetaType
from shapely import (
    Polygon,
    LineString,
    MultiPolygon,
    Point
)
import processing

from gully_erosion_estimation import (
    DEBUG,
    EPS
)
from gully_erosion_estimation.geometry import is_orphaned
from gully_erosion_estimation.converter import Converter
from gully_erosion_estimation.utils import vector_layers_to_geodataframe


PathLike = Path | str
AnyPolygon = Polygon | MultiPolygon


class Extent(t.NamedTuple):
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    epsg: str

    @staticmethod
    def from_raster(raster: QgsRasterLayer, epsg: str | None = None):
        extent = raster.extent()
        if epsg is None:
            epsg = raster.crs().geographicCrsAuthId()
        if not epsg:
            raise InvalidCoordinateSystem('EPSG code not provided.')
        return Extent(
            extent.xMinimum(),
            extent.xMaximum(),
            extent.yMinimum(),
            extent.yMaximum(),
            epsg
        )

    def __str__(self):
        """Makes the object usable as an input to QGIS processing."""
        return ' '.join([
            ','.join(str(coord) for coord in self[:4]),
            f'[EPSG:{self.epsg}]'
        ])


class InvalidCoordinateSystem(Exception):
    def __init__(self, msg):
        self.msg = msg
        super().__init__(self.msg)


T = t.TypeVar('T', bound='Raster')


class Raster:

    def __init__(
        self, path: PathLike,
        mask: Polygon | MultiPolygon | None = None,
        epsg: str | None = None
    ):
        self.path = path
        self.mask = mask
        if not isinstance(self.path, Path):
            self.path = Path(self.path)
            if not self.path.exists():
                raise FileNotFoundError('DEM file not found.')
        if not self.qgs.crs().isValid() and epsg is None:
            raise InvalidCoordinateSystem(
                f'{self.path} does not have a valid '
                'coordinate reference system.'
            )
        elif epsg is None:
            self.epsg = self.qgs.crs().geographicCrsAuthId()
        else:
            self.epsg = epsg
        assert self.epsg

    def __str__(self):
        return os.fspath(self.path)

    def __sub__(self, other: Raster) -> Raster:
        return raster_difference(self, other)

    @cached_property
    def masked(self):
        return self.apply_mask()

    @cached_property
    def qgs(self) -> QgsRasterLayer:
        assert isinstance(self.path, Path)
        return QgsRasterLayer(self.path.as_posix())

    @cached_property
    def size_x(self) -> float:
        return self.qgs.rasterUnitsPerPixelX()

    @cached_property
    def size_y(self) -> float:
        return self.qgs.rasterUnitsPerPixelY()

    @cached_property
    def extent(self) -> Extent:
        return Extent.from_raster(self.qgs, epsg=self.epsg)

    def apply_mask(
        self: T,
        mask: Polygon | MultiPolygon | None = None
    ) -> T:
        if mask is None:
            mask = self.mask
        converter = Converter()
        assert mask is not None
        mask = mask.buffer(distance=self.size_x)
        converter.add_polygons([mask])
        assert self.epsg is not None
        mask_as_qgs_vector_layer = converter.to_vector_layer(self.epsg)
        masked = processing.run('sagang:cliprasterwithpolygon', {
            'INPUT': [str(self)],
            'OUTPUT': 'TEMPORARY_OUTPUT',
            'POLYGONS': mask_as_qgs_vector_layer,
            'EXTENT': 1
        })
        return type(self)(
            Path(masked['OUTPUT']), epsg=self.epsg
        )


class DEM(Raster):

    def __init__(
        self, path: PathLike,
        mask: Polygon | MultiPolygon | None = None,
        epsg: str | None = None
    ):
        super().__init__(path, mask, epsg)

    @cached_property
    def dem_preproc(self) -> DEM:
        return self.hydro_preprocessing()

    def hydro_preprocessing(self) -> DEM:
        dem = self
        if self.mask is not None:
            dem = self.masked
        sink_route = processing.run(
            'sagang:sinkdrainageroutedetection', {
                'ELEVATION': str(dem),
                'SINKROUTE': 'TEMPORARY_OUTPUT',
                'THRESHOLD': False,
                'THRSHEIGHT': 100
            }
        )
        dem_preproc = processing.run(
            'sagang:sinkremoval', {
                'DEM': str(dem),
                'SINKROUTE': sink_route['SINKROUTE'],
                'DEM_PREPROC': 'TEMPORARY_OUTPUT',
                'METHOD': 1,
                'THRESHOLD': False,
                'THRSHEIGHT': 100
            }
        )
        return DEM(
            Path(dem_preproc['DEM_PREPROC']),
            self.mask,
            epsg=self.epsg
        )

    @staticmethod
    def from_raster(raster: Raster):
        return DEM(
            raster.path,
            raster.mask,
            raster.epsg
        )

    def sample(
        self,
        shapes: list[LineString],
        epsg: str
    ) -> gpd.GeoDataFrame:
        converter = Converter()
        converter.add_lines(shapes)

        profiles = processing.run('sagang:profilesfromlines', {
            'DEM': str(self.dem_preproc),
            'VALUES': None,
            'LINES': converter.to_vector_layer(epsg),
            'NAME': 'FID',
            'PROFILE': 'TEMPORARY_OUTPUT',
            'PROFILES': 'TEMPORARY_OUTPUT',
            'SPLIT': False
        })
        return gpd.read_file(profiles['PROFILE'])

    def line_profiles(
        self,
        points: t.Sequence[Point],
        epsg: str = '3844',
        out_file: Path | None = None
    ) -> gpd.GeoSeries:
        converter = Converter()
        converter.add_points(points)
        # NOTE: epsg defaults to 3844 for debugging reasons
        layer = converter.to_vector_layer(epsg)

        def add_x_y():
            """Adds x y information to layer.

            Could also use the 'Add X/Y Fields to Layer' tool
            but this approach does not create another layer.
            """
            provider = layer.dataProvider()
            provider.addAttributes(
                [
                    QgsField('x', QMetaType.Type.Double),
                    QgsField('y', QMetaType.Type.Double),
                ]
            )
            if DEBUG >= 1:
                print(
                    f'Adding x y information as field for layer {layer.id()}.'
                )
            layer.updateFields()
            layer.startEditing()
            for i in range(1, layer.featureCount() + 1):
                feature = layer.getFeature(i)
                geom = feature.geometry().asPoint()
                feature.setAttributes(
                    [geom.x(), geom.y()]
                )
                layer.updateFeature(feature)
            layer.commitChanges()

        add_x_y()

        def snap_pour_point_to_closest_grid_cell() -> str:
            grid_points = processing.run('native:pixelstopoints', {
                'INPUT_RASTER': str(self.dem_preproc),
                'RASTER_BAND': 1,
                'FIELD_NAME': 'VALUE',
                'OUTPUT': 'TEMPORARY_OUTPUT'})

            snapped_points = processing.run('native:snapgeometries', {
                'INPUT': layer,
                'REFERENCE_LAYER': grid_points['OUTPUT'],
                'TOLERANCE': 10,
                'BEHAVIOR': 3,
                'OUTPUT': 'TEMPORARY_OUTPUT'})
            return snapped_points['OUTPUT']

        def get_flow_path_profiles():
            # NOTE: this tool will not create a point
            # on the corresponding starting point
            if DEBUG >= 1:
                print('Generating flow path profiles from points.')
            snapped_points = snap_pour_point_to_closest_grid_cell()
            profiles = processing.run(
                'sagang:leastcostpaths', {
                    'SOURCE': snapped_points,
                    'DEM': str(self.dem_preproc),
                    'VALUES': None,
                    'POINTS': 'TEMPORARY_OUTPUT',
                    'LINE': 'TEMPORARY_OUTPUT'})
            return list(Path(profiles['LINE']).parent.glob('*.shp'))

        profiles = get_flow_path_profiles()  # type: ignore
        if not profiles:
            raise FileNotFoundError(
                'No profiles could be generated. Could be a projection issue.'
            )
        assert profiles is not None
        profiles_gdf = vector_layers_to_geodataframe(
            profiles  # type: ignore
        )
        profiles_geoms: gpd.GeoSeries = profiles_gdf.geometry  # type: ignore
        # Check profiles that would fall outside of mask
        # (happens when the user defined the wrong boundary for the gully).
        orphaned = profiles_geoms.apply(lambda profile: is_orphaned(
            profile, profiles_geoms
        ))
        profiles_orphaned = profiles_geoms[orphaned]
        print(
            f'Found {profiles_orphaned.shape[0]} profiles which '
            'would fall outside of the defined boundary'
        )
        profiles_valid = profiles_geoms[~orphaned]  # type: ignore
        if out_file:
            profiles_valid.to_file(out_file)
        return profiles_valid  # type: ignore


def multilevel_b_spline(
    points: gpd.GeoDataFrame,
    cell_size: float,
    elevation_field: str = 'Z'
) -> Raster:
    converter = Converter()
    converter.add_points(points.geometry)
    converter.add_attribute(
        type_=QMetaType.Double,
        name=elevation_field,
        values=points[elevation_field]
    )
    vector_layer = converter.to_vector_layer(epsg=points.crs.to_epsg())
    extent_param = (
        'TARGET_USER_XMIN TARGET_USER_XMAX TARGET_USER_YMIN TARGET_USER_YMAX'
    )
    return Raster(processing.run('sagang:multilevelbspline', {
        'SHAPES': vector_layer,
        'FIELD': elevation_field,
        extent_param: None,
        'TARGET_USER_SIZE': cell_size,
        'TARGET_USER_FITS': 0,
        'TARGET_OUT_GRID': 'TEMPORARY_OUTPUT',
        'METHOD': 0,
        'EPSILON': EPS,
        'LEVEL_MAX': 14
    })['TARGET_OUT_GRID'])


WeightingMethods = t.Literal['exponential', 'inverse distance']


def inverse_distance_weighted(
    points: gpd.GeoDataFrame,
    cell_size: float,
    power: int = 2,
    elevation_field: str = 'Z',
    weighting_method: WeightingMethods = 'inverse distance'
) -> Raster:
    dw_weighting = {
        'inverse distance': 1,
        'exponential': 2
    }
    converter = Converter()
    converter.add_points(points.geometry)
    converter.add_attribute(
        type_=QMetaType.Double,
        name=elevation_field,
        values=points[elevation_field]
    )
    vector_layer = converter.to_vector_layer(epsg=points.crs.to_epsg())
    extent_param = (
        'TARGET_USER_XMIN TARGET_USER_XMAX TARGET_USER_YMIN TARGET_USER_YMAX'
    )
    return Raster(
        processing.run('sagang:inversedistanceweightedinterpolation', {
            'POINTS': vector_layer,
            'FIELD': elevation_field,
            'CV_METHOD': 0,
            'CV_SUMMARY': 'TEMPORARY_OUTPUT',
            'CV_RESIDUALS': 'TEMPORARY_OUTPUT',
            'CV_SAMPLES': 10,
            extent_param: None,
            'TARGET_USER_SIZE': cell_size,
            'TARGET_USER_FITS': 0,
            'TARGET_OUT_GRID': 'TEMPORARY_OUTPUT',
            'SEARCH_RANGE': 1,
            'SEARCH_RADIUS': 1000,
            'SEARCH_POINTS_ALL': 1,
            'SEARCH_POINTS_MIN': 1,
            'SEARCH_POINTS_MAX': 20,
            'DW_WEIGHTING': dw_weighting[weighting_method],
            'DW_IDW_POWER': power,
            'DW_BANDWIDTH': 1
        })['TARGET_OUT_GRID'])


@dataclass
class Evaluator:
    dem: DEM
    estimation_dem: DEM
    truth_dem: DEM
    gully_cover: DEM
    estimation_surface: Polygon

    def get_masked(self) -> list[DEM]:
        return [
            dem.apply_mask(self.estimation_surface) for dem in
            (self.dem, self.estimation_dem, self.truth_dem, self.gully_cover)
        ]

    def evaluate(self):
        gully_cover_volume = raster_volume(self.gully_cover)
        truth_volume = gully_cover_volume - raster_volume(self.truth_dem)
        estimation_volume = (
            gully_cover_volume
            - raster_volume(self.estimation_dem)
        )
        error = abs(truth_volume - estimation_volume) / truth_volume
        print(
            f'Estimation: {estimation_volume:.3f}\n'
            f'Truth: {truth_volume:.3f}\n'
            f'Error: {error:%}'
        )


def raster_volume(raster: Raster) -> float:
    # Not sure if this breaks for other versions of qgis/SAGA
    feedback = QgsProcessingFeedback()
    processing.run('sagang:rastervolume', {
        'GRID': str(raster),
        'METHOD': 0,
        'LEVEL': 3
    }, feedback=feedback)

    log = feedback.textLog().splitlines()

    def is_result_line(string):
        return 'Volume:' in string

    return float(next(filter(is_result_line, log)).split(': ')[-1])


def raster_difference(a: Raster, b: Raster) -> Raster:
    return Raster(
        processing.run('sagang:rasterdifference', {
            'A': str(a),
            'B': str(b),
            'C': 'TEMPORARY_OUTPUT'
        })['C']
    )


def align_rasters(
    rasters: t.Sequence[Raster],
    reference_raster: Raster
) -> t.Generator[Raster, None, None]:
    extent = str(reference_raster.extent)
    for i, dem in enumerate(rasters, start=1):
        if DEBUG >= 1:
            print(f'Aligned {i} rasters.', end='\r')
        aligned = processing.run('sagang:resampling', {
            'INPUT': [str(dem.path)],
            'OUTPUT': 'TEMPORARY_OUTPUT',
            'KEEP_TYPE': False,
            'SCALE_UP': 3,
            'SCALE_DOWN': 3,
            'TARGET_USER_XMIN TARGET_USER_XMAX TARGET_USER_YMIN TARGET_USER_YMAX': extent,
            'TARGET_USER_SIZE': dem.size_x,
            'TARGET_USER_FITS': 0
        })
        yield Raster(aligned['OUTPUT'], epsg=reference_raster.epsg)
