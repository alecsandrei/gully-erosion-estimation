"""Used for DEM preprocessing and analysis."""

from __future__ import annotations

from typing import Sequence
from dataclasses import dataclass
from pathlib import Path
from functools import cached_property
import time

import geopandas as gpd
from qgis.core import (
    QgsField,
    QgsRasterLayer,
    QgsCoordinateReferenceSystem,
)
from qgis.PyQt.QtCore import QMetaType
import shapely.geometry

from gully_automation.geometry import is_orphaned
from gully_automation.converter import Converter
from gully_automation import processing, DEBUG
from gully_automation.utils import vector_layers_to_geodataframe


PathLike = Path | str


class DEM:

    def __init__(
        self, path: PathLike,
        mask: shapely.Polygon | shapely.MultiPolygon | None = None,
        epsg: str | None = None
    ):
        self.path = path
        self.mask = mask
        self.epsg = epsg
        if not isinstance(self.path, Path):
            self.path = Path(self.path)
            if not self.path.exists():
                raise FileNotFoundError('DEM file not found.')

    @cached_property
    def dem_preproc(self) -> Path:
        return self.hydro_preprocessing()

    @cached_property
    def dem_masked(self) -> Path:
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

    def hydro_preprocessing(self) -> Path:
        sink_route = processing.run(
            'sagang:sinkdrainageroutedetection', {
                'ELEVATION': self.path.as_posix(),
                'SINKROUTE': 'TEMPORARY_OUTPUT',
                'THRESHOLD': False,
                'THRSHEIGHT': 100
            }
        )
        dem_preproc = processing.run(
            'sagang:sinkremoval', {
                'DEM': self.path.as_posix(),
                'SINKROUTE': sink_route['SINKROUTE'],
                'DEM_PREPROC': 'TEMPORARY_OUTPUT',
                'METHOD': 1,
                'THRESHOLD': False,
                'THRSHEIGHT': 100
            }
        )
        return Path(dem_preproc['DEM_PREPROC'])

    def apply_mask(
        self,
        mask: shapely.Polygon | shapely.MultiPolygon | None = None
    ) -> Path:
        if mask is None:
            mask = self.mask
        converter = Converter()
        assert mask is not None
        converter.add_polygons([mask])
        assert self.epsg is not None
        mask_as_qgs_vector_layer = converter.to_vector_layer(self.epsg)
        # masked_dem = processing.run(
        #     'gdal:cliprasterbymasklayer', {
        #         'INPUT': self.path.as_posix(),
        #         'MASK': mask_as_qgs_vector_layer,
        #         'SOURCE_CRS': QgsCoordinateReferenceSystem(f'EPSG:{self.epsg}'),
        #         'TARGET_CRS': QgsCoordinateReferenceSystem(f'EPSG:{self.epsg}'),
        #         'TARGET_EXTENT': None,
        #         'NODATA': None,
        #         'ALPHA_BAND': False,
        #         'CROP_TO_CUTLINE': False,
        #         'KEEP_RESOLUTION': False,
        #         'SET_RESOLUTION': False,
        #         'X_RESOLUTION': None,
        #         'Y_RESOLUTION': None,
        #         'MULTITHREADING': False,
        #         'OPTIONS': '',
        #         'DATA_TYPE': 0,
        #         'EXTRA': '',
        #         'OUTPUT': 'TEMPORARY_OUTPUT'}
        # )
        masked_dem = processing.run('sagang:cliprasterwithpolygon', {
            'INPUT': [self.path.as_posix()],
            'OUTPUT': 'TEMPORARY_OUTPUT',
            'POLYGONS': mask_as_qgs_vector_layer,
            'EXTENT': 1
        })
        return Path(masked_dem['OUTPUT'])

    def sample(
        self,
        shapes: list[shapely.LineString],
        epsg: str
    ) -> gpd.GeoDataFrame:
        converter = Converter()
        converter.add_lines(shapes)

        profiles = processing.run('sagang:profilesfromlines', {
            'DEM': self.dem_preproc.as_posix(),
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
        points: Sequence[shapely.Point],
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
                print(f'UPDATING FIELDS FOR {type(layer)}.')
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
                'INPUT_RASTER': self.dem_preproc.as_posix(),
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
            profiles = processing.run(
                'sagang:leastcostpaths', {
                    'SOURCE': snap_pour_point_to_closest_grid_cell(),
                    'DEM': self.dem_preproc.as_posix(),
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
        # (user defined the wrong boundary).
        orphaned = profiles_geoms.apply(lambda profile: is_orphaned(
            profile, profiles_geoms
        ))
        profiles_orphaned = profiles_geoms[orphaned]
        print(
            f'Found {profiles_orphaned.shape[0]} profiles which '
            'would fall outside of the defined boundary'
        )
        profiles_valid: gpd.GeoSeries = profiles_geoms[
            ~orphaned
        ]  # type: ignore
        if out_file:
            profiles_valid.to_file(out_file)
        return profiles_valid


def multilevel_b_spline(
    points: gpd.GeoDataFrame,
    dem: DEM,
    elevation_field: str = 'Z'
) -> Path:
    converter = Converter()
    converter.add_points(points.geometry)
    converter.add_attribute(
        type_=QMetaType.Double,
        name='Z',
        values=points['Z']
    )
    vector_layer = converter.to_vector_layer(epsg=points.crs.to_epsg())
    return Path(processing.run('sagang:multilevelbspline', {
        'SHAPES': vector_layer,
        'FIELD': elevation_field,
        'TARGET_USER_XMIN TARGET_USER_XMAX TARGET_USER_YMIN TARGET_USER_YMAX': None,
        'TARGET_USER_SIZE': dem.size_x,
        'TARGET_USER_FITS': 0,
        'TARGET_OUT_GRID': 'TEMPORARY_OUTPUT',
        'METHOD': 0,
        'EPSILON': 0.0001,
        'LEVEL_MAX': 14
    })['TARGET_OUT_GRID'])


def inverse_distance_weighted(points: gpd.GeoDataFrame, power: int = 1, elevation_field: str = 'Z'):
    if 'Z' not in points:
        raise NotImplementedError(
            'Input geodataframe should have a Z column.'
        )
    converter = Converter()
    converter.add_points(points.geometry)
    converter.add_attribute(
        type_=QMetaType.Double,
        name='Z',
        values=points['Z']
    )
    vector_layer = converter.to_vector_layer(epsg=points.crs.to_epsg())
    return Path(processing.run('sagang:inversedistanceweightedinterpolation', {
        'POINTS': vector_layer,
        'FIELD': 'Z',
        'CV_METHOD': 0,
        'CV_SUMMARY': 'TEMPORARY_OUTPUT',
        'CV_RESIDUALS': 'TEMPORARY_OUTPUT',
        'CV_SAMPLES': 10,
        'TARGET_USER_XMIN TARGET_USER_XMAX TARGET_USER_YMIN TARGET_USER_YMAX': None,
        'TARGET_USER_SIZE': 0.5,
        'TARGET_USER_FITS': 0,
        'TARGET_OUT_GRID': 'TEMPORARY_OUTPUT',
        'SEARCH_RANGE': 1,
        'SEARCH_RADIUS': 1000,
        'SEARCH_POINTS_ALL': 1,
        'SEARCH_POINTS_MIN': 1,
        'SEARCH_POINTS_MAX': 20,
        'DW_WEIGHTING': 1,
        'DW_IDW_POWER': power,
        'DW_BANDWIDTH': 1
    })['TARGET_OUT_GRID'])


@dataclass
class Evaluate:
    dem: DEM
    estimation_dem: DEM
    truth_dem: DEM
    gully_cover: DEM
    estimation_surface: shapely.Polygon

    def get_masked(self):
        return [dem.apply_mask(self.estimation_surface) for dem in
                (self.dem, self.estimation_dem, self.truth_dem, self.gully_cover)]


if __name__ == '__main__':

    print(DEM('/media/alex/alex/gullies/saveni_aval_2012_mbs.asc').hydro_preprocessing())
