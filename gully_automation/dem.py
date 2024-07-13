"""Used for DEM preprocessing and analysis."""

from __future__ import annotations

from typing import Sequence
from pathlib import Path
from functools import cached_property
import time

from qgis.core import (
    QgsField,
    QgsProject,
    QgsFeature,
    QgsVectorFileWriter,
    QgsCoordinateTransformContext,
    QgsProcessingFeatureSourceDefinition,
)
from qgis.PyQt.QtCore import QVariant
import shapely.geometry

from gully_automation.converter import Converter
from gully_automation import processing


PathLike = Path | str


class DEM:

    def __init__(self, dem: PathLike):
        self.dem = dem
        if not isinstance(self.dem, Path):
            self.dem = Path(self.dem)
            if not self.dem.exists():
                raise FileNotFoundError('DEM file not found.')

    @cached_property
    def dem_preproc(self):
        return self.hydro_preprocessing()

    def hydro_preprocessing(self) -> Path:
        sink_route = processing.run(
            'sagang:sinkdrainageroutedetection', {
                'ELEVATION': self.dem.as_posix(),
                'SINKROUTE': 'TEMPORARY_OUTPUT',
                'THRESHOLD': False,
                'THRSHEIGHT': 100
            }
        )
        dem_preproc = processing.run(
            "sagang:sinkremoval", {
                'DEM': self.dem.as_posix(),
                'SINKROUTE': sink_route['SINKROUTE'],
                'DEM_PREPROC': 'TEMPORARY_OUTPUT',
                'METHOD': 1,
                'THRESHOLD': False,
                'THRSHEIGHT': 100
            }
        )
        return dem_preproc['DEM_PREPROC']

    def profiles(
        self,
        points: Sequence[shapely.Point],
        epsg: str = '3844'
    ) -> list[Path]:
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
                    QgsField('x', QVariant.Double),
                    QgsField('y', QVariant.Double),
                ]
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

        # NOTE: this tool will not create a point
        # on the corresponding starting point
        profiles = processing.run(
            'sagang:leastcostpaths', {
                'SOURCE': layer,
                'DEM': self.dem_preproc,
                'VALUES': None,
                'POINTS': 'TEMPORARY_OUTPUT',
                'LINE': 'TEMPORARY_OUTPUT'
            })

        profiles_shp = list(Path(profiles['POINTS']).parent.glob('*.shp'))
        time.sleep(999)
        if not profiles_shp:
            raise FileNotFoundError(
                'No profiles could be generated. Could be a projection issue.'
            )
        elif len(profiles_shp) == 1:
            return profiles_shp
        else:
            def sort_func(path: Path):
                return int(path.stem.replace('POINTS', ''))
            return sorted(profiles_shp, key=sort_func)


if __name__ == '__main__':

    print(DEM('/media/alex/alex/gullies/saveni_aval_2012_mbs.asc').hydro_preprocessing())
