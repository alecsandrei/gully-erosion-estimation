"""Used for DEM preprocessing and analysis."""

from __future__ import annotations

from typing import Sequence
from pathlib import Path
from functools import cached_property
import os

from qgis.core import (
    QgsField,
    QgsProject,
    QgsFeature,
    QgsProcessingFeatureSourceDefinition
)
import shapely.geometry

from gully_automation.converter import Converter
from gully_automation import processing


PathLike = Path | str


class DEM:

    def __init__(self, dem: PathLike):
        self.dem = os.fspath(dem)

    @cached_property
    def dem_preproc(self):
        return self.hydro_preprocessing()

    def hydro_preprocessing(self) -> Path:
        sink_route = processing.run(
            'sagang:sinkdrainageroutedetection', {
                'ELEVATION': self.dem,
                'SINKROUTE': 'TEMPORARY_OUTPUT',
                'THRESHOLD': False,
                'THRSHEIGHT': 100
            }
        )
        dem_preproc = processing.run(
            "sagang:sinkremoval", {
                'DEM': self.dem,
                'SINKROUTE': sink_route['SINKROUTE'],
                'DEM_PREPROC': 'TEMPORARY_OUTPUT',
                'METHOD': 1,
                'THRESHOLD': False,
                'THRSHEIGHT': 100
            }
        )
        return dem_preproc['DEM_PREPROC']

    def profiles(self, points: Sequence[shapely.Point]):
        converter = Converter()
        converter.add_points(points)
        layer = converter.to_vector_layer('4326')

        def add_x_y():
            provider = layer.dataProvider()
            layer.startEditing()
            provider.addAttributes(
                [
                    QgsField('x', typeName='double'),
                    QgsField('y', typeName='double'),
                ]
            )
            layer.updateFields()
            rows = []
            for i in range(1, layer.featureCount() + 1):
                feature = layer.getFeature(i)
                geom = feature.geometry().asPoint()
                feature.setAttributes(
                    [geom.x(), geom.y()]
                )
            provider.addFeatures(rows)
            layer.commitChanges()

        QgsProject.instance().addMapLayer(layer, False)
        profiles = processing.run("sagang:profilefrompoints", {
            'GRID': self.dem_preproc,
            'VALUES': None,
            'TABLE': QgsProcessingFeatureSourceDefinition(layer.id()),
            'X': 'x',
            'Y': 'y',
            'RESULT':'TEMPORARY_OUTPUT'
            }
        )
        print(profiles)


if __name__ == '__main__':

    print(DEM('/media/alex/alex/gullies/saveni_aval_2012_mbs.asc').hydro_preprocessing())