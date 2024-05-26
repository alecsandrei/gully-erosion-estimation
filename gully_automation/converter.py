"""Used to convert between shapely and QGIS."""

from typing import Sequence

from qgis.core import (
    QgsPoint,
    QgsVectorLayer,
    QgsFeature,
    QgsAbstractGeometry
)
import shapely.geometry


class Converter:

    geoms: list[QgsAbstractGeometry] = []

    def add_points(
        self,
        points: Sequence[shapely.geometry.Point]
    ):
        for point in points:
            if point.has_z:
                coords = (point.x, point.y, point.z)
            else:
                coords = (point.x, point.y)  # type: ignore
            self.geoms.append(QgsPoint(*coords))

    def to_vector_layer(self, epsg: str):
        types = set()
        for geom in self.geoms:
            types.add(geom.geometryType())
        if len(types) == 1:
            geometry_name = types.__iter__().__next__()
        else:
            geometry_name = 'GeometryCollection'
        print(f'{geometry_name}?crs=epsg:{epsg}')
        layer = QgsVectorLayer(
            f'{geometry_name}?crs=epsg:{epsg}', 'converted', 'memory'
        )
        pr = layer.dataProvider()
        layer.startEditing()
        features = []
        for geom in self.geoms:
            feature = QgsFeature()
            feature.setGeometry(geom)
            features.append(feature)
        pr.addFeatures(features)
        layer.commitChanges()
        return layer
