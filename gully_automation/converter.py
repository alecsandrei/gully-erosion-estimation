"""Used to convert between shapely and QGIS."""

from typing import Sequence
from collections import UserList

from qgis.core import (
    QgsPoint,
    QgsPolygon,
    QgsMultiPolygon,
    QgsVectorLayer,
    QgsLineString,
    QgsMultiLineString,
    QgsFeature,
    QgsAbstractGeometry
)
import shapely.geometry


class InvalidGeometry(Exception):
    def __init__(self, geometry: QgsAbstractGeometry):
        self.geometry = geometry
        super().__init__(f'Invalid input geometry {self.geometry!r}')


class EmptyGeometry(Exception):
    def __init__(self, geometry: QgsAbstractGeometry):
        self.geometry = geometry
        super().__init__(f'Empty input geometry {self.geometry!r}')


class Geoms(UserList[QgsAbstractGeometry]):

    def append(self, geom: QgsAbstractGeometry):
        if not geom.isValid():
            raise InvalidGeometry(geom)
        elif geom.isEmpty():
            raise EmptyGeometry(geom)
        super().append(geom)


class Converter:
    """Used to convert between Shapely and PyQGIS geometry."""

    def __init__(self):
        self.geoms = Geoms()

    def add_points(
        self,
        points: Sequence[shapely.Point]
    ):
        for point in points:
            if point.has_z:
                coords = (point.x, point.y, point.z)
            else:
                coords = (point.x, point.y)  # type: ignore
            self.geoms.append(QgsPoint(*coords))

    def add_lines(
        self,
        lines: Sequence[shapely.LineString | shapely.MultiLineString]
    ):
        for line in lines:
            wkt = line.wkt
            if isinstance(line, shapely.LineString):
                qgs_line = QgsLineString()
                if not qgs_line.fromWkt(wkt):
                    raise Exception(
                        f'Failed to convert {line!r}..'
                    )
                self.geoms.append(qgs_line)
            elif isinstance(line, shapely.MultiLineString):
                qgs_multilinestring = QgsMultiLineString()
                if not qgs_multilinestring.fromWkt(wkt):
                    raise Exception(
                        f'Failed to convert {line!r}.'
                    )
                self.geoms.append(qgs_multilinestring)
            else:
                raise Exception(f'{line!r} not allowed as input')

    def add_polygons(
        self,
        polygons: Sequence[shapely.Polygon | shapely.MultiPolygon]
    ):
        for polygon in polygons:
            wkt = polygon.wkt
            if isinstance(polygon, shapely.Polygon):
                qgs_polygon = QgsPolygon()
                if not qgs_polygon.fromWkt(wkt):
                    raise Exception(
                        f'Failed to convert {polygon!r} to QgsPolygon.'
                    )
                self.geoms.append(qgs_polygon)
            elif isinstance(polygon, shapely.MultiPolygon):
                qgs_multipolygon = QgsMultiPolygon()
                if not qgs_multipolygon.fromWkt(wkt):
                    raise Exception(
                        f'Failed to convert {polygon!r} to QgsMultiPolygon.'
                    )
                self.geoms.append(qgs_multipolygon)
            else:
                raise Exception(f'{polygon!r} not allowed as input')

    def to_vector_layer(self, epsg: str):
        types = set()
        for geom in self.geoms:
            types.add(geom.geometryType())
        if len(types) == 1:
            geometry_name = types.__iter__().__next__()
        else:
            geometry_name = 'GeometryCollection'
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
