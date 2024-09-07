"""Used to convert between shapely and QGIS."""

from __future__ import annotations

from typing import Sequence, TypedDict
from collections import UserList
import itertools

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
from qgis.core import QgsField
from qgis.PyQt.QtCore import QMetaType
import shapely.geometry

from gully_automation import DEBUG

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


class Attribute(TypedDict):
    name: str
    type_: QMetaType
    values: Sequence


class Converter:
    """Used to convert geometries between Shapely and PyQGIS."""

    def __init__(self):
        self.geoms = Geoms()
        self.attributes: list[Attribute] = []

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
        lines: (
            Sequence[shapely.LineString | shapely.MultiLineString]
        )
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

    def add_attribute(
        self, type_: QMetaType, name: str, values: Sequence
    ):
        self.attributes.append(
            {'name': name,
             'type_': type_,
             'values': values}
        )

    def to_vector_layer(self, epsg: str):
        for attribute in self.attributes:
            if (
                (len1 := len(attribute['values'])) and len1 != (len2 := len(self.geoms))
            ):
                raise Exception(
                    f'Shape mismatch between attributes and geometries: {len1} {len2}.'
                )
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
        if self.attributes:
            for attribute in self.attributes:
                if DEBUG >= 1:
                    print(f'Adding field {attribute["name"]} to the layer.')
                pr.addAttributes(
                    [
                        QgsField(attribute['name'], attribute['type_']),
                        QgsField(attribute['name'], attribute['type_'])
                    ]
                )
            if DEBUG >= 1:
                print(f'UPDATING FIELDS FOR {type(layer)}.')
            layer.updateFields()
        layer.startEditing()
        features = []
        attribute_values = zip(
            *[attribute['values'] for attribute in self.attributes]
        )
        for geom, values in itertools.zip_longest(
            self.geoms,
            attribute_values,
            fillvalue=None
        ):
            feature = QgsFeature()
            feature.setGeometry(geom)
            if self.attributes:
                feature.setAttributes(list(values))
            features.append(feature)
        pr.addFeatures(features)
        layer.commitChanges()
        return layer
