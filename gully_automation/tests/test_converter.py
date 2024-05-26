import shapely.geometry

from gully_automation.converter import Converter


def test_add_point():
    points = [
        shapely.Point(0, 0),
        shapely.Point(1, 0),
        shapely.Point(2, 0)
    ]
    converter = Converter()
    converter.add_points(points)
    layer = converter.to_vector_layer()
    assert layer.featureCount() == 3
