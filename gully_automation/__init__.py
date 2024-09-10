import sys
import os
from qgis.core import *
import osgeo.ogr


osgeo.ogr.UseExceptions()


# Initialize QGIS Application
# Supply path to qgis install location
QgsApplication.setPrefixPath("/home/alex/miniforge3/envs/qgis", True)
# Create a reference to the QgsApplication.  Setting the
# second argument to False disables the GUI.
qgs = QgsApplication([], False)

# Load providers
qgs.initQgis()

# Add the path to Processing framework
sys.path.append('/home/alex/.local/share/QGIS/QGIS3/profiles/default/python/plugins')
# importing the Saga NextGen Provider
try:
    from processing_saga_nextgen.saga_nextgen_plugin import SagaNextGenAlgorithmProvider
except ModuleNotFoundError:
    print('Install the SAGA Next Gen plugin.')
    exit(1)
# creating the Saga NextGen Provider
provider = SagaNextGenAlgorithmProvider()
# loading all algorithms belonging to the Saga NextGen Provider
provider.loadAlgorithms()
# adding the Saga NextGen processing Provider to the registry.
QgsApplication.processingRegistry().addProvider(provider=provider)

from qgis.analysis import QgsNativeAlgorithms
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

import processing
from processing.core.Processing import Processing

Processing.initialize()

EPS = os.getenv('EPSILON', 1e-5)
DEBUG = int(os.getenv('DEBUG', 0))
CACHE = int(os.getenv('CACHE', 0))
MODEL = os.getenv('MODEL', None)

