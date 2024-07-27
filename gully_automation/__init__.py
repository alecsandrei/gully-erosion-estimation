import sys
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
from processing_saga_nextgen.saga_nextgen_plugin import SagaNextGenAlgorithmProvider
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

EPS = 1e-10
