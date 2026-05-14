import numpy as np

from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

# load the project
projData = loadProject(projectPath)

# calculate spectrum using standard options
from resistics.project.spectra import calculateSpectra

calculateSpectra(projData, calibrate=False, polreverse={"Hy": True},)
projData.refresh()