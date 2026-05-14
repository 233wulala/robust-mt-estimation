from datapaths import projectPath, imagePath
from resistics.project.io import loadProject
from resistics.statistics.utils import getStatNames
from resistics.project.statistics import calculateStatistics, getStatisticData

projData = loadProject(projectPath)
stats, remotestats = getStatNames()

# ⭐ 传入 zOutputFile，Z 会在计算过程中实时写入文件
calculateStatistics(projData, stats=stats, zOutputFile="Z_impedance_30_30.csv")

statData = getStatisticData(
    projData, "site1", "meas", "transferFunction", declevel=1
)

fig = statData.crossplot(
    0,
    crossplots=[
        ["ExHxReal", "ExHxImag"],
        ["ExHyReal", "ExHyImag"],
        ["EyHxReal", "EyHxImag"],
        ["EyHyReal", "EyHyImag"],
    ],
    xlim=[-7.5, 7.5],
    ylim=[-7.5, 7.5],
)
fig.savefig(imagePath / "usingStats_statistic_transferfunction_crossplot_3")