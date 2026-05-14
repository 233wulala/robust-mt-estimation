import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，只保存文件；若想弹窗改成 "TkAgg"
import matplotlib.pyplot as plt

from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

# load the project
projData = loadProject(projectPath)

from resistics.project.time import viewTime
from resistics.common.plot import plotOptionsTime, getPaperFonts

# plotOptions = plotOptionsTime(plotfonts=getPaperFonts())
# fig = viewTime(
#     projData,
#     "2018-01-03 01:00:00",
#     "2018-01-03 01:15:00",
#     plotoptions=plotOptions,
#     save=False,
# )
# fig.savefig(imagePath / "viewTime_noise_6")
# fig.savefig(imagePath / "viewTime_noise_6.pdf", format="pdf")

# calculate spectrum using standard options
from resistics.project.spectra import calculateSpectra

calculateSpectra(projData, calibrate=False, polreverse={"Hy": True},)
projData.refresh()

from resistics.project.spectra import viewSpectraStack
from resistics.common.plot import plotOptionsSpec

# plotOptions = plotOptionsSpec(plotfonts=getPaperFonts())
# fig = viewSpectraStack(
#     projData,
#     "site1",
#     "meas",
#     coherences=[["Ex", "Hy"], ["Ey", "Hx"]],
#     plotoptions=plotOptions,
#     save=False,
#     show=False,
# )
# fig.savefig(imagePath / "viewSpectraStack")

# process the spectra to estimate the transfer function
from resistics.project.transfunc import processProject

processProject(projData, outchans=["Ex", "Ey"])

from resistics.project.transfunc import getTransferFunctionData

# 计算传输函数
freq1 = getTransferFunctionData(projData, site = "site1", sampleFreq= 0.5)

from resistics.transfunc.io import TransferFunctionWriter

# 目标站点名称
site_name = "site1"

# 检查站点是否存在
if site_name not in projData.sites:
    raise ValueError(f"项目中不存在站点 {site_name}")

# 获取传输函数数据（确保已计算）
tf_data = freq1

# 计算edi
# 设置输出路径
edi_path = "edi_files/noise_mm_70_5_1.edi"

# 创建 TransferFunctionWriter 实例
writer = TransferFunctionWriter(
    filepath=edi_path,
    tfData=tf_data,
    sites="site1",      # 根据实际情况填写
    polarisations=["ExHy", "EyHx"],
)

# 写入 EDI 文件
writer.writeEdi()

print(f"EDI 文件已生成：{edi_path}")

# plot impedance tensor and save the plot
from resistics.project.transfunc import viewImpedance
from resistics.common.plot import plotOptionsTransferFunction

plotoptions = plotOptionsTransferFunction(plotfonts=getPaperFonts())
plotoptions["xlim"] = [1, 100000]  # x轴范围
plotoptions["res_ylim"] = [1, 10000]  # 视电阻率 y 轴范围
plotoptions["phase_ylim"] = [-180, 180]  # 相位 y 轴范围
plotoptions["phase_ticks"] = np.arange(-180, 181, 90)

figs = viewImpedance(
    projData,
    sites=["site1"],
    oneplot=True,  # 把所有极化分量（polarisations）画在同一个图上
    polarisations=["ExHy", "EyHx"],
    plotoptions=plotoptions,
    save=False,
)
# figs[0].savefig(imagePath / "1")
figs[0].savefig(imagePath / "noise_mm_70_5_1")