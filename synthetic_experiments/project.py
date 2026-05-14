from datapaths import projectPath # import the project path
from resistics.project.io import newProject

# define reference time for project
referenceTime = "2018-01-01 00:00:00"  # 创建项目的参考时间
# create a new project and print infomation
# newProject()方法根据需要创建新文件夹，并返回一个ProjectData包含项目信息的对象
projData = newProject(projectPath, referenceTime)
# printInfo()方法可以查看项目信息
projData.printInfo()
# create a new site
projData.createSite("site1")
projData.printInfo()