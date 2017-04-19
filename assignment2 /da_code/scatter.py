# Scatterplot Matrix
from matplotlib import pyplot
from pandas import read_csv
from pandas.tools.plotting import scatter_matrix
filename = "abalone.csv"
names = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings','M','F','I']
data = read_csv(filename, names=names)

for label in "MFI":
    data[label] = data["Sex"] == label
del data["Sex"]
names.remove('Sex')

scatter_matrix(data)
pyplot.show()
