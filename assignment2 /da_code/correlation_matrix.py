from matplotlib import pyplot
from pandas import read_csv
import numpy

filename = "abalone.csv"
names = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings','M','F','I']
data = read_csv(filename, names=names)

for label in "MFI":
    data[label] = data["Sex"] == label
del data["Sex"]
names.remove('Sex')

correlations = data.corr()
# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,11,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
#ax.set_xticklabels(names)
#ax.set_yticklabels(names)
pyplot.show()