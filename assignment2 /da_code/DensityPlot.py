from matplotlib import pyplot
from pandas import read_csv

from pandas import set_option
filename = "abalone.csv"
names = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
data = read_csv(filename, names=names)

for label in "MFI":
    data[label] = data["Sex"] == label
del data["Sex"]


data.plot(kind='density', subplots=True, layout=(5, 3), sharex=False)
pyplot.show()