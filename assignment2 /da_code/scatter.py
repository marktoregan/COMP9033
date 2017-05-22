# Scatterplot Matrix
from matplotlib import pyplot
from pandas import read_csv
from pandas.tools.plotting import scatter_matrix
filename = "abalone.csv"
names = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
df = read_csv(filename, names=names)

df = df.drop(df.index[[1257,3996,3291,1858,3505,3800,2051,335,112]])

#for label in "MFI":
 #   df[label] = df["Sex"] == label
del df["Sex"]

df = df.dropna()


scatter_matrix(df)
pyplot.show()
