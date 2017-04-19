# View first 20 rows
from pandas import read_csv
from pandas import set_option
filename = "abalone.csv"
names = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
data = read_csv(filename, names=names)

#print(data.describe())

for label in "MFI":
    data[label] = data["Sex"] == label
del data["Sex"]


cleaned = data.dropna()
shape = cleaned.shape

#two gone
#print(shape)

types = data.dtypes
#print(types)


set_option('display.width', 100)
set_option('precision', 3)
description = data.describe()
#print(description)


w_counts = data.groupby('Whole weight').size()
print(w_counts)


height_counts = data.groupby('Height').size()
print(height_counts)

rings_counts = data.groupby('Rings').size()
print(rings_counts)





f_counts = data.groupby('F').size()
#print(f_counts)

i_counts = data.groupby('I').size()
#print(i_counts)

total = 1527 + 1307 + 1341
#print(total)


set_option('display.width', 100)
set_option('precision', 3)
correlations = data.corr(method='pearson')
#print(correlations)

skew = cleaned.skew()
#print(skew)

#Notes

#clean rings, exists a -1
#clean  'Whole weight' , minus exists

# SEX has two rows missing delete them later


# Height and rings possibly skewed

# All infant data is in negative values, consider dropping it completely.
