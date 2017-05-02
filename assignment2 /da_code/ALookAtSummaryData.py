# View first 20 rows
from pandas import read_csv
from pandas import set_option
filename = "abalone.csv"
names = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
df = read_csv(filename,skiprows=1, names=names)

#print(df.describe())

#df = df.dropna()

#two gone
#print(shape)

#types = df.dtypes
#print(types)

print(df.ix[335])

#print(df.columns)
df = df.drop(df.index[[1257,3996,3291,1858,3505,3800,1417,2051,335]])

print(df.describe())

w_counts = df.groupby('Whole weight').size()
print(w_counts)

Shell_weight_counts = df.groupby('Shell weight').size()
print(Shell_weight_counts)


Shucked_weight_counts = df.groupby('Shucked weight').size()
print(Shucked_weight_counts)


#height_counts = df.groupby('Height').size()
#print(height_counts)

#rings_counts = df.groupby('Rings').size()
#print(rings_counts)

