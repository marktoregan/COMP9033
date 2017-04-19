# View first 20 rows
import pandas as pd
from pandas import set_option
filename = "abalone.csv"
names = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
df = pd.read_csv(filename, names=names) #, index_col=['Sex'])
df.columns.name = 'Id'

w_counts = df.groupby('Whole weight').size()

#print(df[pd.isnull(df).any(axis=1)])

rings_counts = df.groupby('Rings').size()
#print(rings_counts)

#print(df[df.Height > .5])

print(df[df.Rings > 20])
#if df['Height'] > 5:
 #   print('how high')


# 1257,3996 erronous
# 3291,1858,3505,3800 missis NaN data
df = df.drop(df.index[[1257,3996,3291,1858,3505,3800]])



df = df.dropna()
shape = df.shape
print(shape)
