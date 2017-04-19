from matplotlib import pyplot
from pandas import read_csv
import seaborn as sns

filename = "abalone.csv"
names = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
df = read_csv(filename, names=names)

print(df.columns)
df = df.drop(df.index[[1257,3996,3291,1858,3505,3800,1417,2051]])

mapping = dict(zip(["I", "F", "M"], [1, 2, 3]))
df.replace({"Sex": mapping}, inplace=True)

print(df['Sex'])

df = df.dropna()

sns.set()


df.plot(kind='box', subplots=True, layout=(5,3), sharex=False, sharey=False)
pyplot.show()