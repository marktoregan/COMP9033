from matplotlib import pyplot
from pandas import read_csv
import seaborn as sns
import numpy as np

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

#rows, columns, position
pyplot.subplot(3,3,1)
pyplot.hist(df['Diameter'],bins=7, color='crimson', alpha=0.7)
pyplot.ylabel('Frequency')
pyplot.xlabel('Lenght in cm')
pyplot.title('Diameter')
pyplot.subplots_adjust(hspace=.5)

pyplot.subplot(3,3,2)
pyplot.hist(df['Height'],bins=20, color='blue', alpha=0.7)
pyplot.ylabel('Frequency')
pyplot.xlabel('Height in cm')
pyplot.title('Height')

pyplot.subplot(3,3,3)
pyplot.hist(df['Whole weight'],bins=20,color='crimson', alpha=0.7)
pyplot.ylabel('Frequency')
pyplot.xlabel('Weight in gm')
pyplot.title('Whole weight')

pyplot.subplot(3,3,4)
pyplot.hist(df['Shucked weight'],bins=20, color='blue', alpha=0.7)
pyplot.ylabel('Frequency')
pyplot.xlabel('Weight in gm')
pyplot.title('Shucked weight')

pyplot.subplot(3,3,5)
pyplot.hist(df['Viscera weight'],bins=20, color='crimson', alpha=0.7)
pyplot.ylabel('Frequency')
pyplot.xlabel('Weight in gm')
pyplot.title('Viscera weight')

pyplot.subplot(3,3,6)
pyplot.hist(df['Shell weight'],bins=20,color='blue', alpha=0.7)
pyplot.ylabel('Frequency')
pyplot.xlabel('Weight in gm')
pyplot.title('Shell weight')

pyplot.subplot(3,3,7)
pyplot.hist(df['Rings'],bins=20, color='crimson', alpha=0.7)
pyplot.ylabel('Frequency')
pyplot.xlabel('Number of rings')
pyplot.title('Rings')


df.plot(kind='box', subplots=True, layout=(5,3), sharex=False, sharey=False)

pyplot.subplot(3,3,8)
pyplot.boxplot(np.log(df['Rings']))

pyplot.show()

#pyplot.subplot(3,3,8)
#pyplot.hist(df['Sex'] ,bins=30, color='blue')
#pyplot.title('Sex')





#pyplot.hist([df['Length']],bins=20)
#pyplot.hist([df['Diameter']],bins=20)
#plt.hist([x, y], color=['r','b'], alpha=0.5)



# Create bins of 2000 each
 # fixed bin size

# Plot a histogram of attacker size
#plt.hist(data1,
 #        bins=bins,
  #       alpha=0.5,
   #      color='#EDD834',
    #     label='Attacker')

#his.hist(his, bins=np.arange(min(his), max(his) + binwidth, binwidth))
#versicolor.plot(kind='hist', subplots=True, layout=(2,2), figsize=(12,6))

#pyplot.title('title')
#pyplot.legend()


#pyplot.figure()


#df.diff().hist(color='k', alpha=0.5, bins=50)


# diameter, length, weight, rings, whole weight, "possibly height"
#Gaussian distribution

