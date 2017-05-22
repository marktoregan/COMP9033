from matplotlib import pyplot
from pandas import read_csv
import seaborn as sns

filename = "abalone.csv"
names = ["Sex","Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Rings"]
df = read_csv(filename, names=names)

print(df.columns)
df = df.drop(df.index[[1257,3996,3291,1858,3505,3800,2051,335,112]])

for label in "MFI":
    df[label] = df["Sex"] == label
del df["Sex"]

df = df.dropna()

sns.set()

#rows, columns, position

pyplot.subplot(3,3,1)
pyplot.hist(df['Length'],bins=20, color='blue', alpha=0.7)
pyplot.ylabel('Frequency')
pyplot.xlabel('Length in mm')
pyplot.title('Length')


pyplot.subplot(3,3,2)
pyplot.hist(df['Diameter'],bins=7, color='crimson', alpha=0.7)
pyplot.ylabel('Frequency')
pyplot.xlabel('Diameter in mm')
pyplot.title('Diameter')
pyplot.subplots_adjust(hspace=.5)

pyplot.subplot(3,3,3)
pyplot.hist(df['Height'],bins=20, color='blue', alpha=0.7)
pyplot.ylabel('Frequency')
pyplot.xlabel('Height in mm')
pyplot.title('Height')

pyplot.subplot(3,3,4)
pyplot.hist(df['Whole weight'],bins=20,color='crimson', alpha=0.7)
pyplot.ylabel('Frequency')
pyplot.xlabel('Whole weight in grams')
pyplot.title('Whole weight')

pyplot.subplot(3,3,5)
pyplot.hist(df['Shucked weight'],bins=20, color='blue', alpha=0.7)
pyplot.ylabel('Frequency')
pyplot.xlabel('Shucked weight in grams')
pyplot.title('Shucked weight')

pyplot.subplot(3,3,6)
pyplot.hist(df['Viscera weight'],bins=20, color='crimson', alpha=0.7)
pyplot.ylabel('Frequency')
pyplot.xlabel('Viscera weight in grams')
pyplot.title('Viscera weight')

pyplot.subplot(3,3,7)
pyplot.hist(df['Shell weight'],bins=20,color='blue', alpha=0.7)
pyplot.ylabel('Frequency')
pyplot.xlabel('Shell Weight in grams')
pyplot.title('Shell weight')

pyplot.subplot(3,3,8)
pyplot.hist(df['Rings'],bins=20, color='crimson', alpha=0.7)
pyplot.ylabel('Frequency')
pyplot.xlabel('Number of rings')
pyplot.title('Rings')


df.plot(kind='box', subplots=True, layout=(5,3), sharex=False, sharey=False)

pyplot.subplot(3,3,8)
#pyplot.boxplot(np.log(df['Rings']))

vars=["Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Rings"]
g = sns.pairplot(df, vars=vars, hue="Sex",size=2)
g.fig.get_children()[-1].set_bbox_to_anchor((1.1, 0.5, 0, 0))

#df = sns.load_dataset('iris')
#sns_plot = sns.pairplot(df, hue='species', size=2.5)
g.savefig("output2.png")

#sns.pairplot(df,hue='Sex')

#g = sns.pairplot(df)

# Additional line to adjust some appearance issue
#pyplot.subplots_adjust(top=0.9)
#pyplot.subplots_adjust(bottom=0.8)

# Set the Title of the graph from here
g.fig.suptitle('Total Bill vs. Tip', fontsize=34,color="r",alpha=0.5)

#g.map_diag(sns.kdeplot)
#g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6);

#g.fig.set_size_inches(12,8)

# Set the xlabel of the graph from here
#g.set_xlabels("Tip",size=50,color="r",alpha=0.5)

# Set the ylabel of the graph from here
#g.set_ylabels("Total Bill",size=50,color="r",alpha=0.5)


pyplot.show()
