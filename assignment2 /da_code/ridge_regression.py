# Scatterplot Matrix
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from matplotlib import pyplot
import seaborn as sns


filename = "abalone.csv"
x_axis = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
y_axis = ['Rings']

df = read_csv(filename, skiprows=1, names=x_axis)

#drop outliers, erronous and missing data
df = df.drop(df.index[[1257,3996,3291,1858,3505,3800,2051,335,112]])
#shuffle data
df = df.sample(frac=1).reset_index(drop=True)

mapping = dict(zip(["I", "F", "M"], [-1, 0, 1]))
df.replace({"Sex": mapping}, inplace=True)
#double check there is no NA data and drop if exists
df = df.dropna()


array = df.values
#X = array[:,0:8] or X = array[:,0:7] ??? not sure yet!!
X = array[:,0:8]
Y = array[:,8]

print(x_axis)
print(X)

#print('next')
print(Y)

num_folds = 10

kfold = KFold(n_splits=10, random_state=7)
#KFold(n_splits=n)
model = Ridge(alpha=0.3,fit_intercept=False)

model.fit(df[x_axis],df[y_axis])

print('coeffient here -------------------->')
print(model.coef_)

scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())

sns.distplot(model.coef_, label='Ridge Model')
pyplot.title('Residual Distribution Plot')
pyplot.legend()
pyplot.show()

#rreg = linear_model.Ridge( alpha=0.1, fit_intercept=False )






# Plot regressors
#r = RidgeRegressor()
#r.fit(X, Y)
#pyplot.plot(X, r.predict(X), 'b', label=u' (alpha=0.0)')
#alpha = 3.0
#r.fit(X, Y, alpha)
#pyplot.plot(X, r.predict(X), 'y', label=u' (alpha=%.1f)' % alpha)

#pyplot.legend()
#pyplot.show()


