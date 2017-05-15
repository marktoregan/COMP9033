# Scatterplot Matrix
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
df = df.dropna()


array = df.values
X = array[:,0:8]
Y = array[:,8]

print('before')
print(X)

#scale data
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

print('after')
print(X)
#scaler_y = StandardScaler().fit(Y)
#rescaled_Y = scaler.transform(Y)

test_size = 0.33
seed = 7

#kfold
num_folds = 10
kfold = KFold(n_splits=num_folds, random_state=seed)



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


model = Ridge(alpha=0.3,fit_intercept=False)

#model.fit(df[x_axis],df[y_axis])
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print("Accuracy: %.3f%%") % (result*100.0)


#print('coeffient here -------------------->')
#print(model.coef_)

scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print(results.mean())
#print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())





#sns.distplot(model.coef_, label='Ridge Model')
#pyplot.title('Residual Distribution Plot')
#pyplot.legend()
# pyplot.show()
