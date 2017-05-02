# Ridge Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(filename, delim_whitespace=True, skiprows=1, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
print(X)
print(Y)
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = Ridge(alpha=0.3)
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print(results.mean())

