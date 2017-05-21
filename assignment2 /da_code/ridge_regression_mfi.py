# Scatterplot Matrix

from math import sqrt
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

#
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns


filename = "abalone.csv"
names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
y_axis = ['Rings']

dataset = read_csv(filename, skiprows=1, names=names)

#drop outliers, erronous and missing data
dataset = dataset.drop(dataset.index[[1257, 3996, 3291, 1858, 3505, 3800, 2051, 335, 112]])
#shuffle data
dataset = dataset.sample(frac=1).reset_index(drop=True)

print(dataset.head(n=5))

# Descriptive statistics
# shape
print(dataset.shape)
# types
print(dataset.dtypes)
# head
print(dataset.head(20))
# descriptions, change precision to 2 places
set_option('precision', 3)
print(dataset.describe())
# correlation
set_option('precision', 3)
print(dataset.corr(method='pearson'))

#start comment out section
## Data visualizations
#
## histograms
#dataset.hist()
#pyplot.show()
#
## density
#dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False)
#pyplot.show()

## box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
#pyplot.show()
#
## scatter plot matrix
#scatter_matrix(dataset)
#pyplot.show()
#
## correlation matrix
#fig = pyplot.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
#fig.colorbar(cax)
#ticks = arange(0,14,1)
#ax.set_xticks(ticks)
#ax.set_yticks(ticks)
#ax.set_xticklabels(names)
#ax.set_yticklabels(names)
#pyplot.show()
#end comment out section

dataset.loc[:, 'Volume'] = dataset['Length'].values * dataset['Diameter'].values * dataset['Height'].values

mapping = dict(zip(["I", "F", "M"], [0, 1, 2]))
dataset.replace({"Sex": mapping}, inplace=True)

#for label in 'MFI':
#    dataset[label] = dataset['Sex'] == label
#del dataset['Sex']


dataset = dataset.dropna()

#x_axis = df.columns.tolist()
names = ['Sex','Volume', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
          'Viscera weight','Shell weight','Rings']
dataset = dataset[names]

# Split-out validation dataset
array = dataset.values
X = array[:,0:9]
Y = array[:,9]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

#X = array[:,0:11]
#del dataset['Rings']

# Evaluate Algorithms
# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'

num_folds = 10
kfold = KFold(n_splits=num_folds, random_state=seed)

# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
models.append(('Ridge', Ridge()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

## Compare Algorithms
#fig = pyplot.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#pyplot.boxplot(results)
#ax.set_xticklabels(names)
#pyplot.show()

# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))

pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
pipelines.append(('ScaledRidge', Pipeline([('Scaler', StandardScaler()),('Ridge', Ridge())])))

results = []
names = []
for name, model in pipelines:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

## Compare Algorithms
#fig = pyplot.figure()
#fig.suptitle('Scaled Algorithm Comparison')
#ax = fig.add_subplot(111)
#pyplot.boxplot(results)
#ax.set_xticklabels(names)
#pyplot.show()

###very import section

## KNN Algorithm tuning
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
#k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
#param_grid = dict(n_neighbors=k_values)
#model = KNeighborsRegressor()
#kfold = KFold(n_splits=num_folds, random_state=seed)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
#grid_result = grid.fit(rescaledX, Y_train)

#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#	print("%f (%f) with: %r" % (mean, stdev, param))

###############
#Calculate the alpha value aka paramter value

#alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
#alphas = numpy.array([3,2,1,0.5,0.3,0.2,0.1,0.05,0.04,0.03,0.02,0.01,0.003,0.002,0.001,0])
#alphas = numpy.array([0.01,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001,0])

alphas = numpy.array([0.01,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001,0])
param_grid = dict(alpha=alphas)

#pip = Pipeline([("pip",Ridge())])
model = Ridge()

#model_before_cv = Ridge()

grid = GridSearchCV(model, param_grid=param_grid,cv=10)
grid_result = grid.fit(X_train, Y_train)
#test


#test end

#grid.fit(rescaledX, Y)

print(grid.best_score_)
print("Best esitimator?: %f " % grid.best_estimator_.alpha)
alpha = grid.best_estimator_.alpha
print("Alpha?: %f " % alpha)


##################

#my stufff
#
#
#
#print(kfold)

print(alpha)
model_with_alpha = Ridge(alpha=alpha,fit_intercept=False)

model_with_alpha = RFE(model_with_alpha, 7)
fit = model_with_alpha.fit(X_train, Y_train)

result_validation = model_with_alpha.score(X_validation, Y_validation)

results = cross_val_score(model_with_alpha, X, Y, cv=kfold)

# Make predictions about the test data
y_pred = model_with_alpha.predict(X_validation)

# Print error measurements
print("")
print('MAE: %.2f Gb/s' % mean_absolute_error(Y_validation, y_pred))
print('RMSE: %.2f Gb/s' % sqrt(mean_squared_error(Y_validation, y_pred)))
print("")
print("Cross Validation Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
print("Model Accuracy: %.3f%%") % (result_validation*100.0)
print("")
print("Rfe Num Features: %d") % fit.n_features_
print("Rfe Selected Features: %s") % fit.support_
print("Rfe Feature Ranking: %s") % fit.ranking_
print("")
print('Coeffients: ')
print(fit.estimator_.coef_)
print("")

