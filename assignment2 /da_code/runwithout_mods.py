from math import sqrt
import numpy
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFE


filename = "abalone.csv"
names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
y_axis = ['Rings']

dataset = read_csv(filename, skiprows=1, names=names)

#drop outliers, erronous and missing data
dataset = dataset.drop(dataset.index[[1257, 3996, 3291, 1858, 3505, 3800, 2051, 335, 112]])

#shuffle data
dataset = dataset.sample(frac=1).reset_index(drop=True)

#Map Male, Female and Infant to 0,1,2
mapping = dict(zip(["I", "F", "M"], [0, 1, 2]))
dataset.replace({"Sex": mapping}, inplace=True)

#Add a a new feature called Volume, its length * diameter * height
dataset.loc[:, 'Volume'] = dataset['Length'].values * dataset['Diameter'].values * dataset['Height'].values

#Drop any NA data from datset
dataset = dataset.dropna()


#Set up the names array
names = ['Sex','Volume', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
          'Viscera weight','Shell weight','Rings']
dataset = dataset[names]

array = dataset.values
#x-axis
X = array[:,0:9]
#Y-axis
Y = array[:,9]

# Split-out validation dataset
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

#10 kfold setup
num_folds = 10
kfold = KFold(n_splits=num_folds, random_state=seed)


alphas = numpy.array([0.01,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001,0])
param_grid = dict(alpha=alphas)

model_before_cv = Ridge()

grid = GridSearchCV(model_before_cv, param_grid=param_grid,cv=10)
grid_result = grid.fit(X_train, Y_train)

#print(grid.best_score_)
print("Best esitimator?: %f " % grid.best_estimator_.alpha)
alpha = grid.best_estimator_.alpha
print("Alpha?: %f " % alpha)

# 10 kflod end

#final trining model

model_with_alpha = Ridge(alpha=alpha,fit_intercept=False)

#Create RFE
model_with_alpha = RFE(model_with_alpha, 7)
#train
fit = model_with_alpha.fit(X_train, Y_train)

#predict
y_pred = model_with_alpha.predict(X_validation)

#cross validation hold out
result_validation = model_with_alpha.score(X_validation, Y_validation)

#all data
results = cross_val_score(model_with_alpha, X, Y, cv=kfold)

# Make predictions about the test data

# Print error measurements
print("")
print("Rfe Num Features: %d") % fit.n_features_
print("Rfe Selected Features: %s") % fit.support_
print("Rfe Feature Ranking: %s") % fit.ranking_
print("")
print('MAE: %.2f ' % mean_absolute_error(Y_validation, y_pred))
print('RMSE: %.2f ' % sqrt(mean_squared_error(Y_validation, y_pred)))
print("")
print("Cross Validation Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
print("Model Accuracy: %.3f%%") % (result_validation*100.0)
print("")
print('Coeffients: ')
print(fit.estimator_.coef_)
print("")