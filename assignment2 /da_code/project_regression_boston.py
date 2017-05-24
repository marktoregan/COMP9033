import numpy
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


filename = "abalone.csv"
names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
y_axis = ['Rings']

dataset = read_csv(filename, skiprows=1, names=names)

#drop outliers, erronous and missing data
dataset = dataset.drop(dataset.index[[1257, 3996, 3291, 1858, 3505, 3800, 2051, 335, 112]])
#shuffle data
dataset = dataset.sample(frac=1).reset_index(drop=True)

dataset.loc[:, 'Volume'] = dataset['Length'].values * dataset['Diameter'].values * dataset['Height'].values

mapping = dict(zip(["I", "F", "M"], [0, 1, 2]))
dataset.replace({"Sex": mapping}, inplace=True)


dataset = dataset.dropna()

#x_axis = df.columns.tolist()
names = ['Sex','Volume', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
          'Viscera weight','Shell weight','Rings']
dataset = dataset[names]

# Split-out validation dataset
array = dataset.values
X = array[:,0:9]
Y = array[:,9]
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Evaluate Algorithms
# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'

num_folds = 10
kfold = KFold(n_splits=num_folds, random_state=seed)


## K neartest neighbours algorithm tuning

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# Tune scaled GBM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=numpy.array([60,80,100,120,140,160,180,200,220]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Make predictions on validation dataset

# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
model.fit(rescaledX, Y_train)
# transform the validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)


print(mean_squared_error(Y_validation, predictions))
