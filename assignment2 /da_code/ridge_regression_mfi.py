# Scatterplot Matrix
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
import numpy as np

filename = "abalone.csv"
x_axis = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
y_axis = ['Rings']

df = read_csv(filename, skiprows=1, names=x_axis)

#drop outliers, erronous and missing data
df = df.drop(df.index[[1257,3996,3291,1858,3505,3800,2051,335,112]])
#shuffle data
df = df.sample(frac=1).reset_index(drop=True)

df.loc[:, 'Volume'] = df['Length'].values * df['Diameter'].values * df['Height'].values
#print(df)
#mapping = dict(zip(["I", "F", "M"], [-1, 0, 1]))
#df.replace({"Sex": mapping}, inplace=True)

for label in 'MFI':
    df[label] = df['Sex'] == label
del df['Sex']


df = df.dropna()

#x_axis = df.columns.tolist()
x_axis = ['Volume','Length','Diameter','Height','Whole weight','Shucked weight',
          'Viscera weight','Shell weight','M','F','I','Rings']
df = df[x_axis]

#set x and y values
array = df.values
Y = array[:,11]


#X = array[:,0:11]
del df['Rings']
X = df.values.astype(np.float)

print(X)



#print('before')
#print(Y)

#scale data
#scaler = StandardScaler().fit(array)
#X = scaler.transform(array)

#print('after')
#print(X)
#scaler_y = StandardScaler().fit(Y)
#rescaled_Y = scaler.transform(Y)

test_size = 0.10
seed = 7

#kfold
num_folds = 10
kfold = KFold(n_splits=num_folds, random_state=seed)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#print(kfold)

model = Ridge(alpha=0.03,fit_intercept=False)

#model.fit(df[x_axis],df[y_axis])

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)


model = RFE(model, 7)
fit = model.fit(X, Y)



model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)

results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_


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
