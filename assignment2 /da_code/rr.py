####################################################################
# Regression Machine Learning with Python                          #
# Ridge Regression                                                 #
# (c) Diego Fernandez Garcia 2016                                  #
# www.exfinsis.com                                                 #
####################################################################

# 1. Packages and Data Import

# 1.1. Packages import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as ml
import sklearn.grid_search as cv
# grid_search for scikit-learn 0.17 model_selection for scikit-learn 0.18

# 1.2. CSV data file import
# Dates must be in format yyyy-mm-dd
data = pd.read_csv('C:\\Users\\Diego\\Desktop\\Regression Machine Learning with Python\\SP500 Returns.txt',
                   index_col='Date', parse_dates=True)

# Imported data chart
data.plot(y=['Yt'])
plt.title('Annual Rolling SP500 Returns 1990-01-31 to 2015-12-31')
plt.legend(loc='upper left')
plt.axvspan('2011-01-31', '2015-12-31', color='green', alpha=0.25)
plt.show()

# 1.3. Delimit training and forecasting ranges
tdata = data['1990-01-31':'2010-12-31']
fdata = data['2011-01-31':'2015-12-31']

# 1.4. Create training and forecasting series

# Create y, y1 and x1 training series
# Lists must be converted to 1-D arrays for scikit-learn package calculation
yt = np.reshape(tdata['Yt'], (len(tdata), 1))
yt1 = np.reshape(tdata['Yt.1'], (len(tdata), 1))
xt1 = np.reshape(tdata['Xt.1'], (len(tdata), 1))

# Create y, y1 and x1 forecasting series
# Lists must be converted to 1-D arrays for scikit-learn package calculation
yf = np.reshape(fdata['Yt'], (len(fdata), 1))
yf1 = np.reshape(fdata['Yt.1'], (len(fdata), 1))
xf1 = np.reshape(fdata['Xt.1'], (len(fdata), 1))

# 2. Ridge Regression fitting and forecasting

# 2.1. Ridge Regression fitting on training range optimal parameters selection through 10-fold cross-validation
# Cross-validation exhaustive grid search with parameter array specification
# L2 penalization is less than L1 (Lasso Regression) therefore alphas should be larger
RIDGEcvgrida = cv.GridSearchCV(ml.Ridge(solver='auto'), cv=10, param_grid={"alpha": [0.1, 1.0, 10.0]})
RIDGEcvgridb = cv.GridSearchCV(ml.Ridge(solver='auto'), cv=10, param_grid={"alpha": [0.1, 1.0, 10.0]})

# Ridge Regression fitting on training range optimal parameters
RIDGEcva = RIDGEcvgrida.fit(yt1, yt)
RIDGEcvb = RIDGEcvgridb.fit(xt1, yt)

RIDGEpara = RIDGEcva.best_estimator_.alpha
RIDGEparb = RIDGEcvb.best_estimator_.alpha

# 2.2. Ridge Regression fitting on training range, predicting on forecasting range
RIDGEa = ml.Ridge(alpha=RIDGEpara, solver='auto')
RIDGEb = ml.Ridge(alpha=RIDGEparb, solver='auto')

# Ridge Regression fitting on training range
RIDGEfita = RIDGEa.fit(yt1, yt).predict(yt1)
tdata['RIDGEfita'] = RIDGEfita
RIDGEfitb = RIDGEb.fit(xt1, yt).predict(xt1)
tdata['RIDGEfitb'] = RIDGEfitb

# Ridge Regression fitting on training range score
RIDGEscorea = RIDGEa.fit(yt1, yt).score(yt1, yt)
RIDGEscoreb = RIDGEb.fit(xt1, yt).score(xt1, yt)

# Ridge Regression predicting on forecasting range
RIDGEfcsta = RIDGEa.fit(yt1, yt).predict(yf1)
fdata['RIDGEfcsta'] = RIDGEfcsta
RIDGEfcstb = RIDGEb.fit(xt1, yt).predict(xf1)
fdata['RIDGEfcstb'] = RIDGEfcstb

# 2.3. Chart Ridge Regression forecast
fdata.plot(y=['Yt', 'RIDGEfcsta'])
plt.title('Ridge Regression Forecast (yt1,yt)')
plt.legend(loc='upper left')
plt.show()

fdata.plot(y=['Yt', 'RIDGEfcstb'])
plt.title('Ridge Regression Forecast (xt1,yt)')
plt.legend(loc='upper left')
plt.show()

# 3. Ridge Regression forecasting accuracy

# 3.1. Calculate Scale-Dependant Mean Absolute Error MAE
ridgemaelista = []
ridgemaelistb = []
for i in range(0, len(yf)):
    ridgemaelista.insert(i, np.absolute(fdata['Yt'][i] - fdata['RIDGEfcsta'][i]))
    ridgemaelistb.insert(i, np.absolute(fdata['Yt'][i] - fdata['RIDGEfcstb'][i]))
RIDGEmaea = np.mean(ridgemaelista)
RIDGEmaeb = np.mean(ridgemaelistb)

# 3.2. Calculate Scale-Independent Symmetric Mean Absolute Percentage Error sMAPE, Mean Absolute Scaled Error MASE
# using random walk as forecasting benchmark.

# Symmetric Mean Absolute Percentage Error sMAPE is used instead as MAPE gives heavier penalty to negative errors.
ridgesmapelista = []
ridgesmapelistb = []
rndsmapelist = []
for i in range(0, len(yf)):
    ridgesmapelista.insert(i, (2 * np.absolute(fdata['Yt'][i] - fdata['RIDGEfcsta'][i])) /
                        (fdata['Yt'][i] + fdata['RIDGEfcsta'][i]))
    ridgesmapelistb.insert(i, (2 * np.absolute(fdata['Yt'][i] - fdata['RIDGEfcstb'][i])) /
                        (fdata['Yt'][i] + fdata['RIDGEfcstb'][i]))
    rndsmapelist.insert(i, (2 * np.absolute(fdata['Yt'][i] - fdata['Yt.1'][i])) /
                       (fdata['Yt'][i] + fdata['Yt.1'][i]))
RIDGEsmapea = np.mean(ridgesmapelista) * 100
RIDGEsmapeb = np.mean(ridgesmapelistb) * 100
RNDsmape = np.mean(rndsmapelist) * 100

RIDGEsmasea = RIDGEsmapea / RNDsmape
RIDGEsmaseb = RIDGEsmapeb / RNDsmape

# 4. Ridge Regression fitting score and forecasting accuracy comparison results printing
print("")
print("Optimal Alpha")
print("RIDGEfita:", RIDGEpara)
print("RIDGEfitb:", RIDGEparb)
print("")
print("Coefficient of Determination R^2")
print("RIDGEfita:", RIDGEscorea)
print("RIDGEfitb:", RIDGEscoreb)
print("")
print("Mean Absolute Error MAE")
print("RIDGEfcsta:", RIDGEmaea)
print("RIDGEfcstb:", RIDGEmaeb)
print("")
print("Symmetric Mean Absolute Percentage Error sMAPE")
print("RIDGEfcsta:", RIDGEsmapea)
print("RIDGEfcstb:", RIDGEsmapeb)
print("")
print("Mean Absolute Scaled Error MASE")
print("RIDGEfcsta:", RIDGEsmasea)
print("RIDGEfcstb:", RIDGEsmaseb)
print("")