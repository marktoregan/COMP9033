import pdb
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from matplotlib.colors import ListedColormap


class Abalone:

    """ Make predictions about abalone given various physical
    attributes. Predict the age with linear regression, or the sex
    with logistic regression. The dataset was downloaded from the UCI
    Machine Learning Repository (http://archive.ics.uci.edu/ml/).
    The data file contains the following columns:
    Name		Data Type	Meas.	Description1
    ----		---------	-----	-----------
    Sex		nominal			M, F, and I (infant)
    Length		continuous	mm	Longest shell measurement
    Diameter	continuous	mm	perpendicular to length
    Height		continuous	mm	with meat in shell
    Whole weight	continuous	grams	whole abalone
    Shucked weight	continuous	grams	weight of meat
    Viscera weight	continuous	grams	gut weight (after bleeding)
    Shell weight	continuous	grams	after being dried
    Rings		integer			+1.5 gives the age in years
    """

    def read_data(self, ycol='sex', fname='abalone.csv'):

        """ Extract data from file, split into train & test subsets.
        By default this returns the sex as the ordinate (ycol='Sex') for a
        logistic regression fit.
        """

        data = pd.read_csv(fname)
        cols = [col.lower() for col in data.columns.values]

        # Index of ycol
        if ycol == 'age':
            ycol = 'rings' # rings column corresponds to age...
        ind = [i for i, col in enumerate(cols) if col == ycol.lower()]

        if len(ind) > 1:
            raise ValueError('The value for ycol={} has multiple '+
                'entries?'.format(ycol))
        elif len(ind) == 0:
            pdb.set_trace()
            raise ValueError("There are no columns of {}".format(ycol))
        else:
            ind = ind[0]

        # Sex column is strings which I'll convert to -1, 0, or 1.
        # 'M' = 0 & 'F' = 1 & 'I' = -1
        sex = data.iloc[:,0].as_matrix()
        m, f, i = sex == 'M', sex == 'F', sex == 'I'
        sex[m], sex[f], sex[i] = 0., 1., -1.
        vsex = sex.reshape(len(sex), 1) #column vector

        # Add 1.5 to age column
        age = (data.iloc[:,8] + 1.5).as_matrix()
        vage = age.reshape(len(age), 1)

        # Append sex and age back to data matrix
        features = np.hstack([vsex, data.iloc[:, 1:8].as_matrix(), vage])

        # Get ordinate and abscissa from ind variable
        print 'Extracting {} for y-value.'.format(ycol)
        y = features[:,ind]
        X = np.delete(features, ind, axis=1)

        # Shuffle the data and split into train and test sest.
        X, y = shuffle(X, y)
        train_pct = 0.8
        upto = int(round(len(X) * train_pct))
        x_train, x_test = X[:upto, :], X[upto:, :]
        y_train, y_test = y[:upto], y[upto:]

        return ({'x_train': x_train, 'x_test': x_test,
                'y_train': y_train, 'y_test': y_test})


    def logistic(self, z):
        return 1 / (1 + np.exp(-z))


    def plot_classification_data(self, data1, data2, beta, logistic_flag=False):

        plt.figure()
        grid_size = .2
        features = np.vstack((data1, data2))
        # generate a grid over the plot
        x_min, x_max = features[:, 0].min() - .5, features[:, 0].max() + .5
        y_min, y_max = features[:, 1].min() - .5, features[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_size), \
            np.arange(y_min, y_max, grid_size))
        # color the grid based on the predictions
        if logistic_flag:
            pdb.set_trace()
            Z = self.logistic(np.dot(np.c_[xx.ravel(), yy.ravel(), \
                np.ones(xx.ravel().shape[0])], beta))
            colorbar_label=r"Value of f($X \beta)$"
        else:
            Z = np.dot(np.c_[xx.ravel(), yy.ravel(), \
                np.ones(xx.ravel().shape[0])], beta)
            colorbar_label=r"Value of $X \beta$"
        Z = Z.reshape(xx.shape)
        background_img = plt.pcolormesh(xx, yy, Z, cmap=plt.cm.coolwarm)


    def classification_error(self, predictions, class_labels):
        n = predictions.size
        num_of_errors = 0.
        for idx in xrange(n):
            if (predictions[idx] >= 0.5 and class_labels[idx]==0) or \
                (predictions[idx] < 0.5 and class_labels[idx]==1):
                num_of_errors += 1

        return num_of_errors/n


    def classify(self, ycol='sex', model='lr', do_pca=True):

        """ Classify if an abalone is Male or Female.
        """

        dat = self.read_data(ycol=ycol)
        xtrain, xtest = dat['x_train'], dat['x_test']
        ytrain, ytest = dat['y_train'], dat['y_test']

        # Remove -1 values in age (ytest/train):
        keep1 = np.where(ytest != -1)[0]
        ytest = ytest[keep1].astype(int)
        xtest = xtest[keep1,:]

        keep2 = np.where(ytrain != -1)[0]
        ytrain = ytrain[keep2].astype(int)
        xtrain = xtrain[keep2,:]

        # Reduce dimensionality to 2 so I can visualize.
        if do_pca:
            pca = PCA(n_components=2)
            xtrain = pca.fit_transform(xtrain) # x_train reduced.
            xtest = pca.fit_transform(xtest)

        # Train & predict logistic regression model.
        reg = linear_model.LogisticRegression()
        reg.fit(xtrain, ytrain)
        predict = reg.predict(xtest)

        err = self.classification_error(predict, ytest)
        print 'Error on the test set: {:4.2f}%'.format(err*100)

        pdb.set_trace()
        self.plot_classification_data(xtest[:,0], xtest[:,1], \
            reg.coef_.T, logistic_flag=True)

        # Also plot the training points.
        plt.scatter(class1_features[:, 0], class1_features[:, 1], \
            c='b', edgecolors='k', s=70)
        plt.scatter(class2_features[:, 0], class2_features[:, 1], \
            c='r', edgecolors='k', s=70)

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        color_bar = plt.colorbar(background_img, orientation='horizontal')
        color_bar.set_label(colorbar_label)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.show()


    def regression(self, model='ols', ycol='age', do_pca=True, alpha=0.3):

            """ Linear regression fitting. Not really relevant for Abalone data.
            - OLS, Ridge, Lasso, ElasticNet, BayseianRidge.
            """

            model = model.lower()
            if 'ols' in model:
                mod = linear_model.LinearRegression()
                print 'Using ordinary least squares regression.'
            elif 'ridge' in model:
                mod = linear_model.Ridge(alpha=alpha)
                print 'Using Ridge regression.'
            elif 'lasso' in model:
                mod = linear_model.Lasso(alpha=alpha)
            elif ('elastic' in model) or ('net' in model):
                mod = linear_model.ElasticNet(alpha=alpha, l1_ratio=0.1)
            elif 'bayseianridge' in model:
                mod = linear_model.BayseianRidge(alpha=alpha)
            else:
                raise NameError("model {} isn't set up.".format(model))

            dat = self.read_data(ycol=ycol)
            xtrain, xtest = dat['x_train'], dat['x_test']
            ytrain, ytest = dat['y_train'], dat['y_test']

            if do_pca:
                print 'Reducing the dimensionality of data with PCA...'
                pca = PCA()
                pca_init = pca.fit(xtrain)
                variance_ratio = pca_init.explained_variance_ratio_
                vr, n_comp = 0, 0
                while vr <= 0.9:
                    vr += variance_ratio[n_comp]
                    n_comp += 1
                print ('{} principle components gives '
                       '{:2.1f}% contribution.'.format(n_comp, vr*100))
                n_comp = 1
                # Use PCA with that many components to reduce dimensionality.
                pca = PCA(n_components=n_comp)
                xtrain = pca.fit_transform(xtrain) # x_train reduced.
                xtest = pca.fit_transform(xtest)

            # Train the model.
            mod.fit(xtrain, ytrain)
            y_predict = mod.predict(xtest)

            # Check out the errors.
            diff = ytest - y_predict
            err = np.dot(diff, diff) / len(diff)
            print 'Error: {}'.format(err)

            # Show results.
            plt.scatter(xtest, ytest,  color='black')
            plt.plot(xtest, y_predict, color='blue', linewidth=2)

            plt.show()

obj_a = Abalone()
#obj_a.read_data()
obj_a.regression()
