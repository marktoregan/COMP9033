from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = GridSearchCV(svr, parameters,cv=10)
clf.fit(iris.data, iris.target)

#parameters, cv=5
#clf = GridSearchCV(PhraseClassifier(), parameters, cv=5)


sorted(clf.cv_results_.keys())

print(clf.cv_results_.keys())

print('Best parameters: %s' % clf.best_params_)

