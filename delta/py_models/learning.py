import sys
import os
import datetime
import time
import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn import *

# target field
tname = 'gts_1'
folds = 10

# get files
cwd = os.getcwd()
dir_path  = os.path.join(cwd, '../data')
input_file_path = os.path.join(dir_path, 'sample_w_status.csv')

# read data
input_data = pd.read_csv(input_file_path, delimiter=',')
n_rows = float(input_data.shape[0])
n_cols = float(input_data.shape[0])
print("Opened %s with size (%f, %f)" % (input_file_path, n_rows, n_cols))

# clean-up dates
print("Fixing dates...")
mabbr = {v: k for k,v in enumerate(calendar.month_abbr)}
for idx, d in enumerate(input_data.response_dt):
    date = d.split('-')
    date[1] = str(mabbr[date[1]])
    date = datetime.datetime.strptime('-'.join(date), '%d-%m-%y').timetuple()
    input_data.response_dt[idx] = time.mktime(date)
print("Done.")

# hunt down nan's
print "\nInitial shape:",input_data.shape
print("\nFound NaN's in columns:")
for cname in sorted(input_data.columns):
    n_nans = input_data[cname].isnull().sum()
    if n_nans > 0 and cname != tname:
        perc = round(100.*(n_nans/n_rows),2)
        if perc >= 25.:
            check = '<---- PROBLEM! Removing column.'
            input_data.pop(cname)
        else:
            check = 'Removing rows.'
            input_data = input_data[~input_data[cname].isnull()]
        print(cname, n_nans, str(perc)+"%", check)
print "\nFinal shape:",input_data.shape

# binary one hot encode ALL categorical data
print("\nOne hot encoding ALL categorical values...")
print("...Started with %i columns" % input_data.shape[1])
cat_dict = input_data[input_data.columns].to_dict(outtype='records')
vec = feature_extraction.DictVectorizer()
cat_vec = vec.fit_transform(cat_dict).toarray()
data_vec = pd.DataFrame(cat_vec)
vec_cols = vec.get_feature_names()
data_vec.columns = vec_cols
data_vec.index = input_data.index
input_data = input_data.drop(input_data.columns, axis=1)
input_data = input_data.join(data_vec)
print("...Ended with %i columns" % input_data.shape[1])

# pull out respondents w/o gts_1 for prediction
x_pred = input_data[input_data[tname].isnull()]
x_pred.pop(tname)

# separate into features and target(s)
X = input_data[~input_data[tname].isnull()]
y = X.pop(tname)

# k-folds
cv = cross_validation.StratifiedKFold(y.values, 3)

# standardize data
X_scl = preprocessing.StandardScaler().fit_transform(X)

# split into train and test sets
xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(
    X_scl, y.values, train_size=0.6)

# binarize targets if we need them
ytrain_bi = preprocessing.label_binarize(ytrain, classes=[1,2,3,4,5])
ytest_bi = preprocessing.label_binarize(ytest, classes=[1,2,3,4,5])

####################################################################################
# linear classification

linreg = linear_model.LinearRegression()
linreg.fit(xtrain, ytrain_bi)
lr_preds = linreg.predict(xtest)
lr_perf = metrics.roc_auc_score(ytest_bi, lr_preds)
lr_prec = metrics.average_precision_score(ytest_bi, lr_preds)
print 'LinReg: Area under the ROC curve = {}'.format(lr_perf)
print 'LinReg: Avg precision = {}'.format(lr_prec)

####################################################################################
# one vs rest classification

one_rest = multiclass.OneVsRestClassifier(svm.LinearSVC())
one_rest.fit(xtrain, ytrain)
one_rest_preds = one_rest.predict(xtest)
one_rest_perf = metrics.confusion_matrix(ytest, one_rest_preds)
one_rest_prec = metrics.accuracy_score(ytest, one_rest_preds)
one_rest_report = metrics.classification_report(ytest, one_rest_preds)
print 'OneRest: Confusion Matrix'
print one_rest_perf
print 'OneRest: Avg precision = {}'.format(one_rest_prec)
print one_rest_report

####################################################################################
# decision trees
tree = grid_search.GridSearchCV(ensemble.RandomForestClassifier(),
                                {'max_depth': np.arange(3, 10)},
                                 n_jobs=-1)
tree.fit(xtrain, ytrain)
tree_preds = tree.predict_proba(xtest)
norm_preds = tree_preds
for idx, n in enumerate(tree_preds):
    lowi = n < max(n)
    n[lowi] = 0
    n[~lowi] = 1
    norm_preds[idx] = n
tree_performance = metrics.roc_auc_score(ytest_bi, tree_preds)
tree_prec = metrics.average_precision_score(ytest_bi, tree_preds)
tree_report = metrics.classification_report(ytest_bi, norm_preds)
print 'Tree: Area under the ROC curve = {}'.format(tree_performance)
print 'Tree: Avg precision = {}'.format(tree_prec)
print tree_report

# predictions
pred = linreg.predict(x_pred)
print pred
pred = one_rest.predict(x_pred)
print pred
pred = tree.predict(x_pred)
print pred

#########
#########
# TO DO #
#########
#########


####################################################################################
# lasso classification, is broken
# lasso = grid_search.GridSearchCV(linear_model.Lasso(),
#                                  {'alpha': np.logspace(-10, -8, 5)},
#                                  {'max_iter': 5000},
#                                  n_jobs=-1)
# lasso.fit(xtrain, ytrain_bi)
# lasso_preds = lasso.predict(xtest)
# lasso_performance = metrics.roc_auc_score(ytest_bi, lasso_preds)
# print 'Lasso: Area under the ROC curve = {}'.format(lasso_performance)

####################################################################################
# gbm classification
# gbm = ensemble.GradientBoostingClassifier(n_estimators=500)
# Parallel(n_jobs=4, backend="threading")(gbm.fit(xtrain, ytrain))
# gbm_preds = gbm.predict_proba(xtest)[:, 1]
# gbm_performance = metrics.roc_auc_score(ytest, gbm_preds)
# print 'GBM: Area under the ROC curve = {}'.format(gbm_performance)
# importances = pd.Series(gbm.feature_importances_, index=data.columns)
# print importances.order(ascending=False)[:10]

####################################################################################
# ridge classification
# ridge = grid_search.GridSearchCV(linear_model.Ridge(),
#                                  {'alpha': np.logspace(-10, 10, 10)},
#                                  n_jobs=-1)
# ridge.fit(xtrain, ytrain_bi)
# ridge_preds = ridge.predict(xtest)
# norm_preds = ridge_preds
# for idx, n in enumerate(ridge_preds):
#     lowi = n < max(n)
#     n[lowi] = 0
#     n[~lowi] = 1
#     norm_preds[idx] = n
# ridge_performance = metrics.roc_auc_score(ytest_bi, ridge_preds)
# ridge_prec = metrics.average_precision_score(ytest_bi, ridge_preds)
# ridge_report = metrics.classification_report(ytest_bi, norm_preds)
# print 'Ridge: Area under the ROC curve = {}'.format(ridge_performance)
# print 'Ridge: Avg precision = {}'.format(ridge_prec)
# print ridge_report

# bagging = ensemble.BaggingRegressor()
# nbayes = naive_bayes.GaussianNB()
# build a pipeline
# # pipe = pipeline.Pipeline(steps=[('pca', pca),
# #                                 ('logistic', logistic)])

####################################################################################
# # pca
# pca = decomposition.PCA()
# print "Original dataset shape:\n%i samples\n%i features\n" % (data_scl.shape[0], data_scl.shape[1])
# pca = decomposition.PCA(n_components=4, whiten=True)
# data_pca = pca.fit(data_scl).transform(data_scl)
# print "Reduced dataset shape:\n%i samples\n%i features\n" % (data_pca.shape[0], data_pca.shape[1])

# # Percentage of variance explained for each components
# print "Variance ratio (first 4 components):\n%s\n" % str(pca.explained_variance_ratio_)

# # describe the new subspace in terms of features
# print "Meaning of the 4 components:"
# for component in pca.components_:
#     for value, name in zip(component, data.columns):
#         print "(%.3f x %s)" % (value, name)
# # plot pca spectrum
# pca.fit(X)

# # Prediction
# # todo: select number of components
# n_components = [20, 40, 60]
# Cs = np.logspace(-4, 4, 3)
# penalty = 'l1'

# # use the pipeline
# estimator = grid_search.GridSearchCV(
#                                      dict(pca__n_components=n_components,
#                                           logistic__C=Cs,
#                                           logistic__penalty=penalty),
#                                      n_jobs=-1,
#                                      c_grid,
#                                      score_func = metrics.zero_one_loss,
#                                      cv = folds)

# estimator.fit(xtrain, ytrain)
# print estimator.best_params_, 1.0 - estimator.best_score_

# rates = np.array([1.0 - x[1] for x in estimator.grid_scores_])
# stds = [np.std(1.0 - x[2]) / math.sqrt(folds) for x in
#         estimator.grid_scores_]

# plt.fill_between(cs, rates - stds, rates + stds, color = 'steelblue',
# alpha = .4)
# plt.plot(cs, rates, 'o-k', label = 'Avg. error rate across folds')
# plt.xlabel('C (regularization parameter)')
# plt.ylabel('Avg. error rate (and +/- 1 s.e.)')
# plt.legend(loc = 'best')
# plt.gca().grid()

# # make a plot
# # plt.figure(1, figsize=(4, 3))
# # plt.clf()
# # plt.axes([.2, .2, .7, .7])
# # plt.plot(pca.explained_variance_, linewidth=2)
# # plt.axis('tight')
# # plt.xlabel('PCA Components')
# # plt.ylabel('Explained Variance')
# # plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
# #             linestyle=':', label='n_components chosen')
# # plt.legend(prop=dict(size=12))
# # plt.show()

# # find any k-means clusters in data
# k_means = cluster.KMeans(n_clusters=5, n_jobs=-1)
# k_means.fit(data_pca)
# y_pred = k_means.predict(data_pca)
# target_names = ['1', '2', '3', '4', '5']

# # plot the result
# for c, i, target_name in zip("bgcmr", [0, 1, 2, 3, 4], target_names):
#     plt.scatter(data_pca[y_pred == i, 0], data_pca[y_pred == i, 1], c=c, label=target_name)
# plt.plot(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], 'r*', label='centers', ms=18)
# plt.legend()
# plt.show()
