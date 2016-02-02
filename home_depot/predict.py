import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

# read data
datadir = '/Users/cavagnolo/ml_fun/home_depot/data/'
outdir = '/Users/cavagnolo/ml_fun/home_depot/output/'
df_train = pd.read_csv(datadir + 'train.csv', encoding="ISO-8859-1")
num_train = df_train.shape[0]
df_all = pd.read_hdf(datadir + 'all_prod_data.h5', 'df')

# form train/test
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']
y_train = df_train['relevance'].values
X_train = df_train.drop(['id', 'relevance'], axis=1).values
X_test = df_test.drop(['id', 'relevance'], axis=1).values

# classifiers
print 'Running bagging...'
rf = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1)
y_pred = clf.fit(X_train, y_train).predict(X_test)

# output result
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(outdir + 'submission.csv', index=False)
