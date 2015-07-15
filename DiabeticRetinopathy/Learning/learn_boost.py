import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from kobra.tr_utils import time_now_str
import numpy as np
import sklearn.preprocessing as prep
from sklearn import metrics

sample_file = '/kaggle/retina/reduced/features/sample/features.csv'

df = pd.read_csv(sample_file)
n_bins = 100

feats = df.ix[:, :n_bins * 2].values.astype(np.float)
levels = df['level'].values
names = df['name'].values

X_train, X_test, Y_train, Y_test = train_test_split(feats, levels, test_size = 0.2)

print "Read, train: {:d}, test: {:d}".format(X_train.shape[0], X_test.shape[0])

inner_estimator = DecisionTreeClassifier(max_depth = 15, class_weight = 'auto')
abc = AdaBoostClassifier(base_estimator = inner_estimator, n_estimators = 175)

print "Instantiated classifier"
print "Starting: ", time_now_str()

#scaler = prep.StandardScaler().fit(X_train)
#X_train_scaled = scaler.transform(X_train)
#X_test_scaled = scaler.transform(X_test)

abc.fit(X_train, Y_train)
a_train = metrics.accuracy_score(Y_train, abc.predict(X_train))
a_test = metrics.accuracy_score(Y_test, abc.predict(X_test))

print "Finished: ", time_now_str()

print "Accuracy: \n\tTrain: {:2.5f}\n\tTest: {:2.5f}".format(a_train, a_test)

