import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from kobra import SKSupervisedLearning
from kobra.tr_utils import time_now_str
import numpy as np

sample_file = '/kaggle/retina/reduced/features/train/features.csv'

df = pd.read_csv(sample_file)
n_bins = 100

feats = df.ix[:, :n_bins * 2].values.astype(np.float)
levels = df['level'].values


X_train, X_test, Y_train, Y_test = train_test_split(feats, levels, test_size = 0.2)

print "Read, train: {:d}, test: {:d}".format(X_train.shape[0], X_test.shape[0])

rf = SKSupervisedLearning(RandomForestClassifier, X_train, Y_train, X_test, Y_test)
#rf.estimation_params = {'max_depth' : [4, 10, 100], 'min_samples_leaf': [3, 5, 20], 
#                         'max_features': [1.0, 0.3, 0.1]}

# parameters tuned from the above
rf.train_params = {'n_estimators' : 500, 'max_features': 'sqrt'}
rf.scoring = "accuracy"
print "Instantiated classifier"

rf.fit_standard_scaler()
#rf.grid_search_classifier()

print "Starting: ", time_now_str()

a_train, a_test = rf.fit_and_validate()

print "Finished: ", time_now_str()

print "Accuracy: \n\tTrain: {:2.5f}\n\tTest: {:2.5f}".format(a_train, a_test)

rf.plot_confusion()