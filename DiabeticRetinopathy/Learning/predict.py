import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from kobra import SKSupervisedLearning
from kobra.tr_utils import time_now_str
import numpy as np
from os import path

sample_file = '/kaggle/retina/reduced/features/train/features.csv'
test_file = '/kaggle/retina/reduced/features/test/features.csv'
pred_dir = '/kaggle/retina'

df = pd.read_csv(sample_file)
df_test = pd.read_csv(test_file)

n_bins = 100

X_train = df.ix[:, :n_bins * 2].values.astype(np.float)
Y_train = df['level'].values

X_test = df_test.ix[:, :n_bins * 2].values.astype(np.float)
Y_test = np.array([])

images = df_test['name'].values

print "Read, train: {:d}, test: {:d}".format(X_train.shape[0], X_test.shape[0])

rf = SKSupervisedLearning(RandomForestClassifier, X_train, Y_train, X_test, Y_test)

# parameters tuned from the above
rf.train_params = {'n_estimators' : 1000, 'max_features': 'sqrt'}
rf.scoring = "accuracy"
print "Instantiated classifier"

print "Starting: ", time_now_str()

rf.fit_and_validate()
Y_test = rf.clf.predict(X_test)
pred = pd.DataFrame(np.array([images, Y_test]).transpose(), columns = ['image', 'level'])
pred.to_csv(path.join(pred_dir, "prediction.csv"), index = False)

print "Finished: ", time_now_str()
