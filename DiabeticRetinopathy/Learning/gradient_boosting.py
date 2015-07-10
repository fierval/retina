from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from kobra import SKSupervisedLearning
from kobra import TrainFiles
import matplotlib.pylab as plt
from kobra.tr_utils import time_now_str
import numpy as np
import time
from sklearn.metrics.metrics import confusion_matrix

X_train, Y_train, X_test, Y_test = TrainFiles.from_csv(sample_file, test_size = 0.2)
print "Read, train: {:d}, test: {:d}".format(X_train.shape[0], X_test.shape[0])

gbc = SKSupervisedLearning(GradientBoostingClassifier, X_train, Y_train, X_test, Y_test)
#gbc.estimation_params = {'max_depth' : [4, 10, 100], 'min_samples_leaf': [3, 5, 20], 
#                         'max_features': [1.0, 0.3, 0.1]}

# parameters tuned from the above
gbc.train_params = {'n_estimators' : 7500, 'max_features': 0.1, 'max_depth': 10, 'min_samples_leaf': 5}
gbc.scoring = "accuracy"
print "Instantiated classifier"

gbc.fit_standard_scaler()
#gbc.grid_search_classifier()

print "Starting: ", time_now_str()

a_train, a_test = gbc.fit_and_validate()

print "Finished: ", time_now_str()

print "Accuracy: \n\tTrain: {:2.5f}\n\tTest: {:2.5f}".format(a_train, a_test)

#downsample the test set for plotting
X, _, Y, _ = train_test_split(X_test, Y_test, test_size = 0.95)

x = np.arange(Y.size)
f = plt.figure(figsize= (15, 7))
plt.scatter(x, Y, color='green')

for pred in gbc.clf.staged_predict(X):
    plt.plot(x, pred, color='red')

plot_confusion(gbc)