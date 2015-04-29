from SupervisedLearning import SKSupervisedLearning
from sklearn.ensemble import GradientBoostingClassifier

from environment import inp_file, sample_file, train_path
from sklearn.cross_validation import train_test_split
from train_files import TrainFiles

# we will be left with 40% of the original data
X, Y, _, _ = TrainFiles.from_csv(inp_file, test_size = 0.6)
tf = TrainFiles(train_path)

tf.dump_to_csv(sample_file, X, Y)
