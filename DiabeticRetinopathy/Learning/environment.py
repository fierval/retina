import os
from os import path

# common environment variables
root_path = "/kaggle/retina"

# train/test directories
train_path = path.join(root_path, 'train')
test_path = path.join(root_path, 'test')
raw_train = path.join(train_path, 'raw')
raw_test = path.join(test_path, 'raw')
sample_train = path.join(train_path, 'sample')

# in CSV representation
inp_file = path.join(train_path, "1dlbp.csv")
sample_file = path.join(train_path, "1dlbp_sample.csv")
labels_file = path.join(root_path, "trainLabels.csv")


