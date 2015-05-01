from SupervisedLearning import SKSupervisedLearning
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from environment import inp_file, sample_file, train_path, raw_train, sample_train, labels_file
from sklearn.cross_validation import train_test_split
from train_files import TrainFiles
from skimage.io.collection import ImageCollection
import os
from os import path
from shutil import copy, rmtree

def get_sample_csv(sample_size = 0.1):
    X, Y, _, _ = TrainFiles.from_csv(inp_file, test_size = sample_size)
    tf = TrainFiles(train_path)

    tf.dump_to_csv(sample_file, X, Y)

def get_sample_of_class(classs, labels, input_path, num = -1):
    """
    Returns images sampled from files by class
    """
        
    existing_files = pd.DataFrame([path.splitext(f)[0] for f in os.listdir(input_path)], columns =[labels.columns[0]])
    class_labels = pd.DataFrame(labels[labels['level'] == classs]['image'], columns =[labels.columns[0]])
    class_labels = class_labels.merge(existing_files, on=class_labels.columns[0])
    if num == -1:
        num = class_labels.count()

    files = [path.join(raw_train, c + ".jpeg") for c in class_labels[:num][class_labels.columns[0]]]
    return ImageCollection(files, conserve_memory = True)


def sample_files(labels, size, inp_path = raw_train, out_path = sample_train):
    if path.exists(out_path):
        rmtree(out_path)
    os.makedirs(out_path)

    files = labels['image']
    levels = labels['level']
    picked_files, _, picked_labels, _ = train_test_split(files, levels, train_size = size)

    inp_files = [path.join(inp_path, f + ".jpeg") for f in picked_files]
    out_files = [path.join(out_path, f + ".jpeg") for f in picked_files]
    for src, dst in zip(inp_files, out_files):
        copy(src, dst)
    
