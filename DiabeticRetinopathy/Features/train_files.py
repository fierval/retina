import os
from os import path
import numpy as np
import csv
import shutil
import sys
from sklearn.cross_validation import train_test_split
from tr_utils import append_to_arr
from zipfile import ZipFile
import pandas as pd

class TrainFiles(object):
    """
    Utilities for dealing with the file system
    """
    def __init__(self, train_path = None, val_path = None,  labels_file = None, debug = True, test_size = 0.1, floor = 0., cutoff = sys.maxint):
        '''
        If validate is set to false - don't attempt to match test set to labels
        '''
        self.train_path = train_path
        self.labels_file = labels_file
        self._cutoff = cutoff
        self._floor = floor
        self.labels = None
        self.debug = debug
        self.validate = (val_path == None) # perform validation if no test set is specified
        self.test_size = test_size
        self.val_path = val_path
        self.isZip = dir != None and path.splitext(train_path)[1] == '.zip'

    def __str__(self):
        return 'train: {0}, validate: {1}, labels: {2}'.format(self.train_path, self.val_path, self.labels_file)

    @property
    def cutoff(self):
        """
        Max file size to consider when listing directory content
        """
        return self._cutoff

    @cutoff.setter
    def cutoff(self, val):
        self._cutoff = val

    @property
    def floor(self):
        return self._floor

    @floor.setter
    def floor(self, val):
        self._floor = val

    def get_size(self, file, dir) :
        return os.stat(path.join(dir, file)).st_size

    def _get_inputs(self, dir):
        if not self.isZip:
            return filter (lambda x: not path.isdir(path.join(dir, x)) and self.get_size(x, dir) > self.floor and self.get_size(x, dir) <= self.cutoff, os.listdir(dir))
        else:
            with ZipFile(dir) as zip:
                l = zip.infolist()
                return map(lambda x: x.filename, filter(lambda x: x.file_size > self.floor and x.file_size <= self.cutoff, l))

    def get_training_inputs(self):
        """
        retrieves file names (not full path) of files containing training "image" data
        """
        return self._get_inputs(self.train_path)

    def get_val_inputs(self):
        """
        retrieves file names (not full path) of files containing training "image" data
        """
        return self._get_inputs(self.val_path)

    def get_labels_csv(self):
        """
        retrieves the values of each class labels assuming they are stored as the following CSV:
        
        | ID | Class |

        """
        return pd.read_csv(self.labels_file, header=0, delimiter=',')
    
    def connect_labeled_data(self, training, load_from_file):
        inputs, inp_path = self._get_inp_input_path(training)

        self.labels = self.get_labels_csv()

        X = np.array([])

        # TODO: this is probably not the best way of doing it
        zip = False
        if self.isZip:
            zip = ZipFile(inp_path)

        for inp in inputs:
            inp_file = path.join(inp_path, inp) if not zip else inp
            x = load_from_file(inp_file)
            x = np.r_[[path.splitext(inp)[0]], x]
            X = append_to_arr(X, x)

            if self.debug:
                print "Processed: " + inp_file
        
        id = self.labels.columns[0]
        dfX = pd.DataFrame(X)
        dfX.columns=np.r_[[id], range(X.shape[1] - 1)]
        dfX = dfX.merge(self.labels, how='inner', on=id)
        
        return np.array(dfX[range(1, X.shape[1])]).astype(float), np.array(dfX.Class).astype(int)

    def _get_inp_input_path(self, training):
        inputs = self.get_training_inputs() if training else self.get_val_inputs()
        inp_path = self.train_path if training else self.val_path
        
        return inputs, inp_path

    def _prepare_inputs(self, load_func):
        if not self.validate:
            X_train, Y_train = self.connect_labeled_data(True, load_func)
            X_test, Y_test = self.connect_labeled_data(False, load_func)
        else:
            X_train, Y_train = self.connect_labeled_data(True, load_func)
            X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = self.test_size, random_state = 1234)
        return X_train, Y_train, X_test, Y_test

    def prepare_inputs(self):
        """
        Read training, validation, labels and output them
        """
        zip = False
        if self.isZip:
            zip = ZipFile(inp_path)

        load_func = (lambda inp_file: np.fromfile(inp_file, dtype='int') if not zip else np.frombuffer(zip.read(inp_file), dtype='int'))
        self._prepare_inputs(load_func)

    def prepare_inputs_csv(self):
        return self._prepare_inputs(np.loadtxt)

    @staticmethod
    def dump_to_csv(csvf, x, y):
        with open(csvf, "wb") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            row = ['Feature ' + str(i) for i in range(0, x[0].size)]
            row.append('Class')
            csv_writer.writerow(row)
            for a in zip(x, y):
                row = [f for f in a[0]]
                row.append(a[1])
                csv_writer.writerow(row)
    
    @staticmethod
    def from_csv(csvf, test_size = 0.1):
        with open (csvf, "rb") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            skip = csv.Sniffer().has_header(csv_file.read(1024))
        X = np.loadtxt(csvf, delimiter = ',', skiprows = 1 if skip else 0)
        Y = X[:, -1]
        X = X[:, :-1].astype('float')

