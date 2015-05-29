import os
from os import path
import numpy as np
import csv
import shutil
import sys
from sklearn.cross_validation import train_test_split
from tr_utils import append_to_arr
from zipfile import ZipFile

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
        with open(self.labels_file, 'rb') as csvlabels:
            lablesreader = csv.reader(csvlabels)
            file_label = map(lambda x: (x[0], int(x[1])), [(row[0], row[1]) for row in lablesreader][1:])
        return file_label
    
    def _connect_labeled_data(self, inp_path, inputs, training):
        if self.labels == None:
            self.labels = self.get_labels_csv()

        X = np.array([])
        Y = np.array([])
        zip = False
        if self.isZip:
            zip = ZipFile(inp_path)

        for inp in inputs:
            inp_file = path.join(inp_path, inp) if not zip else inp
            x = np.fromfile(inp_file, dtype='int') if not zip else np.frombuffer(zip.read(inp_file), dtype='int')
            x = x.astype('float')
            X = append_to_arr(X, x)

            if training or self.validate:
                label_name = path.splitext(path.split(inp_file)[1])[0]
                label = filter(lambda x: x[0] == label_name, self.labels)[0][1]
                Y = np.append(Y, label)

            if self.debug:
                print "Processed: " + inp_file
        return X, Y

    def _connect_labeled_data_csv(self, inp_path, inputs, training):
        if self.labels == None:
            self.labels = self.get_labels_csv()

        X = np.array([])
        Y = np.array([])
        for inp in inputs:
            inp_file = path.join(inp_path, inp)
            x = np.loadtxt(inp_file)
            X = append_to_arr(X, x)

            if training or self.validate:
                label_name = path.splitext(inp)[0]
                label = filter(lambda x: x[0] == label_name, self.labels)[0][1]
                Y = np.append(Y, label)

            if self.debug:
                print "Processed: " + inp_file
        return X, Y

    def _get_inp_input_path(self, training):
        inputs = self.get_training_inputs() if training else self.get_val_inputs()
        inp_path = self.train_path if training else self.val_path
        
        return inputs, inp_path

    def connect_labeled_data(self, training):
        """
        Read the training/validation file names and produce two arrays, which once zipped
        and iterated over will form a tuple (itemI, classI)
        """
        inputs, inp_path = self._get_inp_input_path(training)
        return self._connect_labeled_data(inp_path, inputs, training)

    def connect_labeled_data_csv(self, training):
        """
        Read the training/validation file names and produce two arrays, which once zipped
        and iterated over will form a tuple (itemI, classI)
        """
        inputs, inp_path = self._get_inp_input_path(training)
        return self._connect_labeled_data_csv(inp_path, inputs, training)

    def prepare_inputs(self):
        """
        Read training, validation, labels and output them
        """
        if not self.validate:
            X_train, Y_train = self.connect_labeled_data(True)
            X_test, Y_test = self.connect_labeled_data(False)
        else:
            X_train, Y_train = self.connect_labeled_data(True)
            X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = self.test_size, random_state = 1234)
        return X_train, Y_train, X_test, Y_test

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
        x, xt, y, yt = train_test_split(X, Y, test_size = test_size, random_state = 1234)
        return x, y, xt, yt

    def prepare_inputs_csv(self):
        if not self.validate:
            X_train, Y_train = self.connect_labeled_data_csv(True)
            X_test, Y_test = self.connect_labeled_data_csv(False)
        else:
            X_train, Y_train = self.connect_labeled_data(True)
            X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = self.test_size, random_state = 1234)

        return X_train, Y_train, X_test, Y_test