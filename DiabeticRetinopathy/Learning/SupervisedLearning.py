import numpy as np
from sklearn.grid_search import GridSearchCV

import sklearn.metrics as metrics
from sklearn import preprocessing as prep


from tr_utils import merge_two_dicts, isEmpty

class SKSupervisedLearning (object):
    """
    Thin wrapper around some learning methods
    """
    def __init__(self, classifier, X_train, Y_train, X_test, Y_test):
        """
        X_train, Y_train - training data: examples + corresponding class labels
        X_test, Y_test - validation data: examples + corresponding class labels
        """
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        self.X_train_scaled = np.array([])
        self.X_test_scaled = np.array([])

        self._classifier = classifier
        self._clf = None

        self._proba_train = None
        self._proba_test = None
        self._train_params = None
        self._estimation_params = None
        self._scaler = None

        # parameters for sklearn grid search
        self._jobs = -1
        self._cv = 10
        self._verbose = 0
        self._scoring = "log_loss"

    @property
    def scaler(self):
        return self._scaler

    @property
    def clf(self):
        if self._clf == None:
            self._clf = self._classifier(**self.train_params) if self.train_params != None else self._classifier()

        return self._clf

    @property
    def proba_train(self):
        return self._proba_train
    @property
    def proba_test(self):
        return self._proba_test

    @property
    def train_params(self):
        """
        Training parameter dictionary specific to each learner
        """
        return self._train_params

    @train_params.setter
    def train_params(self, val):
        self._train_params = val

    @property
    def estimation_params(self):
        """
        Dictionary of paramters to estimate, specific to each learner:
        e.g.:
        {'gamma': [0.001, 0.1, 1], 'C': [1, 10, 100]}
        """
        return self._estimation_params

    @estimation_params.setter
    def estimation_params(self, val):
        self._estimation_params = val

    @property
    def jobs(self):
        return self._jobs

    @jobs.setter
    def jobs(self, val):
        self._jobs = val

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, val):
        self._cv = val

    @property
    def scoring(self):
        return self._scoring

    @scoring.setter
    def scoring(self, val):
        self._scoring = val

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, val):
        self._verbose = val

    @property
    def proba_train(self):
        return self._proba_train

    @property
    def proba_test(self):
        return self._proba_test

    def _pick_examples(self):
        '''
        If we have scaled examples - pick them, else pick X_train, X_test
        '''
        return (self.X_train, self.X_test) \
            if isEmpty(self.X_train_scaled) or isEmpty(self.X_test_scaled) \
            else (self.X_train_scaled, self.X_test_scaled)

    def remove_scaling(self):
        self.X_test_scaled = None
        self.X_train_scaled = None

    def grid_search_classifier(self) :
        """
        Grid search for the best classifier, given parameters.
        Returns best score
        Sets the classifier to the best classifier given training and estimation parameters
        See sklearn GridSearchCV for details
        """

        gs = False

        if self.train_params == None and self.estimation_params == None:
            raise AttributeError("Cannot have train_params and estimation_params both absent")
        # first - grid-search for the best parameters
        if self.estimation_params:
            X_train, X_test = self._pick_examples()
            Y_train = self.Y_train

            clf = self._classifier(**self.train_params) if self.train_params != None else self._classifier()
            gs = GridSearchCV(clf, self.estimation_params, scoring = self.scoring, cv = self.cv, n_jobs=self.jobs, verbose = self.verbose)
            gs.fit(X_train, Y_train)

            print gs.best_params_
            print gs.best_score_
    
        # if we have specified parameters of our own - we need to add those
        if gs:
            self.train_params = merge_two_dicts(gs.best_params_, self.train_params) if self.train_params != None else gs.best_params_

        self._clf = self._classifier(**self.train_params)
        return gs.best_score_
    
    def _fit_scaler(self, scaler_class, X):
        return scaler_class().fit(X)
    
    # TODO: other scalers?
    def fit_standard_scaler(self):
        """
        Standard scaler scales samples 'vertically', (by feature), by removing the mean and reducing to unit std.
        Computes a scaler and transforms both train and validation sets based upon it
        """
        self._scaler = self._fit_scaler(prep.StandardScaler, self.X_train)
        self.X_train_scaled = self._scaler.transform(self.X_train)
        self.X_test_scaled = self._scaler.transform(self.X_test)
        
    def fit_and_validate(self):
        '''
        Returns training & testing log loss
        '''
        X_train, X_test = self._pick_examples()

        # shorthand
        Y_train = self.Y_train
        Y_test = self.Y_test

        self.clf.fit(X_train, Y_train)

        # get probabilities
        self._proba_train = self.clf.predict_proba(X_train)
        self._proba_test = self.clf.predict_proba(X_test)

        return metrics.log_loss(Y_train, self.proba_train), np.array([]) if isEmpty(Y_test) else metrics.log_loss(Y_test, self.proba_test)

    def predict_actual(self, X_actual_test):
        '''
        Return actual prediction on a set where we don't have labels
        '''
        return self.clf.predict_proba(X_actual_test)    