import sys

import numpy as np
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool

from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import check_X_y, check_array


class CustomBaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=20, samp_percent=0.10, replacement=True,
                 weighted=True, proba_voting=True, print_progress=False, n_jobs=-1):
        super().__init__()
        self.n_estimators = n_estimators
        self.samp_percent = samp_percent
        self.replacement = replacement
        self.weighted = weighted
        self.proba_voting = proba_voting
        self.print_progress = print_progress
        self.n_jobs = n_jobs

        self.classifiers_ = []
        self.precisions_ = []
        self.sample_ys_ = []
        self.sample_yhats_ = []

        self.conf = []
        self.roc = []
        self.auc = []
        self.cost = []

    def _bag_indexes(self, x, y):
        np.random.seed()
        aug_matrix = np.hstack([x, y.reshape((len(y), 1))])
        np.random.shuffle(aug_matrix)
        size = int(len(y) * self.samp_percent) if not self.replacement else len(y)
        indexes = np.random.choice(range(len(y)), size=size, replace=self.replacement)

        # get training indexes
        training_x = aug_matrix[indexes][:, :-1]
        training_y = aug_matrix[indexes][:, -1]

        # get testing indexes
        inv_indexes = [i for i in range(len(aug_matrix)) if i not in indexes]
        testing_x = aug_matrix[inv_indexes][:, :-1]
        testing_y = aug_matrix[inv_indexes][:, -1]
        return training_x, training_y, testing_x, testing_y

    def fit_single_classifier(self, i, x, y):
        if self.print_progress:
            sys.stdout.write('Estimator ' + str(i) + ': ')

        # Bagging samples
        training_x, training_y, testing_x, testing_y = self._bag_indexes(x, y)

        # Fitting classifier
        c = MLPClassifier()
        c.fit(training_x, training_y)

        # Getting classifier f1 score
        testing_yhat = c.predict(testing_x)
        precision = f1_score(testing_y, testing_yhat, average='weighted')

        if self.print_progress:
            sys.stdout.write('Fitted!\n')

        return (c, precision, testing_y, testing_yhat)

    def fit(self, x, y):
        x, y = check_X_y(x, y)
        x, y = x.copy(), y.copy()

        self.n_features_ = x.shape[1]
        self.n_classes_ = len(np.unique(y))
        self.classes_, y = np.unique(y, return_inverse=True)

        # Creating pool
        pool = Pool(processes=1 if self.n_jobs <= -1 else min(cpu_count(), self.n_jobs))

        # Mapping an array of args to a function (MapReduce)
        results = pool.map(self.fit_single_classifier,
                           range(self.n_estimators),
                           [x] * self.n_estimators,
                           [y] * self.n_estimators)
        self.classifiers_ = [item[0] for item in results]
        self.precisions_ = [item[1] for item in results]
        self.sample_ys_ = [item[2] for item in results]
        self.sample_yhats_ = [item[3] for item in results]

        return self

    def predict(self, x):
        x = check_array(x)

        results = np.zeros((x.shape[0], self.n_classes_))
        for classifier, precision in zip(self.classifiers_, self.precisions_):
            if self.proba_voting:
                res = classifier.predict_proba(x)
                results += res * (precision if self.weighted else 1)
            else:
                res = classifier.predict(x)
                for i in range(len(res)):
                    results[i][res[i]] += precision if self.weighted else 1

        # select index of largest row in x
        result = np.argmax(results, axis=1)

        return self.classes_[result]

    def predict_proba(self, x):
        yhats = []
        for clf in self.classifiers_:
            yhats.append(clf.predict_proba(x))
        return np.sum(yhats, axis=0) / self.n_estimators

    def predict_single_classifier_proba(self, x, classifier_index):
        return self.classifiers_[classifier_index].predict_proba(x)