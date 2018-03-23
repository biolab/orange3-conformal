"""Evaluation module contains methods for evaluation of conformal predictors.

Function :py:func:`run` produces Results of an appropriate type by using a Sampler on a given data set
to split it into a training and testing set.

Structure:

- Sampler (sampling methods)
    - :py:class:`RandomSampler`
    - :py:class:`CrossSampler`
    - :py:class:`LOOSampler`

- Results (evaluation results)
    - :py:class:`ResultsClass`
    - :py:class:`ResultsRegr`

- Evaluation methods
    - :py:func:`run`
    - :py:func:`run_train_test`
"""

import numpy as np
import time
from sklearn.model_selection import KFold

from Orange.classification import LogisticRegressionLearner
from Orange.data import Table

from orangecontrib.conformal.classification import TransductiveClassifier, ConformalClassifier, CrossClassifier
from orangecontrib.conformal.nonconformity import InverseProbability
from orangecontrib.conformal.regression import TransductiveRegressor, InductiveRegressor, CrossRegressor


class Sampler:
    """Base class for various data sampling/splitting methods.

    Attributes:
        data (Table): Data set for sampling.
        n (int): Size of the data set.

    Examples:
        >>> s = CrossSampler(Table('iris'), 4)
        >>> for train, test in s.repeat(3):
        >>>     print(train)
    """

    def __init__(self, data):
        """Initialize the data set."""
        self.data = data
        self.n = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        """Extending samplers should implement the __next__ method to return the selected
        and remaining part of the data.
        """
        raise NotImplementedError

    def repeat(self, rep=1):
        """Repeat sampling several times."""
        for r in range(rep):
            for train, test in self:
                yield train, test


class RandomSampler(Sampler):
    """Randomly samples a subset of data in proportion a:b.

    Attributes:
        k (float): Size of the selected subset.

    Examples:
        >>> s = RandomSampler(Table('iris'), 3, 2)
        >>> train, test = next(s)
    """

    def __init__(self, data, a, b):
        """Initialize the data set and the size of the desired selection."""
        super().__init__(data)
        self.k = a*self.n//(a+b)

    def __iter__(self):
        """Return a special iterator over a single split of data."""
        yield next(self)

    def __next__(self):
        """Splits the data based on a random permutation."""
        perm = np.random.permutation(self.n)
        train_ind, test_ind = perm[:self.k], perm[self.k:]
        return self.data[train_ind], self.data[test_ind]


class CrossSampler(Sampler):
    """Sample the data in :py:attr:`k` folds. Shuffle the data before determining the folds.

    Attributes:
        k (int): Number of folds.

    Examples:
        >>> s = CrossSampler(Table('iris'), 4)
        >>> for train, test in s:
        >>>     print(train)
    """
    def __init__(self, data, k):
        super().__init__(data)
        self.k = k
        self.kf = None

    def __next__(self):
        """Compute the next fold. Initializes a new k-fold split on each repetition of the entire
        sampling procedure.
        """
        if self.kf is None:
            self.kf = iter(KFold(self.k, shuffle=True).split(self.data))
        try:
            train_ind, test_ind = next(self.kf)
            return self.data[train_ind], self.data[test_ind]
        except StopIteration:
            self.kf = None
            raise StopIteration


class LOOSampler(CrossSampler):
    """Leave-One-Out sampler is a cross sampler with the number of folds equal to the size of the data set.

    Examples:
        >>> s = LOOSampler(Table('iris'))
        >>> for train, test in s:
        >>>     print(len(test))
    """

    def __init__(self, data):
        super().__init__(data, len(data))


class Results:
    """Contains results of an evaluation of a conformal predictor
    returned by the :py:func:`run` function.

    Examples:
        >>> cp = CrossClassifier(InverseProbability(LogisticRegressionLearner()), 5)
        >>> r = run(cp, 0.1, RandomSampler(Table('iris'), 2, 1))
        >>> print(r.accuracy())
    """

    def __init__(self):
        self.preds = []
        self.refs = []
        self.tm = 0

    def add(self, pred, ref):
        """Add a new predicted and corresponding reference value."""
        self.preds.append(pred)
        self.refs.append(ref)

    def concatenate(self, r):
        """Concatenate another set of results."""
        self.preds += r.preds
        self.refs += r.refs
        self.tm += r.tm

    def accuracy(self):
        """Compute the accuracy of the predictor averaging verdicts of individual predictions. This is the fraction
        of instances that contain the actual/reference class among the predicted ones for classification and the fraction of
        instances that contain the actual value within the predicted range for regression."""
        v = [p.verdict(r) for p, r in zip(self.preds, self.refs)]
        return np.mean(v)

    def time(self):
        return self.tm


class ResultsClass(Results):
    """Results of evaluating a conformal classifier. Provides classification specific efficiency measures.

    Examples:
        >>> cp = CrossClassifier(InverseProbability(LogisticRegressionLearner()), 5)
        >>> r = run(cp, 0.1, RandomSampler(Table('iris'), 2, 1))
        >>> print(r.singleton_criterion())
    """

    def accuracy(self, class_value=None, eps=None):
        """Compute accuracy for test instances with a given class value. If this parameter is not given,
        compute accuracy over all instances, regardless of their class."""
        v = [p.verdict(r, eps) for p, r in zip(self.preds, self.refs) if class_value is None or r == class_value]
        return np.mean(v)

    def confidence(self):
        """Average confidence of predictions."""
        return np.mean([pred.confidence() for pred in self.preds])

    def credibility(self):
        """Average credibility of predictions."""
        return np.mean([pred.credibility() for pred in self.preds])

    def confusion(self, actual, predicted):
        """Compute the number of singleton predictions of class `predicted` when the actual class is `actual`.

        Examples:
            Drawing a confusion matrix.

            >>> data = Table('iris')
            >>> cp = CrossClassifier(InverseProbability(LogisticRegressionLearner()), 3)
            >>> r = run(cp, 0.1, RandomSampler(data, 2, 1))
            >>> values = data.domain.class_var.values
            >>> form = '{: >20}'*(len(values)+1)
            >>> print(form.format('actual\\predicted', *values))
            >>> for a in values:
            >>>     c = [r.confusion(a, p) for p in values]
            >>>     print(('{: >20}'*(len(c)+1)).format(a, *c))
                actual\predicted         Iris-setosa     Iris-versicolor      Iris-virginica
                     Iris-setosa                  18                   0                   0
                 Iris-versicolor                   0                  14                   4
                  Iris-virginica                   0                   0                  12
        """
        return sum(pred.classes()[0] == predicted and ref == actual
                   for pred, ref in zip(self.preds, self.refs) if len(pred.classes()) == 1)

    def multiple_criterion(self):
        """Number of cases with multiple predicted classes."""
        c = [len(pred.classes()) > 1 for pred in self.preds]
        return np.mean(c)

    def singleton_criterion(self):
        """Number of cases with a single predicted class."""
        c = [len(pred.classes()) == 1 for pred in self.preds]
        return np.mean(c)

    def empty_criterion(self):
        """Number of cases with no predicted classes."""
        c = [len(pred.classes()) == 0 for pred in self.preds]
        return np.mean(c)

    def singleton_correct(self):
        """Fraction of singleton predictions that are correct."""
        c = [pred.verdict(ref) for pred, ref in zip(self.preds, self.refs) if len(pred.classes()) == 1]
        return np.mean(c)


class ResultsRegr(Results):
    """Results of evaluating a conformal regressor. Provides regression specific efficiency measures.

    Examples:
        >>> ir = InductiveRegressor(AbsErrorKNN(Euclidean(), 10, average=True))
        >>> r = run(ir, 0.1, RandomSampler(Table('housing'), 2, 1))
        >>> print(r.interdecile_range())
    """

    def widths(self):
        return [pred.width() for pred in self.preds]

    def median_range(self):
        """Median width of predicted ranges."""
        return np.median(self.widths())

    def mean_range(self):
        """Mean width of predicted ranges."""
        return np.mean(self.widths())

    def std_dev(self):
        """Standard deviation of widths of predicted ranges."""
        return np.std(self.widths())

    def interdecile_range(self):
        """Difference between the first and ninth decile of widths of predicted ranges."""
        w = self.widths()
        return np.percentile(w, 90) - np.percentile(w, 10)

    def interdecile_mean(self):
        """Mean width discarding the smallest and largest 10% of widths of predicted ranges."""
        w = self.widths()
        decile = int(0.1*len(w))
        return np.mean(w[decile:len(w)-decile])


def run(cp, eps, sampler, rep=1):
    """Run method is used to repeat an experiment one or more times with different splits of the dataset
    into a training and testing set. The splits are defined by the provided sampler. The conformal predictor
    itself might further split the testing set internally for its computations (e.g. inductive or cross predictors).

    Run the conformal predictor `cp` on the datasets defined by the provided sampler and number of repetitions
    and construct the results. Fit the conformal predictor on each training set returned by the sampler and
    evaluate it on the corresponding test set.
    Inductive conformal predictors use one third of the training set (random subset) for calibration.

    For more control over the exact datasets used for training, testing and calibration see :py:func:`run_train_test`.

    Returns:
        :py:class:`ResultsClass` or :py:class:`ResultsRegr`

    Examples:
        >>> cp = CrossClassifier(InverseProbability(LogisticRegressionLearner()), 5)
        >>> r = run(cp, 0.1, CrossSampler(Table('iris'), 4), rep=3)
        >>> print(r.accuracy(), r.empty_criterion())

    The above example uses a :py:class:`CrossSampler` to define training and testing datasets. Each fold is used as the test
    set and the rest as a training set. The entire process is repeated three times with different fold splits
    and results in 3*n predictions, where n is the size of the dataset.
    """

    classification = isinstance(cp, ConformalClassifier)
    results = ResultsClass() if classification else ResultsRegr()
    for train, test in sampler.repeat(rep):
        r = run_train_test(cp, eps, train, test)
        results.concatenate(r)
    return results


def run_train_test(cp, eps, train, test, calibrate=None):
    """Fits the conformal predictor `cp` on the training dataset and evaluates it on the testing set.
    Inductive conformal predictors use the provided calibration set or default to extracting one third
    of the training set (random subset) for calibration.

    Returns:
        :py:class:`ResultsClass` or :py:class:`ResultsRegr`

    Examples:
        >>> tab = Table('iris')
        >>> cp = CrossClassifier(InverseProbability(LogisticRegressionLearner()), 4)
        >>> r = run_train_test(cp, 0.1, tab[:100], tab[100:])
        >>> print(r.accuracy(), r.singleton_criterion())
    """

    classification = isinstance(cp, ConformalClassifier)
    results = ResultsClass() if classification else ResultsRegr()
    start = time.time()
    if isinstance(cp, TransductiveClassifier) or isinstance(cp, TransductiveRegressor) or \
            isinstance(cp, CrossClassifier) or isinstance(cp, CrossRegressor):
        cp.fit(train)
    else:  # Inductive predictor
        if calibrate is None:
            train, calibrate = next(RandomSampler(train, 2, 1))
        cp.fit(train, calibrate)
    for inst in test:
        results.add(cp.predict(inst, eps), inst.get_class())
    finish = time.time()
    results.tm = finish-start
    return results
