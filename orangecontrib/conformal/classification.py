"""Classification module contains methods for conformal classification.

Conformal classifiers predict a set of classes (not always a single class) under a given
significance level (error rate). Every classifier works in combination with a nonconformity measure
and on average predicts the correct class with the given error rate. Lower error rates result in
smaller sets of predicted classes.

Structure:

- ConformalClassifier
    - Transductive (:py:class:`TransductiveClassifier`)
    - Inductive (:py:class:`InductiveClassifier`)
    - Cross (:py:class:`CrossClassifier`)
"""

from copy import deepcopy

import numpy as np
from sklearn.model_selection import KFold

from Orange.data import Table, Instance

from orangecontrib.conformal.base import ConformalPredictor
from orangecontrib.conformal.nonconformity import ClassNC


class PredictionClass:
    """Conformal classification prediction object,
    which is produced by the :py:func:`ConformalClassifier.predict` method.

    Attributes:
        p (List): List of pairs (p-value, class)
        eps (float): Default significance level (error rate).

    Examples:
        >>> train, test = next(LOOSampler(Table('iris')))
        >>> tcp = TransductiveClassifier(InverseProbability(NaiveBayesLearner()), train)

        >>> prediction = tcp.predict(test[0], 0.1)
        >>> print(prediction.confidence(), prediction.credibility())

        >>> prediction = tcp.predict(test[0])
        >>> print(prediction.classes(0.1), prediction.classes(0.9))
    """

    def __init__(self, p, eps):
        """Initialize the prediction.

        Args:
            p (List): List of pairs (p-value, class)
            eps (float): Default significance level (error rate).
        """
        self.p = p
        self.eps = eps

    def classes(self, eps=None):
        """ Compute the set of classes under the default or given `eps` value.

        Args:
            eps (float): Significance level (error rate).

        Returns:
            List of predicted classes.
        """
        if eps is None:
            eps = self.eps
            assert(eps is not None)
        return [y for p_y, y in self.p if p_y > eps]

    def verdict(self, ref, eps=None):
        """Conformal classification prediction is correct when the actual class appears
        among the predicted classes.

        Args:
            ref: Reference/actual class
            eps (float): Significance level (error rate).

        Returns:
            True if the prediction with default or specified `eps` is correct.
        """
        return ref in self.classes(eps)

    def confidence(self):
        """Confidence is an efficiency measure of a single prediction.

        Computes minimum :math:`\\mathit{eps}` that would still result in a prediction of a single label.
        :math:`\\mathit{eps} = \\text{second\_largest}(p_i)`

        Returns:
            float: Confidence :math:`1-\\mathit{eps}`.
        """
        return 1-sorted([p_y for p_y, y in self.p], reverse=True)[1]

    def credibility(self):
        """Credibility is an efficiency measure of a single prediction.
        Small credibility indicates an unusual example.

        Computes minimum :math:`\\mathit{eps}` that would result in an empty prediction set.
        :math:`\\mathit{eps} = \\text{max}(p_i)`

        Returns:
            float: Credibility :math:`\\mathit{eps}`.
        """
        return max(p_y for p_y, y in self.p)


class ConformalClassifier(ConformalPredictor):
    """Base class for conformal classifiers."""

    def __init__(self, nc_measure, mondrian=False):
        """Verify that the nonconformity measure can be used for classification."""
        assert isinstance(nc_measure, ClassNC), "Inappropriate nonconformity measure for classification"
        self.nc_measure = nc_measure
        self.mondrian = mondrian

    def __str__(self):
        return "{} ({})".format(self.__class__.__name__, self.nc_measure)

    def p_values(self, example):
        """Extending classes should implement this method to return a list of pairs (p-value, class)
        for a given example.

        Conformal classifier assigns an assumed class value to the given example and computes its nonconformity.
        P-value is the ratio of more nonconformal (stranger) instances that the given example.
        """
        raise NotImplementedError

    def predict(self, example, eps=None):
        """Compute a classification prediction object from p-values for a given example and significance level.

        Args:
            example (Instance): Orange row instance.
            eps (float): Default significance level (error rate).

        Returns:
            PredictionClass: Classification prediction object.
        """
        ps = self.p_values(example)
        return PredictionClass(ps, eps)

    def __call__(self, example, eps):
        """Compute predicted classes for a given example and significance level.

        Args:
            example (Instance): Orange row instance.
            eps (float): Significance level (error rate).

        Returns:
            List of predicted classes.
        """
        pred = self.predict(example)
        return pred.classes(eps)


class TransductiveClassifier(ConformalClassifier):
    """Transductive classification.

    Examples:
        >>> train, test = next(LOOSampler(Table('iris')))
        >>> tcp = TransductiveClassifier(ProbabilityMargin(NaiveBayesLearner()), train)
        >>> print(tcp(test[0], 0.1))
    """

    def __init__(self, nc_measure, train=None, mondrian=False):
        """Initialize transductive classifier with a nonconformity measure and a training set.

        Fit the conformal classifier to the training set if present.

        Args:
            nc_measure (ClassNC): Classification nonconformity measure.
            train (Optional[Table]): Table of examples used as a training set.
            mondrian (bool): Use a mondrian setting for computing p-values.
        """
        super().__init__(nc_measure, mondrian=mondrian)
        if train is not None:
            self.fit(train)

    def fit(self, train):
        """Fit the conformal classifier to the training set and store the domain.

        Args:
            train (Optional[Table]): Table of examples used as a training set.
        """
        self.train = train
        self.domain = self.train.domain

    def p_values(self, example):
        """Compute p-values for every possible class.

        Transductive classifier appends the given example with an assumed class value to the training set
        and compares its nonconformity against all other instances.

        Args:
            example (Instance): Orange row instance.

        Returns:
            List of pairs (p-value, class)
        """
        ps = []
        temp = example.get_class()
        for yi, y in enumerate(self.domain.class_var.values):
            example.set_class(yi)
            data = Table(self.domain, np.vstack((self.train, example)))
            self.nc_measure.fit(data)
            scores = np.array([self.nc_measure.nonconformity(row) for row in data
                               if not self.mondrian or self.mondrian and row.get_class() == y])
            alpha, alpha_n = scores[:-1], scores[-1]
            p_y = sum(scores >= alpha_n) / len(scores)
            ps.append((p_y, y))
        example.set_class(temp)
        return ps


class InductiveClassifier(ConformalClassifier):
    """Inductive classification.

    Attributes:
        alpha: Nonconformity scores of the calibration instances. Computed by the :py:func:`fit` method.

    Examples:
        >>> train, test = next(LOOSampler(Table('iris')))
        >>> train, calibrate = next(RandomSampler(train, 2, 1))
        >>> icp = InductiveClassifier(InverseProbability(LogisticRegressionLearner()), train, calibrate)
        >>> print(icp(test[0], 0.1))
    """

    def __init__(self, nc_measure, train=None, calibrate=None, mondrian=False):
        """Initialize inductive classifier with a nonconformity measure, training set and calibration set.
        If present, fit the conformal classifier to the training set and compute the nonconformity scores of
        calibration set.

        Args:
            nc_measure (ClassNC): Classification nonconformity measure.
            train (Optional[Table]): Table of examples used as a training set.
            calibrate (Optional[Table]): Table of examples used as a calibration set.
            mondrian (bool): Use a mondrian setting for computing p-values.
        """
        super().__init__(nc_measure, mondrian=mondrian)
        if train is not None and calibrate is not None:
            self.fit(train, calibrate)

    def fit(self, train, calibrate):
        """Fit the conformal classifier to the training set, compute and store nonconformity scores (:py:attr:`alpha`)
        on the calibration set and store the domain.

        Args:
            train (Optional[Table]): Table of examples used as a training set.
            calibrate (Optional[Table]): Table of examples used as a calibration set.
        """
        self.domain = train.domain
        self.calibrate = calibrate
        self.nc_measure.fit(train)
        self.alpha = np.array([self.nc_measure.nonconformity(inst) for inst in calibrate])

    def p_values(self, example):
        """Compute p-values for every possible class.

        Inductive classifier assigns an assumed class value to the given example and compares its nonconformity
        against all other instances in the calibration set.

        Args:
            example (Instance): Orange row instance.

        Returns:
            List of pairs (p-value, class)
        """
        classes = []
        ps = []
        temp = example.get_class()
        for yi, y in enumerate(self.domain.class_var.values):
            example.set_class(yi)
            alpha_n = self.nc_measure.nonconformity(example)
            if self.mondrian:
                alpha = np.array([a for a, cal in zip(self.alpha, self.calibrate) if cal.get_class() == y])
            else:
                alpha = self.alpha
            p_y = (sum(alpha >= alpha_n)+1) / (len(alpha)+1)
            ps.append((p_y, y))
        example.set_class(temp)
        return ps


class CrossClassifier(InductiveClassifier):
    """Cross classification.

    Examples:
        >>> train, test = next(LOOSampler(Table('iris')))
        >>> ccp = CrossClassifier(InverseProbability(LogisticRegressionLearner()), 3, train)
        >>> print(ccp(test[0], 0.1))
    """

    def __init__(self, nc_measure, k, train=None, mondrian=False):
        """Initialize cross classifier with a nonconformity measure, number of folds and training set.
        If present, fit the conformal classifier to the training set.

        Args:
            nc_measure (ClassNC): Classification nonconformity measure.
            k (int): Number of folds.
            train (Optional[Table]): Table of examples used as a training set.
            mondrian (bool): Use a mondrian setting for computing p-values.
        """
        # store the unfitted nonconformity measure for making copies to fit on individual folds
        super().__init__(nc_measure, mondrian=mondrian)
        self.nc_measure_base = deepcopy(self.nc_measure)
        self.k = k
        if train is not None:
            self.fit(train)

    def fit(self, train):
        """Fit the cross classifier to the training set. Split the training set into k folds for use as
        training and calibration set with an inductive classifier. Concatenate the computed nonconformity scores
        and store them (:py:attr:`InductiveClassifier.alpha`).

        Args:
            train (Table): Table of examples used as a training set.
        """
        self.domain = train.domain
        self.calibrate = train
        self.nc_measure.fit(train)
        self.alpha = np.array([])
        for train_index, calibrate_index in KFold(self.k, shuffle=True).split(train):
            icp = InductiveClassifier(deepcopy(self.nc_measure_base), train[train_index], train[calibrate_index])
            self.alpha = np.concatenate((self.alpha, icp.alpha))


class LOOClassifier(CrossClassifier):
    """Leave-one-out classifier is a cross conformal classifier with the number of folds equal
    to the size of the training set.

    Examples:
        >>> train, test = next(LOOSampler(Table('iris')))
        >>> loocp = LOOClassifier(InverseProbability(LogisticRegressionLearner()), train)
        >>> print(loocp(test[0], 0.1))
    """

    def __init__(self, nc_measure, train=None, mondrian=False):
        super().__init__(nc_measure, 0, train, mondrian)

    def fit(self, train):
        self.k = len(train)
        super().fit(train)
