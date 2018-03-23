"""Regression module contains methods for conformal regression.

Conformal regressors predict a range of values (not always a single value) under a given
significance level (error rate). Every regressors works in combination with a nonconformity measure
and on average predicts the correct value with the given error rate. Lower error rates result in
narrower ranges of predicted values.

Structure:

- ConformalRegressor
    - Inductive (:py:class:`InductiveRegressor`)
    - Cross (:py:class:`CrossRegressor`)
"""

from copy import deepcopy

import numpy as np
from sklearn.model_selection import KFold

from Orange.data import Instance, Unknown

from orangecontrib.conformal.base import ConformalPredictor
from orangecontrib.conformal.nonconformity import RegrNC


class PredictionRegr:
    """Conformal regression prediction object,
    which is produced by the :py:func:`ConformalRegressor.predict` method.

    Attributes:
        lo (float): Lowest value of the predicted range.
        hi (float): Highest value of the predicted range.

    Examples:
        >>> train, test = next(LOOSampler(Table('housing')))
        >>> ccr = CrossRegressor(AbsError(LinearRegressionLearner()), 5, train)
        >>> prediction = ccr.predict(test[0], 0.1)
        >>> print(prediction.width())
    """

    def __init__(self, lo, hi):
        """Initialize the prediction.

        Args:
            lo (float): Lowest value of the predicted range.
            hi (float): Highest value of the predicted range.
        """
        self.lo = lo
        self.hi = hi

    def range(self):
        """Predicted range: :py:attr:`lo`, :py:attr:`hi`."""
        return self.lo, self.hi

    def verdict(self, ref):
        """Conformal regression prediction is correct when the actual value appears
        in the predicted range.

        Args:
            ref: Reference/actual value

        Returns:
            True if the prediction is correct.
        """
        return self.lo <= ref <= self.hi

    def width(self):
        """Width of the predicted range: :py:attr:`hi` - :py:attr:`lo`."""
        if np.isnan(self.lo) or np.isnan(self.hi):
            return 0
        else:
            return self.hi-self.lo


class ConformalRegressor(ConformalPredictor):
    """Base class for conformal regression."""

    def __init__(self, nc_measure):
        """Verify that the nonconformity measure can be used for regression."""
        assert isinstance(nc_measure, RegrNC), "Inappropriate nonconformity measure for regression"
        self.nc_measure = nc_measure

    def __str__(self):
        return "{} ({})".format(self.__class__.__name__, self.nc_measure)

    def predict(self, example, eps):
        """Compute a regression prediction object for a given example and significance level.

        Function determines what is the :py:attr:`eps`-th lowest nonconformity score and computes
        the range of values that would result in a lower or equal nonconformity. This inverse
        of the nonconformity score is computed by the nonconformity measure's
        :py:func:`cp.nonconformity.RegrNC.predict` function.

        Args:
            example (Instance): Orange row instance.
            eps (float): Default significance level (error rate).

        Returns:
            PredictionRegr: Regression prediction object.
        """
        s = int(eps*(len(self.alpha)+1)) - 1
        s = min(max(s, 0), len(self.alpha)-1)
        nc = self.alpha[s]
        lo, hi = self.nc_measure.predict(example, nc)
        return PredictionRegr(lo, hi)

    def __call__(self, example, eps):
        """Compute predicted range for a given example and significance level.

        Args:
            example (Instance): Orange row instance.
            eps (float): Significance level (error rate).

        Returns:
            Predicted range as a pair (`PredictionRegr.lo`, `PredictionRegr.hi`)
        """
        pred = self.predict(example, eps)
        return pred.range()


class TransductiveRegressor(ConformalRegressor):
    """Transductive regression. TODO
    """
    pass


class InductiveRegressor(ConformalRegressor):
    """Inductive regression.

    Attributes:
        alpha: Nonconformity scores of the calibration instances. Computed by the :py:func:`fit` method.
            Must be *sorted* in increasing order.

    Examples:
        >>> train, test = next(LOOSampler(Table('housing')))
        >>> train, calibrate = next(RandomSampler(train, 2, 1))
        >>> icr = InductiveRegressor(AbsError(LinearRegressionLearner()), train, calibrate)
        >>> print(icr(test[0], 0.1))
    """

    def __init__(self, nc_measure, train=None, calibrate=None):
        """Initialize inductive regressor with a nonconformity measure, training set and calibration set.
        If present, fit the conformal regressor to the training set and compute the nonconformity scores of
        calibration set.

        Args:
            nc_measure (RegrNC): Regression nonconformity measure.
            train (Optional[Table]): Table of examples used as a training set.
            calibrate (Optional[Table]): Table of examples used as a calibration set.
        """
        super().__init__(nc_measure)
        if train is not None and calibrate is not None:
            self.fit(train, calibrate)

    def fit(self, train, calibrate):
        """Fit the conformal regressor to the training set, compute and store sorted nonconformity scores (:py:attr:`alpha`)
        on the calibration set and store the domain.

        Args:
            train (Optional[Table]): Table of examples used as a training set.
            calibrate (Optional[Table]): Table of examples used as a calibration set.
        """
        self.domain = train.domain
        self.nc_measure.fit(train)
        self.alpha = [self.nc_measure.nonconformity(inst) for inst in calibrate]
        self.alpha = np.array(sorted(self.alpha, reverse=True))


class CrossRegressor(InductiveRegressor):
    """Cross regression.

    Examples:
        >>> train, test = next(LOOSampler(Table('housing')))
        >>> ccr = CrossRegressor(AbsError(LinearRegressionLearner()), 4, train)
        >>> print(ccr(test[0], 0.1))
    """

    def __init__(self, nc_measure, k, train=None):
        """Initialize cross regressor with a nonconformity measure, number of folds and training set.
        If present, fit the conformal regressor to the training set.

        Args:
            nc_measure (RegrNC): Regression nonconformity measure.
            k (int): Number of folds.
            train (Optional[Table]): Table of examples used as a training set.
        """
        # store the unfitted nonconformity measure for making copies to fit on individual folds
        super().__init__(nc_measure)
        self.nc_measure_base = deepcopy(self.nc_measure)
        self.k = k
        if train is not None:
            self.fit(train)

    def fit(self, train):
        """Fit the cross regressor to the training set. Split the training set into k folds for use as
        training and calibration set with an inductive regressor. Concatenate the computed nonconformity scores
        and store them (:py:attr:`InductiveRegressor.alpha`).

        Args:
            train (Table): Table of examples used as a training set.
        """
        self.domain = train.domain
        self.nc_measure.fit(train)
        self.alpha = np.array([])
        for train_index, calibrate_index in KFold(self.k, shuffle=True).split(train):
            icr = InductiveRegressor(deepcopy(self.nc_measure_base), train[train_index], train[calibrate_index])
            self.alpha = np.concatenate((self.alpha, icr.alpha))
        self.alpha = np.array(sorted(self.alpha, reverse=True))


class LOORegressor(CrossRegressor):
    """Leave-one-out regressor is a cross conformal regressor with the number of folds equal
    to the size of the training set.

    Examples:
        >>> train, test = next(LOOSampler(Table('housing')))
        >>> ccr = LOORegressor(AbsError(LinearRegressionLearner()), train)
        >>> print(ccr(test[0], 0.1))
    """

    def __init__(self, nc_measure, train=None):
        super().__init__(nc_measure, 0, train)

    def fit(self, train):
        self.k = len(train)
        super().fit(train)
