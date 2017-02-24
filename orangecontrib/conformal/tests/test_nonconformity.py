from unittest import TestCase

import sklearn

import Orange

import orangecontrib.conformal as cp


class TestClassification(TestCase):
    def setUp(self):
        self.tab = Orange.data.Table('iris')
        self.inst, self.tab = self.tab[0], self.tab[1:]
        self.train, self.calibrate = self.tab[0::2], self.tab[1::2]

    def test_Models(self):
        nc = cp.nonconformity.InverseProbability(Orange.classification.LogisticRegressionLearner())
        predictor = cp.classification.InductiveClassifier(nc, self.train, self.calibrate)
        self.assertListEqual(predictor(self.inst.x, 0.1), ['Iris-setosa'])

        nc = cp.nonconformity.ProbabilityMargin(Orange.classification.LogisticRegressionLearner())
        predictor = cp.classification.InductiveClassifier(nc, self.train, self.calibrate)
        self.assertListEqual(predictor(self.inst.x, 0.1), ['Iris-setosa'])

    def test_SVM(self):
        tab = Orange.data.Table('titanic')
        inst, tab = tab[0], tab[1:]
        train, calibrate = tab[:1500], tab[1500:]

        nc = cp.nonconformity.SVMDistance(sklearn.svm.SVC())
        predictor = cp.classification.InductiveClassifier(nc, train, calibrate)
        self.assertListEqual(predictor(inst.x, 0.1), ['no'])

    def test_Neighbours(self):
        nc = cp.nonconformity.KNNDistance(Orange.distance.Euclidean, 10)
        predictor = cp.classification.InductiveClassifier(nc, self.train, self.calibrate)
        self.assertListEqual(predictor(self.inst.x, 0.1), ['Iris-setosa'])

        nc = cp.nonconformity.KNNFraction(Orange.distance.Euclidean, 10, weighted=True)
        predictor = cp.classification.InductiveClassifier(nc, self.train, self.calibrate)
        self.assertListEqual(predictor(self.inst.x, 0.1), ['Iris-setosa'])

    def test_Mahalanobis(self):
        nc = cp.nonconformity.KNNDistance(Orange.distance.Mahalanobis, 5)
        predictor = cp.classification.InductiveClassifier(nc, self.train, self.calibrate)
        self.assertListEqual(predictor(self.inst.x, 0.1), ['Iris-setosa'])

        Mah = Orange.distance.MahalanobisDistance(self.train)
        nc = cp.nonconformity.KNNDistance(Mah, 5)
        predictor = cp.classification.InductiveClassifier(nc, self.train, self.calibrate)
        self.assertListEqual(predictor(self.inst.x, 0.1), ['Iris-setosa'])


class TestRegression(TestCase):
    def setUp(self):
        self.tab = Orange.data.Table('housing')[:100]
        self.inst, self.tab = self.tab[1], self.tab[2:]
        self.train, self.calibrate = self.tab[:70], self.tab[70:]

    def test_Models(self):
        nc = cp.nonconformity.AbsError(Orange.regression.LinearRegressionLearner())
        predictor = cp.regression.InductiveRegressor(nc, self.train, self.calibrate)
        lo, hi = predictor(self.inst.x, 0.1)
        self.assertTrue(lo <= float(self.inst.y) <= hi)

        nc = cp.nonconformity.AbsErrorNormalized(Orange.regression.LinearRegressionLearner(),
                                                 Orange.distance.Euclidean, 10)
        predictor = cp.regression.InductiveRegressor(nc, self.train, self.calibrate)
        lo, hi = predictor(self.inst.x, 0.1)
        self.assertTrue(lo <= float(self.inst.y) <= hi)

    def test_RandomForest(self):
        nc = cp.nonconformity.AbsErrorRF(Orange.regression.RandomForestRegressionLearner(),
                                         sklearn.ensemble.RandomForestRegressor())
        predictor = cp.regression.InductiveRegressor(nc, self.train, self.calibrate)
        lo, hi = predictor(self.inst.x, 0.1)
        self.assertTrue(lo <= float(self.inst.y) <= hi)

    def test_ErrorModel(self):
        nc = cp.nonconformity.ErrorModelNC(Orange.regression.SVRLearner(),
                                           Orange.regression.LinearRegressionLearner())
        predictor = cp.regression.InductiveRegressor(nc, self.train, self.calibrate)
        lo, hi = predictor(self.inst.x, 0.1)
        self.assertTrue(lo <= float(self.inst.y) <= hi)

    def test_Neighbours(self):
        nc = cp.nonconformity.AbsErrorKNN(Orange.distance.Euclidean, 10)
        predictor = cp.regression.InductiveRegressor(nc, self.train, self.calibrate)
        lo, hi = predictor(self.inst.x, 0.1)
        self.assertTrue(lo <= float(self.inst.y) <= hi)

        nc = cp.nonconformity.AvgErrorKNN(Orange.distance.Euclidean, 10)
        predictor = cp.regression.InductiveRegressor(nc, self.train, self.calibrate)
        lo, hi = predictor(self.inst.x, 0.1)
        self.assertTrue(lo <= float(self.inst.y) <= hi)
