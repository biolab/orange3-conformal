from unittest import TestCase

import Orange

import orangecontrib.conformal as cp


class TestClassification(TestCase):
    def setUp(self):
        self.tab = Orange.data.Table('iris')
        self.inst, self.tab = self.tab[0], self.tab[1:]
        self.train, self.test = self.tab[0::2], self.tab[1::2]
        self.train, self.calibrate = self.train[0::2], self.train[1::2]
        self.nc = cp.nonconformity.InverseProbability(Orange.classification.NaiveBayesLearner())

    def test_transductive(self):
        predictor = cp.classification.TransductiveClassifier(self.nc, self.train)
        self.assertListEqual(predictor(self.inst.x, 0.1), ['Iris-setosa'])

    def test_inductive(self):
        predictor = cp.classification.InductiveClassifier(self.nc, self.train, self.calibrate)
        self.assertListEqual(predictor(self.inst.x, 0.1), ['Iris-setosa'])

    def test_cross(self):
        predictor = cp.classification.CrossClassifier(self.nc, 4, self.train)
        self.assertListEqual(predictor(self.inst.x, 0.1), ['Iris-setosa'])

    def test_loo(self):
        predictor = cp.classification.LOOClassifier(self.nc, self.train)
        self.assertListEqual(predictor(self.inst.x, 0.1), ['Iris-setosa'])

    def test_nc_type(self):
        nc_class = cp.nonconformity.InverseProbability(Orange.classification.LogisticRegressionLearner())
        nc_regr = cp.nonconformity.AbsError(Orange.regression.LinearRegressionLearner())
        cp.classification.TransductiveClassifier(nc_class)
        self.assertRaises(AssertionError, cp.classification.TransductiveClassifier, nc_regr)
        cp.classification.InductiveClassifier(nc_class)
        self.assertRaises(AssertionError, cp.classification.InductiveClassifier, nc_regr)
        cp.classification.CrossClassifier(nc_class, 5)
        self.assertRaises(AssertionError, cp.classification.CrossClassifier, nc_regr, 5)


class TestRegression(TestCase):
    def setUp(self):
        self.tab = Orange.data.Table('housing')
        self.inst, self.tab = self.tab[1], self.tab[2:]
        self.train, self.test = self.tab[0::2], self.tab[1::2]
        self.train, self.calibrate = self.train[0::2], self.train[1::2]
        self.nc = cp.nonconformity.AbsError(Orange.regression.LinearRegressionLearner())

    def test_inductive(self):
        predictor = cp.regression.InductiveRegressor(self.nc, self.train, self.calibrate)
        lo, hi = predictor(self.inst.x, 0.2)
        self.assertTrue(lo <= float(self.inst.y) <= hi)

    def test_cross(self):
        predictor = cp.regression.CrossRegressor(self.nc, 4, self.train)
        lo, hi = predictor(self.inst.x, 0.2)
        self.assertTrue(lo <= float(self.inst.y) <= hi)

    def test_loo(self):
        predictor = cp.regression.LOORegressor(self.nc, self.train)
        lo, hi = predictor(self.inst.x, 0.2)
        self.assertTrue(lo <= float(self.inst.y) <= hi)

    def test_nc_type(self):
        nc_class = cp.nonconformity.InverseProbability(Orange.classification.LogisticRegressionLearner())
        nc_regr = cp.nonconformity.AbsError(Orange.regression.LinearRegressionLearner())
        cp.regression.InductiveRegressor(nc_regr)
        self.assertRaises(AssertionError, cp.regression.InductiveRegressor, nc_class)
        cp.regression.CrossRegressor(nc_regr, 5)
        self.assertRaises(AssertionError, cp.regression.CrossRegressor, nc_class, 5)


class TestMondrian(TestCase):
    def setUp(self):
        self.tab = Orange.data.Table('iris')
        self.inst, self.tab = self.tab[0], self.tab[1:]
        self.train, self.test = self.tab[0::2], self.tab[1::2]
        self.train, self.calibrate = self.train[0::2], self.train[1::2]
        self.nc = cp.nonconformity.InverseProbability(Orange.classification.NaiveBayesLearner())

    def test_transductive(self):
        predictor = cp.classification.TransductiveClassifier(self.nc, self.train, mondrian=True)
        self.assertListEqual(predictor(self.inst.x, 0.1), ['Iris-setosa'])

    def test_inductive(self):
        predictor = cp.classification.InductiveClassifier(self.nc, self.train, self.calibrate, mondrian=True)
        self.assertListEqual(predictor(self.inst.x, 0.1), ['Iris-setosa'])

    def test_cross(self):
        predictor = cp.classification.CrossClassifier(self.nc, 7, self.train, mondrian=True)
        self.assertListEqual(predictor(self.inst.x, 0.1), ['Iris-setosa'])

    def test_loo(self):
        predictor = cp.classification.LOOClassifier(self.nc, self.train, mondrian=True)
        self.assertListEqual(predictor(self.inst.x, 0.1), ['Iris-setosa'])


class TestEfficiency(TestCase):
    def test_individual_classification(self):
        tab = Orange.data.Table('iris')
        train, test = tab[:123], tab[123]  # borderline case
        nc = cp.nonconformity.InverseProbability(Orange.classification.LogisticRegressionLearner())
        pred = cp.classification.CrossClassifier(nc, 3, train)
        p = pred.predict(test.x)
        cred, conf = p.credibility(), p.confidence()
        self.assertLess(cred, 0.5)
        d = 1e-6
        self.assertEqual(len(pred(test.x, 1-(conf-d))), 1)
        self.assertGreater(len(pred(test.x, 1-(conf+d))), 1)
        self.assertEqual(len(pred(test.x, (cred+d))), 0)
        self.assertGreater(len(pred(test.x, (cred-d))), 0)
