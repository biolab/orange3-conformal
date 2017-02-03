from unittest import TestCase

import Orange

import orangecontrib.conformal as cp


class TestClassification(TestCase):
    def setUp(self):
        self.tab = Orange.data.Table('iris')
        self.train, self.test = next(cp.evaluation.RandomSampler(self.tab, 2, 1))
        self.train, self.calibrate = next(cp.evaluation.RandomSampler(self.train, 2, 1))

    def test_readme(self):
        nc = cp.nonconformity.InverseProbability(Orange.classification.NaiveBayesLearner())
        ic = cp.classification.InductiveClassifier(nc)
        r = cp.evaluation.run(ic, 0.1, cp.evaluation.RandomSampler(self.tab, 2, 1))
        print(r.accuracy())

    def test_transductive(self):
        pass

    def test_inductive(self):
        pass

    def test_cross(self):
        pass

    def test_loo(self):
        pass

    def test_nonconformity(self):
        pass

class TestRegression(TestCase):
    def setUp(self):
        self.tab = Orange.data.Table('housing')
        self.train, self.test = next(cp.evaluation.RandomSampler(self.tab, 2, 1))
        self.train, self.calibrate = next(cp.evaluation.RandomSampler(self.train, 2, 1))

    def test_transductive(self):
        pass

    def test_inductive(self):
        pass

    def test_cross(self):
        pass

    def test_loo(self):
        pass

    def test_nonconformity(self):
        pass
