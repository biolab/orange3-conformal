from unittest import TestCase

import Orange

import orangecontrib.conformal as cp


class TestReadme(TestCase):
    def setUp(self):
        self.tab = Orange.data.Table('iris')
        self.train, self.test = next(cp.evaluation.RandomSampler(self.tab, 2, 1))
        self.train, self.calibrate = next(cp.evaluation.RandomSampler(self.train, 2, 1))

    def test_readme(self):
        nc = cp.nonconformity.InverseProbability(Orange.classification.NaiveBayesLearner())
        ic = cp.classification.InductiveClassifier(nc)
        r = cp.evaluation.run(ic, 0.1, cp.evaluation.RandomSampler(self.tab, 2, 1))
        self.assertGreater(r.accuracy(), 0.8)
