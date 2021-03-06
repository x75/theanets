import theanets

import util


class TestNetwork(util.Base):
    def setUp(self):
        self.exp = theanets.Regressor([self.NUM_INPUTS, 10, self.NUM_OUTPUTS])

    def assert_progress(self, **kwargs):
        train0, valid0 = next(self.exp.itertrain([self.INPUTS, self.OUTPUTS]))
        trainN, validN = self.exp.train(
            [self.INPUTS, self.OUTPUTS],
            algorithm='sgd',
            patience=2,
            min_improvement=0.01,
            batch_size=self.NUM_EXAMPLES,
            **kwargs)
        assert trainN['loss'] < valid0['loss']   # should have made progress!

    def test_input_noise(self):
        self.assert_progress(input_noise=0.001)

    def test_input_dropout(self):
        self.assert_progress(input_dropout=0.1)

    def test_hidden_noise(self):
        self.assert_progress(hidden_noise=0.001)

    def test_hidden_dropout(self):
        self.assert_progress(hidden_dropout=0.1)

    def test_noise(self):
        self.assert_progress(noise={'*:out': 0.001})

    def test_dropout(self):
        self.assert_progress(dropout={'*:out': 0.1})

    def test_hidden_l1(self):
        self.assert_progress(hidden_l1=0.001)

    def test_weight_l1(self):
        self.assert_progress(weight_l1=0.001)

    def test_weight_l2(self):
        self.assert_progress(weight_l2=0.001)

    def test_contractive(self):
        self.assert_progress(contractive=0.001)
