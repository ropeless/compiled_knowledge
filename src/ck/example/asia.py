from ck.pgm import PGM


class Asia(PGM):
    """
    This PGM is the well known, pedagogical 'Asia' Bayesian network.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)

        asia = self.new_rv('asia', ('yes', 'no'))
        tub = self.new_rv('tub', ('yes', 'no'))
        smoke = self.new_rv('smoke', ('yes', 'no'))
        lung = self.new_rv('lung', ('yes', 'no'))
        bronc = self.new_rv('bronc', ('yes', 'no'))
        either = self.new_rv('either', ('yes', 'no'))
        xray = self.new_rv('xray', ('yes', 'no'))
        dysp = self.new_rv('dysp', ('yes', 'no'))

        self.new_factor(asia).set_dense().set_flat(0.01, 0.99)
        self.new_factor(tub, asia).set_dense().set_flat(0.05, 0.01, 0.95, 0.99)
        self.new_factor(smoke).set_dense().set_flat(0.5, 0.5)
        self.new_factor(lung, smoke).set_dense().set_flat(0.1, 0.01, 0.9, 0.99)
        self.new_factor(bronc, smoke).set_dense().set_flat(0.6, 0.3, 0.4, 0.7)
        self.new_factor(either, lung, tub).set_dense().set_flat(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        self.new_factor(xray, either).set_dense().set_flat(0.98, 0.05, 0.02, 0.95)
        self.new_factor(dysp, bronc, either).set_dense().set_flat(0.9, 0.8, 0.7, 0.1, 0.1, 0.2, 0.3, 0.9)
