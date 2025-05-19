from ck.pgm import PGM


class Sprinkler(PGM):
    """
    This PGM is the well known, pedagogical 'Sprinkler' Bayesian network.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)

        rain = self.new_rv('rain', ['not raining', 'raining'])
        sprinkler = self.new_rv('sprinkler', ['off', 'on'])
        grass = self.new_rv('grass', ['dry', 'damp', 'wet'])

        f_g = self.new_factor(grass, rain, sprinkler)
        f_r = self.new_factor(rain)
        f_s = self.new_factor(sprinkler)

        f_r.set_dense().set_flat(0.8, 0.2)
        f_s.set_dense().set_flat(0.9, 0.1)
        # fmt: off
        f_g.set_dense().set_flat(
            # not raining, raining   # rain
            # off, on,     off,  on  # sprinkler
            0.90, 0.01, 0.02, 0.01,  # grass dry
            0.09, 0.01, 0.08, 0.04,  # grass damp
            0.01, 0.98, 0.90, 0.95,  # grass wet
        )
        # fmt: on
