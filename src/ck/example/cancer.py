from ck.pgm import PGM


class Cancer(PGM):
    """
    This PGM is the well known, pedagogical 'Cancer' Bayesian network.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)

        pollution = self.new_rv('pollution', ('low', 'high'))
        smoker = self.new_rv('smoker', ('True', 'False'))
        cancer = self.new_rv('cancer', ('True', 'False'))
        xray = self.new_rv('xray', ('positive', 'negative'))
        dyspnoea = self.new_rv('dyspnoea', ('True', 'False'))

        pgm_factor0 = self.new_factor(pollution)
        pgm_function_2511325233376 = pgm_factor0.set_cpt()
        pgm_function_2511325233376.set_cpd((), (0.9, 0.1))
        pgm_factor1 = self.new_factor(smoker)
        pgm_function_2511322114176 = pgm_factor1.set_cpt()
        pgm_function_2511322114176.set_cpd((), (0.3, 0.7))
        pgm_factor2 = self.new_factor(cancer, pollution, smoker)
        pgm_function_2511324995136 = pgm_factor2.set_cpt()
        pgm_function_2511324995136.set_cpd((0, 0), (0.03, 0.97))
        pgm_function_2511324995136.set_cpd((1, 0), (0.05, 0.95))
        pgm_function_2511324995136.set_cpd((0, 1), (0.001, 0.999))
        pgm_function_2511324995136.set_cpd((1, 1), (0.02, 0.98))
        pgm_factor3 = self.new_factor(xray, cancer)
        pgm_function_2511331552432 = pgm_factor3.set_cpt()
        pgm_function_2511331552432.set_cpd((0,), (0.9, 0.1))
        pgm_function_2511331552432.set_cpd((1,), (0.2, 0.8))
        pgm_factor4 = self.new_factor(dyspnoea, cancer)
        pgm_function_2511325139344 = pgm_factor4.set_cpt()
        pgm_function_2511325139344.set_cpd((0,), (0.65, 0.35))
        pgm_function_2511325139344.set_cpd((1,), (0.3, 0.7))
