from ck.pgm import PGM


class Earthquake(PGM):

    def __init__(self):
        super().__init__(self.__class__.__name__)

        pgm_rv0 = self.new_rv('Burglary', ('True', 'False'))
        pgm_rv1 = self.new_rv('Earthquake', ('True', 'False'))
        pgm_rv2 = self.new_rv('Alarm', ('True', 'False'))
        pgm_rv3 = self.new_rv('JohnCalls', ('True', 'False'))
        pgm_rv4 = self.new_rv('MaryCalls', ('True', 'False'))
        pgm_factor0 = self.new_factor(pgm_rv0)
        pgm_factor1 = self.new_factor(pgm_rv1)
        pgm_factor2 = self.new_factor(pgm_rv2, pgm_rv0, pgm_rv1)
        pgm_factor3 = self.new_factor(pgm_rv3, pgm_rv2)
        pgm_factor4 = self.new_factor(pgm_rv4, pgm_rv2)

        pgm_function0 = pgm_factor0.set_dense()
        pgm_function0.set_flat(0.01, 0.99)

        pgm_function1 = pgm_factor1.set_dense()
        pgm_function1.set_flat(0.02, 0.98)

        pgm_function2 = pgm_factor2.set_dense()
        pgm_function2.set_flat(
            0.95, 0.94, 0.29, 0.001, 0.05,
            0.06, 0.71, 0.999
        )

        pgm_function3 = pgm_factor3.set_dense()
        pgm_function3.set_flat(0.9, 0.05, 0.1, 0.95)

        pgm_function4 = pgm_factor4.set_dense()
        pgm_function4.set_flat(0.7, 0.01, 0.3, 0.99)
