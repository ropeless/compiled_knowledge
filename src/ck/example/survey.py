from ck.pgm import PGM


class Survey(PGM):

    def __init__(self):
        super().__init__(self.__class__.__name__)

        pgm_rv0 = self.new_rv('A', ('young', 'adult', 'old'))
        pgm_rv1 = self.new_rv('S', ('M', 'F'))
        pgm_rv2 = self.new_rv('E', ('high', 'uni'))
        pgm_rv3 = self.new_rv('O', ('emp', 'self'))
        pgm_rv4 = self.new_rv('R', ('small', 'big'))
        pgm_rv5 = self.new_rv('T', ('car', 'train', 'other'))
        pgm_factor0 = self.new_factor(pgm_rv0)
        pgm_factor1 = self.new_factor(pgm_rv1)
        pgm_factor2 = self.new_factor(pgm_rv2, pgm_rv0, pgm_rv1)
        pgm_factor3 = self.new_factor(pgm_rv3, pgm_rv2)
        pgm_factor4 = self.new_factor(pgm_rv4, pgm_rv2)
        pgm_factor5 = self.new_factor(pgm_rv5, pgm_rv3, pgm_rv4)

        pgm_function0 = pgm_factor0.set_dense()
        pgm_function0.set_flat(0.3, 0.5, 0.2)

        pgm_function1 = pgm_factor1.set_dense()
        pgm_function1.set_flat(0.6, 0.4)

        pgm_function2 = pgm_factor2.set_dense()
        pgm_function2.set_flat(
            0.75, 0.64, 0.72, 0.7, 0.88,
            0.9, 0.25, 0.36, 0.28, 0.3,
            0.12, 0.1
        )

        pgm_function3 = pgm_factor3.set_dense()
        pgm_function3.set_flat(0.96, 0.92, 0.04, 0.08)

        pgm_function4 = pgm_factor4.set_dense()
        pgm_function4.set_flat(0.25, 0.2, 0.75, 0.8)

        pgm_function5 = pgm_factor5.set_dense()
        pgm_function5.set_flat(
            0.48, 0.58, 0.56, 0.7, 0.42,
            0.24, 0.36, 0.21, 0.1, 0.18,
            0.08, 0.09
        )
