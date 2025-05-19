from ck.pgm import PGM


class Stress(PGM):
    """
    This PGM is the 'Stress' factor graph.

    The Stress factor graph uses many of the different PGM software
    components. This can be used to stress test the PGM implementation
    and processing pipelines.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)

        no_name = self.new_rv('', 2)
        x1 = self.new_rv('x1', 2)
        x2 = self.new_rv('x2', ['one', 'two'])
        x3 = self.new_rv('x3', ['one', 'two', 'non-ascii 😊'])
        x4 = self.new_rv('x4', [1.1, 1.23, None, 'string'])
        x5 = self.new_rv('x5', 3)
        A = self.new_rv('A', 2)
        B = self.new_rv('B', 2)
        C = self.new_rv('C', 2)
        D = self.new_rv('D', 2)
        E = self.new_rv('E', [True, False])

        self.new_factor(no_name).set_zero()

        self.new_factor(x1).set_dense().set_flat(1, 2)

        self.new_factor(x1, x2).set_clause(1, 0)

        self.new_factor(x2, x3).set_cpt().set_all(
            (0.1, 0.9),
            (0.2, 0.8),
            (0.3, 0.7),
        )

        f_3_4 = self.new_factor(x3, x4).set_sparse()
        f_3_4[0, 0] = 11
        f_3_4[1, 1] = 11
        f_3_4[2, 2] = 22

        f_4_5 = self.new_factor(x4, x5).set_compact()
        f_4_5[0, 0] = 11
        f_4_5[1, 1] = 11
        f_4_5[2, 2] = 22

        f_ab = self.new_factor(A, B)
        f_ab.set_dense().set_flat(2, 3, 5, 7)

        self.new_factor(A, C).set_sparse()[0, 1] = 11
        self.new_factor(A, D).set_sparse()[1, 0] = 13
        self.new_factor(B, C).set_clause(0, 1)
        self.new_factor(B, D).set_cpt().set_all([0.2, 0.8], [0.3, 0.7])

        self.new_factor(A, E)  # leave with no potential function

        self.new_factor(B, E).function = f_ab.function  # shared function

        # Two factors on the same random variables
        self.new_factor(C, E).set_sparse()
        self.new_factor(C, E).set_sparse()
