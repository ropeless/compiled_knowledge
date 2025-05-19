from ck.pgm import PGM


class Student(PGM):
    """
    This PGM is the well known, pedagogical 'Student' Bayesian network.

    Reference:
    Probabilistic Graphical Models: Principles and Techniques
    Daphne Koller & Nir Friedman. MIT Press, 2009.
    Figure 3.1, page 48.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)

        difficult = self.new_rv('difficult', ('Yes', 'No'))
        intelligent = self.new_rv('intelligent', ('Yes', 'No'))
        grade = self.new_rv('grade', ('1', '2', '3'))
        sat = self.new_rv('sat', ('High', 'Low'))
        letter = self.new_rv('letter', ('Yes', 'No'))

        self.new_factor(difficult).set_cpt().set_all(
            (0.6, 0.4),
        )
        self.new_factor(intelligent).set_cpt().set_all(
            (0.7, 0.3),
        )
        self.new_factor(grade, difficult, intelligent).set_cpt().set_all(
            (0.3, 0.4, 0.3),
            (0.05, 0.25, 0.7),
            (0.9, 0.08, 0.02),
            (0.5, 0.3, 0.2)
        )
        self.new_factor(sat, intelligent).set_cpt().set_all(
            (0.95, 0.05),
            (0.2, 0.8),
        )
        self.new_factor(letter, grade).set_cpt().set_all(
            (0.1, 0.9),
            (0.4, 0.6),
            (0.99, 0.01),
        )
