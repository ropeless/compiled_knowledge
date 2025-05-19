from ck.pgm import PGM


class Rain(PGM):
    """
    This PGM is the pedagogical 'Rain' Bayesian network.
    See Adnan Darwiche, 2009, Modeling and Reasoning with Bayesian Networks, p127.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)

        a = self.new_rv('Winter', (True, False))
        b = self.new_rv('Sprinkler', (True, False))
        c = self.new_rv('Rain', (True, False))
        d = self.new_rv('Wet Grass', (True, False))
        e = self.new_rv('Slippery Road', (True, False))

        f_a = self.new_factor(a).set_cpt()
        f_ba = self.new_factor(b, a).set_cpt()
        f_ca = self.new_factor(c, a).set_cpt()
        f_dbc = self.new_factor(d, b, c).set_cpt()
        f_ec = self.new_factor(e, c).set_cpt()

        f_a.set_cpd((), (0.6, 0.4))

        f_ba.set_cpd(0, (0.2, 0.8))
        f_ba.set_cpd(1, (0.75, 0.25))

        f_ca.set_cpd(0, (0.8, 0.2))
        f_ca.set_cpd(1, (0.1, 0.9))

        f_dbc.set_cpd((0, 0), (0.95, 0.05))
        f_dbc.set_cpd((0, 1), (0.9, 0.1))
        f_dbc.set_cpd((1, 0), (0.8, 0.2))
        f_dbc.set_cpd((1, 1), (0, 1))

        f_ec.set_cpd(0, (0.7, 0.3))
        f_ec.set_cpd(1, (0, 1))
