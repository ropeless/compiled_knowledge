from ck.pgm import PGM


class Empty(PGM):
    """
    This PGM has no random variables.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
