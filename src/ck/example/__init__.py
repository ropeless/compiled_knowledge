"""
A package of standard probabilistic graphical models.
"""
from ck.pgm import PGM
from ck.example.alarm import Alarm
from ck.example.binary_clique import BinaryClique
from ck.example.bow_tie import BowTie
from ck.example.cancer import Cancer
from ck.example.asia import Asia
from ck.example.chain import Chain
from ck.example.child import Child
from ck.example.clique import Clique
from ck.example.cnf_pgm import CNF_PGM
from ck.example.diamond_square import DiamondSquare
from ck.example.earthquake import Earthquake
from ck.example.empty import Empty
from ck.example.hailfinder import Hailfinder
from ck.example.hepar2 import Hepar2
from ck.example.insurance import Insurance
from ck.example.loop import Loop
from ck.example.mildew import Mildew
from ck.example.munin import Munin
from ck.example.pathfinder import Pathfinder
from ck.example.rectangle import Rectangle
from ck.example.rain import Rain
from ck.example.run import Run
from ck.example.sachs import Sachs
from ck.example.sprinkler import Sprinkler
from ck.example.survey import Survey
from ck.example.star import Star
from ck.example.stress import Stress
from ck.example.student import Student
from ck.example.triangle_square import TriangleSquare
from ck.example.truss import Truss


# A dictionary with entries, `name: class`, for all example PGM classes.
#
# Example usage:
#     from ck.example import ALL_EXAMPLES
#
#     my_pgm: PGM = ALL_EXAMPLES['Alarm']()
#
ALL_EXAMPLES = {
    name: pgm_class
    for name, pgm_class in globals().items()
    if (
           not name.startswith('_')
           and name != PGM.__name__
           and isinstance(pgm_class, type)
           and issubclass(pgm_class, PGM)
    )
}
