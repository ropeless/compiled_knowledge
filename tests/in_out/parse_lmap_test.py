import unittest

from ck.in_out.parse_ace_lmap import read_lmap, LiteralMap
from ck.pgm import Indicator


class Test_lmap_parser(unittest.TestCase):

    def test_parse_empty(self):
        to_parse = ("""
            c this is a comment
            cc$K$ALWAYS_SUM
            cc$S$NORMAL
            cc$N$0
            cc$v$0
            cc$t$0
        """)

        lmap: LiteralMap = read_lmap(to_parse)

        self.assertEqual(len(lmap.rvs), 0)
        self.assertEqual(len(lmap.indicators), 0)
        self.assertEqual(len(lmap.params), 0)

    def test_parse_variables(self):
        to_parse = ("""
            c this is a comment
            cc$K$ALWAYS_SUM
            cc$S$NORMAL
            cc$N$10
            
            cc$t$0
            
            cc$v$2
            cc$V$x$2
            cc$V$y$3
            
            cc$I$-1$0.1$+$x$0
            cc$I$+1$0.2$+$x$1
            
            cc$I$2$0.3$+$y$0
            cc$I$3$0.4$+$y$1
            cc$I$4$0.5$+$y$2

        """)

        lmap: LiteralMap = read_lmap(to_parse)

        self.assertEqual(len(lmap.rvs), 2)
        self.assertEqual(len(lmap.indicators), 5)
        self.assertEqual(len(lmap.params), 5)

        self.assertEqual(lmap.rvs['x'].rv_idx, 0)
        self.assertEqual(lmap.rvs['x'].number_of_states, 2)
        self.assertEqual(lmap.rvs['x'].name, 'x')
        self.assertEqual(lmap.rvs['y'].rv_idx, 1)
        self.assertEqual(lmap.rvs['y'].number_of_states, 3)
        self.assertEqual(lmap.rvs['y'].name, 'y')

        self.assertEqual(lmap.indicators[-1], Indicator(0, 0))
        self.assertEqual(lmap.indicators[+1], Indicator(0, 1))

        self.assertEqual(lmap.indicators[2], Indicator(1, 0))
        self.assertEqual(lmap.indicators[3], Indicator(1, 1))
        self.assertEqual(lmap.indicators[4], Indicator(1, 2))

        self.assertEqual(lmap.params[-1], 0.1)
        self.assertEqual(lmap.params[+1], 0.2)

        self.assertEqual(lmap.params[2], 0.3)
        self.assertEqual(lmap.params[3], 0.4)
        self.assertEqual(lmap.params[4], 0.5)

    def test_parse_existing_variables(self):
        to_parse = ("""
            c this is a comment
            cc$K$ALWAYS_SUM
            cc$S$NORMAL
            cc$N$10
    
            cc$t$0

            cc$v$2
            cc$V$x$2
            cc$V$y$3
    
            cc$I$-1$0.1$+$x$0
            cc$I$+1$0.2$+$x$1
    
            cc$I$2$0.3$+$y$0
            cc$I$3$0.4$+$y$1
            cc$I$4$0.5$+$y$2
    
        """)

        lmap: LiteralMap = read_lmap(to_parse, node_names=['y', 'x'])

        self.assertEqual(len(lmap.rvs), 2)
        self.assertEqual(len(lmap.indicators), 5)
        self.assertEqual(len(lmap.params), 5)

        self.assertEqual(lmap.rvs['x'].rv_idx, 1)
        self.assertEqual(lmap.rvs['x'].number_of_states, 2)
        self.assertEqual(lmap.rvs['x'].name, 'x')
        self.assertEqual(lmap.rvs['y'].rv_idx, 0)
        self.assertEqual(lmap.rvs['y'].number_of_states, 3)
        self.assertEqual(lmap.rvs['y'].name, 'y')

        self.assertEqual(lmap.indicators[-1], Indicator(1, 0))
        self.assertEqual(lmap.indicators[+1], Indicator(1, 1))

        self.assertEqual(lmap.indicators[2], Indicator(0, 0))
        self.assertEqual(lmap.indicators[3], Indicator(0, 1))
        self.assertEqual(lmap.indicators[4], Indicator(0, 2))

        self.assertEqual(lmap.params[-1], 0.1)
        self.assertEqual(lmap.params[+1], 0.2)

        self.assertEqual(lmap.params[2], 0.3)
        self.assertEqual(lmap.params[3], 0.4)
        self.assertEqual(lmap.params[4], 0.5)

    def test_parse_parameters(self):
        to_parse = ("""
            c this is a comment
            cc$K$ALWAYS_SUM
            cc$S$NORMAL
            cc$N$6    
            cc$v$0
            
            cc$t$1
            cc$T$my-table$6
            
            cc$C$1$0.1$+$
            cc$C$-2$0.2$+$
            cc$C$3$0.3$+$
            cc$C$-4$0.4$+$
            cc$C$5$0.5$+$
            cc$C$-6$0.6$+$
        """)

        lmap: LiteralMap = read_lmap(to_parse)
        self.assertEqual(len(lmap.rvs), 0)
        self.assertEqual(len(lmap.indicators), 0)
        self.assertEqual(len(lmap.params), 6)

        self.assertEqual(lmap.params[1], 0.1)
        self.assertEqual(lmap.params[-2], 0.2)
        self.assertEqual(lmap.params[3], 0.3)
        self.assertEqual(lmap.params[-4], 0.4)
        self.assertEqual(lmap.params[5], 0.5)
        self.assertEqual(lmap.params[-6], 0.6)


if __name__ == '__main__':
    unittest.main()
