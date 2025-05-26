"""
Functionality for parsing literal mapping files (lmap), as produced by the software ACE.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Sequence

from ck.in_out.parser_utils import ParseError, ParserInput
from ck.pgm import Indicator


def read_lmap(
        input_stream,
        *,
        check_counts: bool = False,
        node_names: Sequence[str] = (),
) -> LiteralMap:
    """
    Parse the input literal-map file or string.
    
    An "lmap" file is produced by the software ACE to help interpret an "nnf" or "ac" file.
    See module `parse_ace_nnf` for more details.

    The returned LiteralMap will provide mapping from literals to indicators
    and mapping from literals to parameter values.

    If a PGM is passed in, then its random variables will be used (for indicators)
    otherwise a PGM will be created and random variables will be added to it as
    needed.

    Args:
        input_stream: an input that can be passed to `ParserInput`.
        node_names: optional ordering of node names (for indicators in returned literal map).
        check_counts: if true then literal and variable counts are checked.

    Returns:
        a LiteralMap object
    """
    parser = _LiteralMapParser(check_counts, node_names)
    parser.parse(input_stream)
    return parser.lmap


@dataclass
class LiteralMapRV:
    name: str
    rv_idx: int
    number_of_states: int


@dataclass
class LiteralMap:
    """
    A data structure to hold a literal-map, i.e., provide mapping
    from literals to indicators and mapping from literals to parameter values.

    Fields:
        rvs[name] = random_variable, where random_variable.name() == name
        indicators[literal_code] = indicator
        params[literal_code] = parameter_value
    """
    rvs: Dict[str, LiteralMapRV]
    indicators: Dict[int, Indicator]
    params: Dict[int, float]


class Parser(ABC):

    def parse(self, input_stream):
        input_stream = ParserInput(input_stream)
        raise_f = lambda msg: input_stream.raise_error(msg)
        try:
            line = input_stream.readline()
            while line:
                line = line.strip()
                if len(line) > 0:
                    if line[0] == 'c' and (len(line) == 1 or line[1] != 'c'):
                        self.comment(raise_f, line)
                    else:
                        line = line.split('$')
                        if line[0] != 'cc':
                            input_stream.raise_error(f'unexpected line start: {line[0]}')
                        code = line[1]
                        if code == 'N':
                            self.number_of_literals(raise_f, int(line[2]))
                        elif code == 'v':
                            self.number_of_rvs(raise_f, int(line[2]))
                        elif code == 'V':
                            self.rv(raise_f, line[2], int(line[3]))
                        elif code == 't':
                            self.number_of_tables(raise_f, int(line[2]))
                        elif code == 'T':
                            self.table(raise_f, line[2], int(line[3]))
                        elif code == 'I':
                            self.indicator(raise_f, int(line[2]), float(line[3]), line[4], line[5], int(line[6]))
                        elif code == 'C':
                            self.parameter(raise_f, int(line[2]), float(line[3]), line[4], line[5])
                        elif code in ['K', 'S']:
                            # ignore
                            pass
                        else:
                            input_stream.raise_error(f'unexpected line code: cc${code}')

                line = input_stream.readline()
            self.done(raise_f)
        except ParseError as e:
            raise e
        except Exception as e:
            input_stream.raise_error(str(e))

    @abstractmethod
    def comment(self, raise_f, message: str) -> None:
        pass

    @abstractmethod
    def number_of_literals(self, raise_f, num_literals: int) -> None:
        pass

    @abstractmethod
    def number_of_rvs(self, raise_f, num_rvs: int) -> None:
        pass

    @abstractmethod
    def rv(self, raise_f, rv_name, num_states: int) -> None:
        pass

    @abstractmethod
    def number_of_tables(self, raise_f, num_tables: int) -> None:
        pass

    @abstractmethod
    def table(self, raise_f, child_rv_name: str, num_states: int) -> None:
        pass

    @abstractmethod
    def indicator(self, raise_f, literal_code: int, weight: float, arithmetic_op: str, rv_name: str,
                  state: int) -> None:
        pass

    @abstractmethod
    def parameter(self, raise_f, literal_code: int, weight: float, arithmetic_op: str, extra: str) -> None:
        pass

    @abstractmethod
    def done(self, raise_f) -> None:
        pass


class _LiteralMapParser(Parser):

    def __init__(self, check_counts: bool, node_names: Sequence[str]):
        self.node_names: Dict[str, int] = {
            name: i
            for i, name in enumerate(node_names)
        }
        self.check_counts = check_counts
        self.lmap = LiteralMap({}, {}, {})
        self._number_of_literals = None
        self._number_of_rvs = None

    def number_of_literals(self, raise_f, num):
        self._number_of_literals = num

    def number_of_rvs(self, raise_f, num):
        self._number_of_rvs = num

    def rv(self, raise_f, rv_name, num_states):
        if rv_name in self.lmap.rvs.keys():
            raise_f(f'duplicated random variable: {rv_name}')

        idx: Optional[int] = self.node_names.get(rv_name)
        if idx is None:
            idx = len(self.node_names)
            self.node_names[rv_name] = idx

        literal_rv = LiteralMapRV(rv_name, idx, num_states)
        self.lmap.rvs[rv_name] = literal_rv

    def indicator(self, raise_f, literal_code, weight, arithmetic_op, rv_name, state):
        rv: Optional[LiteralMapRV] = self.lmap.rvs.get(rv_name)
        if rv is None:
            raise_f(f'unknown random variable: {rv_name}')
        if literal_code in self.lmap.indicators.keys() or literal_code in self.lmap.params.keys():
            raise_f(f'duplicated indicator literal: {literal_code}')
        self.lmap.indicators[literal_code] = Indicator(rv.rv_idx, state)
        self.lmap.params[literal_code] = weight

    def parameter(self, raise_f, literal_code, weight, arithmetic_op, extra):
        if literal_code in self.lmap.indicators.keys() or literal_code in self.lmap.params.keys():
            raise_f(f'duplicated parameter literal: {literal_code}')
        self.lmap.params[literal_code] = weight

    def comment(self, raise_f, message: str) -> None:
        pass

    def number_of_tables(self, raise_f, num_tables: int) -> None:
        pass

    def table(self, raise_f, child_rv_name: str, num_states: int) -> None:
        pass

    def done(self, raise_f) -> None:
        if self.check_counts:
            # Perform consistency checks
            if self._number_of_rvs is not None:
                got_rvs: int = len(self.lmap.rvs)
                if got_rvs != self._number_of_rvs:
                    raise_f(f'unexpected number of random variables: expected {self._number_of_rvs}, got {got_rvs}')

            if self._number_of_literals is not None:
                got_params: int = len(self.lmap.params)
                expect_params: int = self._number_of_literals * 2
                if got_params != expect_params:
                    raise_f(f'unexpected number of parameters: expected {expect_params}, got: {got_params}')
