from __future__ import annotations

import string
from typing import Protocol, Optional, List

from ck.in_out.parser_utils import ParseError, ParserInput
from ck.pgm import PGM, Factor, State
from ck.utils.iter_extras import combos_ranges


def read_network(input_stream, *, name: Optional[str] = None, network_builder: Optional[NetworkBuilder] = None) -> PGM:
    """
    Parser for Hugin "net" file format.

    Read a network and return it as a PGM.
    The input can be a string or a stream.
    If the input is empty, then its is treated as an error.

    This input is expected to conform to the following format.

    <network> ::= <net_block> <node_block>* <potential_block>*

    <net_block> ::= 'net' <block>

    <node_block> ::= 'node' <NAME> <block>

    <potential_block> ::= 'potential' <link> <block>

    <block> ::= '{' <sentence>* '}'
    <sentence> ::= <NAME> '=' <value> ';'

    <link> ::= '(' <NAME> ')'
             | '(' <NAME> '|' <NAME>+ ')'

    <value> ::= <STRING> | <NUMBER> | <list>
    <list> ::='(' <value>* ')'

    The sentences of a <net_block> are ignored.

    The sentences of a <node_block>
        <name> of 'states' mandatory, with value that is a list of <STRING>
        other sentences are ignored.

    The sentences of a <potential_block>
        <name> of 'data' mandatory, with value that is a list of (list of)* <NUMBER> (shape matching the link)
        other sentences are ignored.

    Here is a simple example input:
        net{}
        node a
        {
          states = ( "a0" "a1" );
        }
        node b
        {
          states = ( "b0" "b1" "b2" );
        }

        potential ( a )
        {
          data = ( 0.5 0.5 );
        }
        potential ( b | a )
        {
          data = ((0.4 0.4 0.2)(0.4 0.4 0.2)) ;
        }
    """
    # Decorate the input stream
    input_stream = ParserInput(input_stream)

    if network_builder is None:
        network_builder = PGM_NetworkBuilder(PGM_NetworkBuilder.add_function_dense)

    bn = network_builder.start(name)

    try:
        _read_net(network_builder, bn, input_stream)
        c = _read_past_space(input_stream)
        if len(c) != 0:
            input_stream.raise_error('unexpected extra characters at end of input')
    except ParseError as e:
        raise e
    except Exception as e:
        input_stream.raise_error(e)

    return network_builder.done(bn)


class NetworkBuilder(Protocol):
    """
    This is a protocol used by read_network to build the network
    """

    def start(self, name) -> object:
        ...

    def add_node(self, network, name, states):
        ...

    def add_factor(self, network, child_name, parent_names, data):
        ...

    def done(self, network) -> PGM:
        ...


class PGM_NetworkBuilder(NetworkBuilder):
    """
    This implementation of NetworkBuilder build a PGM object.
    At construction time, pass one of the add_function_[...] methods
    to the constructor, which selects the type of potential function
    that is used to hold parameters.
    """

    @staticmethod
    def add_function_dense(factor: Factor, data):
        function = factor.set_dense()
        shape = function.shape

        def my_iter():
            for key in combos_ranges(shape, flip=True):
                parent = key[1:]
                child = key[0]
                d = data
                for i in parent:
                    d = d[i]
                yield d[child]

        function.set_iter(my_iter())

    @staticmethod
    def add_function_cpt(factor: Factor, data):
        function = factor.set_cpt()
        shape = function.shape
        parent_shape = shape[1:]
        for parent in combos_ranges(parent_shape):
            cpd = data
            for i in parent:
                cpd = cpd[i]
            function.set_cpd(parent, cpd)

    @staticmethod
    def add_function_sparse(factor: Factor, data):
        function = factor.set_sparse()
        shape = function.shape
        for key in combos_ranges(shape):
            parent = key[1:]
            child = key[0]
            d = data
            for i in parent:
                d = d[i]
            function[key] = d[child]

    @staticmethod
    def add_function_compact(factor: Factor, data):
        function = factor.set_compact()
        shape = function.shape
        for key in combos_ranges(shape):
            parent = key[1:]
            child = key[0]
            d = data
            for i in parent:
                d = d[i]
            function[key] = d[child]

    def __init__(self, potential_function_type):
        """
        Args:
            potential_function_type: should be one of:
                PGM_NetworkBuilder.add_function_dense,
                PGM_NetworkBuilder.add_function_cpt,
                PGM_NetworkBuilder.add_function_sparse,
                PGM_NetworkBuilder.add_function_compact.
        """
        self.add_pot_function = potential_function_type

    def start(self, name):
        return [PGM(name), {}]

    def add_node(self, pgm_pair, name, states):
        pgm, rv_map = pgm_pair
        rv = pgm.new_rv(name, states)
        rv_map[name] = rv

    def add_factor(self, pgm_pair, child_name, parent_names, data):
        pgm, rv_map = pgm_pair
        rvs = [rv_map[rv_name] for rv_name in [child_name] + parent_names]
        factor = pgm.new_factor(*rvs)
        self.add_pot_function(factor, data)

    def done(self, pgm_pair) -> PGM:
        pgm, rv_map = pgm_pair
        return pgm


def _read_net(network_builder: NetworkBuilder, bn, input_stream) -> None:
    _read_net_block(input_stream)
    _read_blocks(network_builder, bn, input_stream)


def _pass_callback(*_) -> None:
    pass


def _read_past_space(input_stream) -> str:
    """
    Returns:
        either empty string, '', if end of input, otherwise a single character string that is not whitespace.
    """
    return input_stream.read_past_space(single_line=False, comment_char='%')


def _read_name(input_stream) -> str:
    """
    Returns:
        the read name
    """
    c = _read_past_space(input_stream)
    name = ""
    while len(c) == 1:
        if (
                c in string.ascii_letters
                or c in string.digits
                or c in "_-."
        ):
            name += c
        else:
            input_stream.pushback(c)
            break
        c = input_stream.read_one()

    if len(name) == 0:
        input_stream.raise_error('expected a name but none read')

    return name


def _read_keyword(input_stream) -> str:
    """
    Returns:
        the read keyword
    """
    c = _read_past_space(input_stream)
    keyword = ""
    while len(c) == 1:
        if c in string.ascii_letters:
            keyword += c
        else:
            input_stream.pushback(c)
            break
        c = input_stream.read_one()
    if len(keyword) == 0:
        input_stream.raise_error('expected a keyword but none read')
    return keyword.lower()


def _read_net_block(input_stream) -> None:
    keyword = _read_keyword(input_stream)
    if keyword != 'net':
        input_stream.raise_error(f'expected keyword "net" but got: {keyword!r}')
    _read_block(_pass_callback, input_stream)


def _read_blocks(network_builder, bn, input_stream) -> None:
    lookahead = _read_past_space(input_stream)
    while lookahead != '':
        input_stream.pushback(lookahead)
        keyword = _read_keyword(input_stream)
        if keyword == 'node':
            _read_node_block(network_builder, bn, input_stream)
        elif keyword == 'potential':
            _read_potential_block(network_builder, bn, input_stream)
        else:
            input_stream.raise_error(f'expected keyword "node" or "potential" but got: {keyword!r}')
        lookahead = _read_past_space(input_stream)


def _read_node_block(network_builder, bn, input_stream) -> None:
    """
    Assumes already read the 'node' keyword from the input.
    """
    name = _read_name(input_stream)

    names = [name, None]
    states = [None]

    def callback(_name: str, _value):
        if _name == 'label':
            if names[1] is not None:
                input_stream.raise_error(f'duplicate "label" sentence in node: {_name}')
            names[0] = names[1] = _value
        elif _name == 'states':
            if states[0] is not None:
                input_stream.raise_error(f'duplicate "states" sentence in node: {_name}')
            assert _value is not None
            states[0] = _value

    _read_block(callback, input_stream)

    states: Optional[List[State]] = states[0]
    name: str = names[0]

    if states is None:
        input_stream.raise_error(f'no "states" found in node: {name}')
    elif len(states) < 2:
        input_stream.raise_error(f'must be at least 2 states in a node: {name}')
    network_builder.add_node(bn, name, states)


def _read_potential_block(network_builder, bn, input_stream):
    """
    Assumes already read the 'potential' keyword from the input.
    """
    link = _read_link(input_stream)
    data = [None]

    def callback(name, value):
        if name == 'data':
            if data[0] is not None:
                input_stream.raise_error(f'duplicate "data" sentence in potential: {link}')
            assert value is not None
            data[0] = value

    _read_block(callback, input_stream)
    network_builder.add_factor(bn, link[0], link[1:], data[0])


def _read_link(input_stream) -> List[str]:
    """
    Returns:
        the read link
    """
    c = _read_past_space(input_stream)
    if c != '(':
        input_stream.raise_error('reading a link, expected "("')

    name = _read_name(input_stream)
    link = [name]

    c = _read_past_space(input_stream)
    if c == ')':
        return link
    if c != '|':
        input_stream.raise_error('reading a link, expected ")" or "|"')

    c = _read_past_space(input_stream)
    while c != ')':
        if c == '':
            input_stream.raise_error('unexpected end of input while reading a link for a potential')
        input_stream.pushback(c)
        name = _read_name(input_stream)
        link.append(name)
        c = _read_past_space(input_stream)
    return link


def _read_block(sentence_callback, input_stream):
    c = _read_past_space(input_stream)
    if c != '{':
        input_stream.raise_error('parse error - expected "{"')
    c = _read_past_space(input_stream)
    while c != '}':
        if c == '':
            input_stream.raise_error('parse error - unexpected end of input while reading a block')
        input_stream.pushback(c)
        _read_sentence(sentence_callback, input_stream)
        c = _read_past_space(input_stream)


def _read_sentence(sentence_callback, input_stream):
    name = _read_name(input_stream)
    c = _read_past_space(input_stream)
    if c != '=':
        input_stream.raise_error('parse error - expected "="')
    value = _read_value(input_stream)
    c = _read_past_space(input_stream)
    if c != ';':
        input_stream.raise_error('parse error - expected ";"')
    sentence_callback(name, value)


def _read_value(input_stream) -> List | str | float:
    """
    Returns:
        (value, lookahead)
    """
    lookahead = _read_past_space(input_stream)
    if lookahead == '':
        input_stream.raise_error('parse error - expected value - unexpected end of input')

    input_stream.pushback(lookahead)
    if lookahead == '(':
        # reading a list
        value = _read_list(input_stream)
    elif lookahead == '"':
        # reading a string
        value = _read_string(input_stream)
    else:
        # reading a number
        value = _read_number(input_stream)
    return value


def _read_list(input_stream) -> List:
    """
    Returns:
        the read list of values
    """
    c = _read_past_space(input_stream)
    if c != '(':
        input_stream.raise_error('parse error - expected "("')
    result = []
    c = _read_past_space(input_stream)
    while c != ')':
        if c == '':
            input_stream.raise_error('parse error - unexpected end of input while reading list')
        input_stream.pushback(c)
        value = _read_value(input_stream)
        result.append(value)
        c = _read_past_space(input_stream)
    return result


def _read_string(input_stream) -> str:
    """
    Returns:
        the read string value
    """
    c = _read_past_space(input_stream)
    if c != '"':
        input_stream.raise_error('parse error - expected open quote (")')
    result = ""
    while True:
        c = input_stream.read_one()
        if c == '':
            input_stream.raise_error('parse error - unexpected end of input while reading string')
        if c == '"':
            break
        if c == '\\':
            c = input_stream.read_one()
            if c in '\\\'\"':
                pass
            elif c == 'n':
                c = '\n'
            elif c == 'r':
                c = '\r'
            elif c == 'v':
                c = '\v'
            elif c == 'f':
                c = '\f'
            elif c == 'a':
                c = '\a'
            elif c == 'b':
                c = '\b'
            elif c == 't':
                c = '\t'
            else:
                input_stream.raise_error(f'parse error - unexpected escape code while reading string: \\{c}')
        result += c
    return result


def _read_number(input_stream) -> float:
    """
    Return:
        the read numeric value
    """
    value = ''
    c = _read_past_space(input_stream)
    while len(c) == 1:
        if not (c in string.digits or c in ".eE+-"):
            input_stream.pushback(c)
            break
        value += c
        c = input_stream.read_one()
    try:
        value = float(value)
    except ValueError:
        input_stream.raise_error('parse error - could not parse number')
    return value
