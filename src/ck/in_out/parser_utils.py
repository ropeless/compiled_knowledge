"""
Support for parsers.

Defines:
    ParserInput
    ParseError.
"""
import re as _re
import string as _string
from io import StringIO as _StringIO


class ParseError(Exception):
    """
    Raised when a parse error occurs.
    Keeps track of the location of the error (line number and character position in the line).
    """

    def __init__(self, line_num: int, char_pos: int, message: str):
        full_msg = f'line {line_num}, pos {char_pos}: {message}'
        super(ParseError, self).__init__(full_msg)
        self.line_num: int = line_num
        self.char_pos: int = char_pos


class ParserInput:
    """
    A wrapper for an input stream of characters with
    integrated exception raising and infinite pushback.
    """

    def __init__(self, input_stream):
        self._input = _check_input(input_stream)
        self._prev_line_len = 0
        self._cur_line = 0
        self._cur_char = 1
        self._lookahead = []

    def read_one(self) -> str:
        """
        Read one character from the input stream.

        Returns:
            a character from the input stream,  or '' if EOF.
        """
        if len(self._lookahead) > 0:
            c = self._lookahead.pop()
        else:
            c = self._input.read(1)
        if c == '\n':
            self._prev_line_len = self._cur_char
            self._cur_line += 1
            self._cur_char = 1
        else:
            self._cur_char += 1
        return c

    def readline(self) -> str:
        """
        Returns:
             the next line (including the trailing '\n') or '' if EOF.
        """
        line = ''
        while True:
            c = self.read_one()
            line += c
            if c == '' or c == '\n':
                return line

    def read_past_space(self, single_line: bool, comment_char=None) -> str:
        """
        Returns:
            either empty string, '', if end of input, otherwise a single character string that is not whitespace.
            If single_line is True, then '\n' is treated as eof.
        """
        c = self.read_one()
        while True:
            if not single_line:
                while len(c) == 1 and c in _string.whitespace:
                    c = self.read_one()
            else:
                while len(c) == 1 and c in _string.whitespace and c != '\n':
                    c = self.read_one()

            if comment_char is None or c != comment_char:
                break

            # in a comment
            c = self.read_one()
            while len(c) == 1 and c != '\n':
                c = self.read_one()

        if single_line and c == '\n':
            self.pushback('\n')
            c = ''

        return c

    def pushback(self, c: str) -> None:
        """
        Push a character back to the input stream.

        Args:
            c: a character to push back to the input stream.
        """
        assert (len(c) == 1)
        self._lookahead.append(c)
        # Do our best to keep the input position counters sensible
        if c == '\n':
            self._cur_line -= 1
            self._cur_char = self._prev_line_len
            self._prev_line_len = 0
        else:
            self._cur_char -= 1

    def raise_error(self, message: str) -> None:
        """
        Raise a ParseError with the given message and the current line number and character position.

        Args:
            message: the error message.
        """
        if isinstance(message, Exception):
            message = f'Exception: {message.__class__.__name__} {message}'
        raise ParseError(self._cur_line, self._cur_char, message)


def _check_input(input_stream):
    """
    If the given argument is a string, then wrap it in a StringIO.
    """
    if isinstance(input_stream, str):
        return _StringIO(input_stream)
    else:
        return input_stream


def escape_string(s: str, *, double_quotes: bool = False, single_quotes: bool = False) -> str:
    """
    Return the given string with backslash escaping for special characters.
    Args:
        s: The string to process.
        double_quotes: should double quotes be escaped?
        single_quotes: should single quotes be escaped?

    Returns:

    """
    s = s.replace('\\', '\\\\')  # must come first
    s = s.replace('\n', '\\n')
    s = s.replace('\r', '\\r')
    s = s.replace('\v', '\\v')
    s = s.replace('\f', '\\f')
    s = s.replace('\f', '\\f')
    s = s.replace('\a', '\\a')
    s = s.replace('\b', '\\b')
    s = s.replace('\t', '\\t')
    if double_quotes:
        s = s.replace('"', '\\"')
    if single_quotes:
        s = s.replace("'", "\\'")
    return s


def clean_string(s: str, valid_chars: str = '[a-zA-Z0-9_]', replace_char: str = '_') -> str:
    """
    Remove any invalid char from the given string and replace with
    the given 'replace_char'

    Args:
        s: The string to process.
        valid_chars: the set of valid characters (as a regular expression).
        replace_char: the character to replace invalid characters with.

    Returns:
        the cleaned string.
    """
    re = _re.compile(valid_chars)
    result = ''
    for c in s:
        if re.match(c):
            result += c
        else:
            result += replace_char
    return result
