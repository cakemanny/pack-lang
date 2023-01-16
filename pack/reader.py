from itertools import islice

from pack.data import Sym, Keyword, Vec, List, Cons, nil, ArrayMap
from pack.exceptions import PackLangError, Unclosed, Unmatched, SyntaxError
from pack.util import take_pairs


# -------------
#  Reader
# -------------

class FileString(str):
    """
    A string that knows where it came from in a file
    """

    def __new__(cls, s, file, lineno, col):
        obj = super().__new__(cls, s)
        obj.__init__(s, file, lineno, col)
        return obj

    def __init__(self, s, file, lineno, col):
        super().__init__()
        self.file = file
        self.lineno = lineno
        self.col = col

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start = idx.start or 0
            if start != 0:
                lineno = self.lineno
                col = self.col
                for c in islice(self, 0, start):
                    if c == '\n':
                        col = 0
                        lineno += 1
                    else:
                        col += 1
                return self.__class__(
                    super().__getitem__(idx), self.file, lineno, col
                )
            return self.__class__(
                super().__getitem__(idx), self.file, self.lineno, self.col
            )

        # Not worried about encumbering a single character
        return super().__getitem__(idx)

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return f"{self.file}:{self.lineno}:{self.col} " + super().__repr__()


def location_from(obj):
    def from_filestring(fs):
        return fs.file, fs.lineno, fs.col
    match obj:
        case FileString() as s:
            return from_filestring(s)
        case Sym(FileString() as ns, _):
            return from_filestring(ns)
        case Sym(_, FileString() as n):
            return from_filestring(n)
        case Keyword(FileString() as ns, _):
            return from_filestring(ns)
        case Sym(_, FileString() as n):
            return from_filestring(n)
        case Cons(maybe_has_location, _):
            return location_from(maybe_has_location)  # will be slightly off
        case PackLangError() as err:
            return err.location
    return None


WHITESPACE = (' ', '\n', '\t', '\v')
WHITESPACE_OR_COMMENT_START = WHITESPACE + (';',)

# Still to do:
#   #
#   ^ for metadata


def is_ident_start(c):
    return (
        'a' <= c <= 'z'
        or 'A' <= c <= 'Z'
        or c in ('+', '-', '*', '/', '<', '>', '!', '=', '&', '_', '.', '?')
        or '🌀' <= c <= '🫸'
    )


def is_ident(c):
    return is_ident_start(c) or (c in ("'")) or '0' <= c <= '9'


def split_ident(n):
    if '/' in n[:-1]:
        idx = n.index('/')
        return n[:idx], n[idx + 1:]
    return None, n


def read_ident(text):
    i = 0
    for c in text:
        if is_ident(c):
            i += 1
        else:
            break

    return split_ident(text[:i]), text[i:]


def read_sym(text):
    (namespace, name), remaining = read_ident(text)
    if namespace is None:
        if name == 'true':
            return True, remaining
        if name == 'false':
            return False, remaining
        if name == 'nil':
            return None, remaining
    return Sym(namespace, name), remaining


def read_keyword(text):
    assert text[0] == ':'
    (namespace, name), remaining = read_ident(text[1:])
    return Keyword(namespace, name), remaining


def read_num(text, prefix=''):
    i = 0
    point = 0
    for c in text:
        if c == '.':
            point += 1
        if '0' <= c <= '9' or c == '.':
            i += 1
        else:
            break
    if point > 1:
        raise SyntaxError("invalid number literal: multiple '.'s", location_from(text))
    if point == 1:
        return float(prefix + text[:i]), text[i:]
    return int(prefix + text[:i]), text[i:]


def read_str(text):
    assert text[0] == '"'
    # reference to text for producing error locations
    location_text = text

    escapes = {
        'a': '\a', 'b': '\b', 'f': '\f', 'r': '\r', 'n': '\n',
        't': '\t', 'v': '\v'
    }

    def aux(text):
        seg_start = 0
        i = 0
        n = len(text)
        while i < n:
            c = text[i]
            if c == '"':
                yield text[seg_start:i]
                yield text[i + 1:]
                break
            if c == '\\':
                # switch to processing escape
                yield text[seg_start:i]
                seg_start = i
                i += 1
                c = text[i]
                if c == '\n':
                    pass
                elif c in ('\\', "'", '"'):
                    yield c
                elif c in escapes:
                    yield escapes[c]
                else:
                    raise SyntaxError(
                        f'unknown escape sequence \\{c}',
                        location_from(location_text)
                    )
                seg_start += 2
            i += 1
        if i == n:
            raise Unclosed('"', location_from(location_text))

    *segments, remaining = aux(text[1:])
    return ''.join(segments), remaining


def read_comment(text):
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '\n':
            i += 1
            break
        i += 1
    return Reader.NOTHING, text[i:]


def close_sequence(opener, elements):
    match opener:
        case '(':
            return List.from_iter(elements)
        case '[':
            return Vec.from_iter(elements)
        case '{':
            try:
                return ArrayMap.from_iter(take_pairs(elements))
            except ValueError:
                raise SyntaxError(
                    'A map literal must contain an even number of forms',
                    location_from(elements)
                ) from None
    raise ValueError('unknown opener', opener)


def read_list(opener, text, closing):
    remaining = text
    elements = []
    while True:
        try:
            elem, remaining = try_read(remaining)
            if elem is Reader.NOTHING:
                raise Unclosed(opener, location_from(text))
            elements.append(elem)
        except Unmatched as unmatched:
            if unmatched.c == closing:
                return close_sequence(opener, elements), unmatched.remaining
            else:
                raise SyntaxError(
                    f"trying to close a {opener!r} with a {unmatched.c!r}",
                    location_from(unmatched)
                )


def read_quoted_like(text, macro, prefix):
    to_quote, remaining = try_read(text)
    if to_quote is Reader.NOTHING:
        raise Unclosed(prefix, location_from(text))
    return Cons(Sym(None, macro), Cons(to_quote, nil)), remaining


def read_quoted(text):
    return read_quoted_like(text, 'quote', "'")


def read_quasiquoted(text):
    return read_quoted_like(text, 'quasiquote', "`")


def read_unquote(text):
    return read_quoted_like(text, 'unquote', "~")


def read_unquote_splicing(text):
    return read_quoted_like(text, 'unquote-splicing', "~@")


class Reader:
    NOTHING = object()
    """A sentinel value for when there was nothing except whitespace or comment"""


def try_read(text):

    if text == '':
        return Reader.NOTHING, text
    c = text[0]

    # eat whitespace
    while c in WHITESPACE_OR_COMMENT_START:
        if c == ';':
            _, text = read_comment(text)
        else:
            text = text[1:]
        if text == '':
            return Reader.NOTHING, text
        c = text[0]

    closer = {'(': ')', '[': ']', '{': '}'}

    c1 = text[1] if text[1:] else ''

    match c:
        case '(' | '[' | '{':
            return read_list(c, text[1:], closer[c])
        case ')' | ']' | '}':
            raise Unmatched(c, text[1:], location_from(text))
        case "'":
            return read_quoted(text[1:])
        case "`":
            return read_quasiquoted(text[1:])
        case "~" if c1 == "@":
            return read_unquote_splicing(text[2:])
        case "~":
            return read_unquote(text[1:])
        case '-' | '+' if '0' <= c1 <= '9':
            return read_num(text[1:], c)
        case n if '0' <= n <= '9':
            return read_num(text)
        case '"':
            return read_str(text)
        case ':':
            return read_keyword(text)
        case s if is_ident(s):
            return read_sym(text)

    raise NotImplementedError(c)


def read_all_forms(text):
    remaining = text
    forms = []
    while True:
        match try_read(remaining):
            case Reader.NOTHING, '':
                break
            case Reader.NOTHING, remaining:
                continue
            case form, remaining:
                forms.append(form)
    return tuple(forms)


def read_forms(previous_lines='', input=input, prompt='=> '):

    line = input(prompt if not previous_lines else '')

    remaining = previous_lines + '\n' + line if previous_lines else line
    remaining = FileString(remaining, "<stdin>", lineno=1, col=0)
    try:
        return read_all_forms(remaining)
    except Unclosed:
        return read_forms(remaining, input, prompt)
