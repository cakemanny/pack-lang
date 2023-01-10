import dataclasses
import importlib
import operator
import os
import sys
import traceback
from collections.abc import Sequence, Mapping, Set
from dataclasses import dataclass
from functools import reduce
from itertools import chain, islice
from typing import Any, Optional, Collection, Iterable, Iterator
from typing import Generic, TypeVar


# ----------------
#  Data Structures
# ----------------


@dataclass(frozen=True, slots=True)
class Sym:
    ns: Optional[str]
    n: str

    def __repr__(self):
        if self.ns:
            return self.ns + '/' + self.n
        return self.n


@dataclass(frozen=True, slots=True)
class Keyword:
    ns: Optional[str]
    n: str

    def __repr__(self):
        if self.ns:
            return f':{self.ns}/{self.n}'
        return f':{self.n}'

    def __call__(self, a_map, default=None):
        try:
            return a_map.get(self, default)
        except AttributeError:
            # clearly not a map
            return None


# should make this a sequence ...
class List:
    @staticmethod
    def from_iter(it):
        if isinstance(it, List):
            return it

        try:
            rev_it = reversed(it)
        except TypeError:
            # not reversible
            rev_it = reversed(tuple(it))
        xs = nil
        for x in rev_it:
            xs = Cons(x, xs)
        return xs


class Nil(List):

    __slots__ = ()

    def __repr__(self):
        return '()'

    def __len__(self):
        return 0

    def __iter__(self):
        yield from ()

    def __reversed__(self):
        return self

    def __add__(self, o):
        if not isinstance(o, List):
            raise TypeError(
                'can only concatenate List (not "%s") to List'
                % (type(o).__name__)
            )
        return o

    if False:
        def __eq__(self, o):
            if isinstance(o, Nil):
                return True
            if o is None:
                return True
            return False

        def __hash__(self):
            return hash(None)


nil = Nil()
# all instances of Nil() are nil
Nil.__new__ = lambda cls: nil


@dataclass(frozen=True, slots=True)
class Cons(List):
    hd: Any
    tl: 'Optional[List]'

    def __post_init__(self):
        if self.tl is not None and self.tl is not nil:
            assert isinstance(self.tl, Cons), type(self.tl)

    def __iter__(self):
        cons = self
        while cons is not None and cons is not nil:
            yield cons.hd
            cons = cons.tl

    def __reversed__(self):
        # FIXME: This should return an iterator, right?
        result = nil
        for x in self:
            result = Cons(x, result)
        return result

    def __repr__(self):
        return '(' + ' '.join(map(repr, self)) + ')'

    def __bool__(self):
        return True

    def __len__(self):
        return sum(map(lambda _: 1, self), 0)

    def __add__(self, o):
        if o is nil or o is None:
            return self
        if not isinstance(o, List):
            raise TypeError(
                'can only concatenate List (not "%s") to List'
                % (type(o).__name__)
            )
        result = o
        for x in reversed(self):
            result = Cons(x, result)
        return result


# itertools recipes
# https://docs.python.org/3/library/itertools.html#itertools-recipes
def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch


@dataclass(frozen=True, slots=True)
class Vec(Sequence):
    """
    A Trie with at most 32 elements in each node
    """
    xs: tuple[Any | 'Vec']
    height: int

    # A classic sign that this should be split into two different
    # subclasses Leaf and non-leaf Node
    def _is_leaf(self):
        return self.height == 0

    def __len__(self):
        # O(log32(n))
        if self._is_leaf():
            return len(self.xs)
        # Since vectors are contiguous, we just need to look at the
        # final subnode
        return (1 << (5 * self.height)) * (len(self.xs) - 1) + len(self.xs[-1])

    def __getitem__(self, idx: int):
        # TODO: accept slices
        if idx < 0:
            return self[len(self) + idx]
        if idx >= len(self):
            raise IndexError('vector index out of range')

        if self._is_leaf():
            return self.xs[idx]

        subvec_idx = idx >> (5 * self.height)

        mask = (1 << (5 * self.height)) - 1

        return self.xs[subvec_idx][mask & idx]

    def __iter__(self):
        if self._is_leaf():
            return iter(self.xs)
        return chain.from_iterable(self.xs)

    def __reversed__(self):
        if self._is_leaf():
            return reversed(self.xs)
        return chain.from_iterable(map(reversed, self.xs))

    def conj(self, x):
        # Should we just use _add_iter with a a 1-tuple?
        if self._is_leaf():
            if len(self.xs) < 32:
                return Vec(self.xs + (x,), 0)
            return Vec((self, Vec((x,), 0)), 1)

        old_tail = self.xs[-1]
        new_tail = old_tail.conj(x)
        if new_tail.height == old_tail.height:
            return Vec(
                self.xs[:-1] + (new_tail,),
                self.height,
            )
        if len(self.xs) < 32:
            return Vec(
                self.xs + new_tail.xs[1:],
                self.height,
            )

        new_new_tail = x
        for i in range(self.height + 1):
            new_new_tail = Vec((new_new_tail,), i)

        return Vec((self, new_new_tail), self.height + 1)

    def _fill(self, ys: Iterator):
        """
        fill the space in the vector from ys, without overflowing or
        increasing height
        """
        if self._is_leaf():
            first = islice(ys, 32 - len(self.xs))
            return Vec(self.xs + tuple(first), 0), ys

        last_node, remaining = self.xs[-1]._fill(ys)
        more_vecs = (
            Vec.from_iter(batch)
            for batch in islice(
                batched(remaining, (1 << (5 * self.height))),
                (32 - len(self.xs)),
            )
        )
        return Vec(
            self.xs[:-1] + (last_node,) + tuple(more_vecs),
            self.height
        ), remaining

    def _add_iter(self, ys):
        "append the contents of an iterator to this vector"
        filled, remaining = self._fill(iter(ys))
        try:
            n = next(remaining)
            remaining = chain(iter([n]), remaining)
            is_more = True
        except StopIteration:
            is_more = False

        if not is_more:
            return filled
        return Vec(xs=(filled,), height=(self.height + 1))._add_iter(remaining)

    def __add__(self, other):
        if not isinstance(other, Vec):
            raise TypeError(
                'can only concatenate Vec (not "%s") to Vec'
                % (type(other).__name__)
            )
        return self._add_iter(other)

    def __call__(self, idx: int):
        return self[idx]

    def __repr__(self):
        return '[' + ' '.join(map(repr, self)) + ']'

    @staticmethod
    def from_iter(xs: Iterable):
        # Every time the first batch in the iterator has less than
        # 32 items we nest the iterator into another iterator that
        # batches up up to 32 of those.
        def aux(it, level):
            it0 = (Vec(ys, level) for ys in batched(it, 32))

            first = next(it0)
            if len(first.xs) < 32:
                return first
            try:
                second = next(it0)
            except StopIteration:
                return first

            # undo having taken the first item
            it0 = chain(iter([first, second]), it0)

            return aux(it0, level + 1)

        # Due to the construction of aux, only an empty xs will cause
        # StopIteration to be raised.
        try:
            return aux(xs, 0)
        except StopIteration:
            return _EMPTY_VEC

    @staticmethod
    def empty():
        return _EMPTY_VEC


_EMPTY_VEC = Vec((), 0)


@dataclass(frozen=True, slots=True)
class ArrayMap(Mapping):
    kvs: tuple

    def __len__(self):
        return len(self.kvs) // 2

    def __iter__(self):
        return islice(self.kvs, 0, None, 2)

    def __getitem__(self, key):
        kvs = self.kvs
        for i in range(0, len(kvs), 2):
            k = kvs[i]
            if key is k or key == k:
                return kvs[i + 1]
        raise KeyError(key)

    def __call__(self, key, default=None):
        return self.get(key, default)

    def __eq__(self, o):
        if not isinstance(o, Mapping):
            return False
        if len(o) != len(self):
            return False
        # inefficient nested loop join
        try:
            for k, v in self.items():
                if v != o[k]:
                    return False
            for k in o:
                if self[k] != o[k]:
                    return False
        except KeyError:
            return False
        return True

    # We implement items and values ourselves because the mixins from
    # Mapping would be O(n^2)
    def items(self) -> Set:
        kvs = self.kvs

        class ItemsView(Set):
            def __iter__(self):
                return take_pairs(kvs)

            def __len__(self):
                return len(kvs) // 2

            def __contains__(self, item):
                return any(item == (k, v) for (k, v) in self)
        return ItemsView()

    def values(self) -> Collection:
        kvs = self.kvs

        class ValuesView:
            def __len__(self):
                return len(kvs) // 2

            def __contains__(self, value):
                for v in iter(self):
                    if v is value or v == value:
                        return True
                return False

            def __iter__(self):
                return islice(kvs, 1, None, 2)
        return ValuesView()

    def __repr__(self):
        return '{' + '  '.join(
            f'{k!r} {v!r}' for (k, v) in self.items()
        ) + '}'

    def assoc(self, key, value):
        new_kvs = list(self.kvs)
        for i in range(0, len(new_kvs), 2):
            if new_kvs[i] == key:
                if new_kvs[i + 1] is value or new_kvs[i + 1] == value:
                    return self
                new_kvs[i + 1] = value
                break
        else:
            new_kvs.append(key)
            new_kvs.append(value)
        if len(new_kvs) > 16:
            return Map.from_iter(take_pairs(new_kvs))
        return ArrayMap(tuple(new_kvs))

    def dissoc(self, key):
        if key not in self.keys():
            return self
        return self.from_iter(
            (k, v) for (k, v) in self.items() if k != key
        )

    @staticmethod
    def empty():
        return _EMPTY_ARRAY_MAP

    @classmethod
    def from_iter(cls, it):
        def aux():
            seen = set()
            seen_add = seen.add
            for k, v in it:
                if k not in seen:
                    seen_add(k)
                    yield k
                    yield v
        return cls(tuple(aux()))


_EMPTY_ARRAY_MAP = ArrayMap(())


@dataclass(frozen=True, slots=True)
class Map(Mapping):
    """
    A HAMT Map. A Map is a 32 element tuple containing either a map entry or a
    another map with further levels of the tree. The index is the first 5
    bits of a 32-bit hash
    """
    xs: tuple

    kindset: int
    "A 32-bit bitset with 0 a map node, 1 for a map entry"

    height: int

    _len: int

    def __post_init__(self):
        assert self._len >= 0
        assert self.height >= 0

    def __len__(self):
        return self._len

    def _hash32(self, k):
        return hash(k) & ((1 << 32) - 1)

    def _idx_for_key(self, k):
        h = self._hash32(k)
        return (h >> (self.height * 5)) & 0b11111

    def _is_leaf(self, idx):
        return bool(self.kindset & (1 << idx))

    def __getitem__(self, k):
        idx = self._idx_for_key(k)
        if self._is_leaf(idx):
            entry = self.xs[idx]
            if entry is None or entry[0] != k:
                raise KeyError(k)
            return entry[1]

        next_map = self.xs[idx]
        if next_map is None:
            raise KeyError(k)
        return next_map[k]

    def __iter__(self):
        for i, x in enumerate(self.xs):
            if self._is_leaf(i):
                yield x[0]
            elif x is not None:
                for k in x:
                    yield k

    def _kindset_setting_subnode(self, idx):
        "copy of kindset with idx now as a subnode"
        # clear bit idx
        return (self.kindset & (((1 << 32) - 1) ^ (1 << idx)))

    def _kindset_setting_leaf(self, idx):
        "copy of kindset with idx now as a leaf slot (map entry)"
        return (self.kindset | (1 << idx))

    def _with_replacement(self, idx, new_value, *, leaf: bool):
        "return a new map with a single item in the xs tuple replaced"
        if leaf:
            (k, v) = new_value  # assertion

        new_xs = tuple(
            new_value if i == idx else x
            for (i, x) in enumerate(self.xs)
        )
        new_kindset = (
            self._kindset_setting_leaf(idx) if leaf
            else self._kindset_setting_subnode(idx)
        )

        len_of_replaced = (
            1 if self._is_leaf(idx)
            else len(node) if (node := self.xs[idx]) is not None
            else 0
        )
        len_of_replacement = (
            1 if leaf
            else len(new_value) if new_value is not None
            else 0
        )
        new_len = (self._len - len_of_replaced + len_of_replacement)

        return Map(new_xs, new_kindset, _len=new_len, height=self.height)

    def assoc(self, k, v):
        idx = self._idx_for_key(k)

        if self._is_leaf(idx):
            entry = self.xs[idx]
            assert entry is not None
            if entry == (k, v):
                return self
            if entry[0] != k:
                if self.height == 0:
                    assert self._hash32(k) == self._hash32(entry[0])
                    raise NotImplementedError('hash collision')
                # conflict, replace entry with self-node
                new_subnode = (
                    dataclasses.replace(
                        Map.empty(), height=(self.height - 1)
                    )
                    .assoc(entry[0], entry[1])
                    .assoc(k, v)
                )
                return self._with_replacement(idx, new_subnode, leaf=False)

            # This will break when we have to bucket things
            assert entry[0] == k
            # Replace a key
            return self._with_replacement(idx, (k, v), leaf=True)
        subnode = self.xs[idx]
        if subnode is None:
            # put (k, v) as a entry in this node
            return self._with_replacement(idx, (k, v), leaf=True)

        new_subnode = subnode.assoc(k, v)
        return self._with_replacement(idx, new_subnode, leaf=False)

    def dissoc(self, key):
        idx = self._idx_for_key(key)

        if self._is_leaf(idx):
            (k, v) = self.xs[idx]
            if k != key:
                if self.height == 0:
                    raise NotImplementedError('hash collision')
                # key is already not there
                return self
            # replace leaf with empty node
            return self._with_replacement(idx, None, leaf=False)
        subnode = self.xs[idx]
        if subnode is None:
            return self
        new_subnode = subnode.dissoc(key)
        assert len(new_subnode) != 0
        if len(new_subnode) == 1:
            new_entry = next(iter(new_subnode.items()))
            return self._with_replacement(idx, new_entry, leaf=True)
        return self._with_replacement(idx, new_subnode, leaf=False)

    def __repr__(self):
        return '{' + '  '.join(
            f'{k!r} {v!r}' for (k, v) in self.items()
        ) + '}'

    @classmethod
    def empty(cls):
        if not hasattr(cls, '_empty'):
            cls._empty = cls(
                tuple([None] * 32), kindset=0, _len=0, height=7
            )
        return cls._empty

    @staticmethod
    def from_iter(it):
        m = Map.empty()
        for k, v in it:
            m = m.assoc(k, v)
        return m

# -------------
#  Constants
# -------------


class Special:
    "A simple namespace for some constants that we use"

    DOT = Sym(None, '.')
    DO = Sym(None, 'do')
    DEF = Sym(None, 'def')
    LETSTAR = Sym(None, 'let*')
    LOOP = Sym(None, 'loop')
    RECUR = Sym(None, 'recur')
    IF = Sym(None, 'if')
    FN = Sym(None, 'fn')
    RAISE = Sym(None, 'raise')
    QUOTE = Sym(None, 'quote')
    VAR = Sym(None, 'var')

    all = {DOT, DO, DEF, LETSTAR, LOOP, RECUR, IF, FN, RAISE, QUOTE, VAR}


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

    segments = list(aux(text[1:]))
    return ''.join(segments[:-1]), segments[-1]


def read_comment(text):
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '\n':
            i += 1
            break
        i += 1
    return None, text[i:]


def take_pairs(xs):
    """yield pairs from an even length iterable: ABCDEF -> AB CD EF"""
    # See https://docs.python.org/3/library/functions.html#zip Tips and tricks
    return zip(it := iter(xs), it, strict=True)


def untake_pairs(pairs):
    "does the reverse of take_pairs. pairs should be an iterable of tuples"
    return chain.from_iterable(pairs)


def close_sequence(opener, elements):
    match opener:
        case '(':
            return List.from_iter(elements)
        case '[':
            return Vec.from_iter(elements)
        case '{':
            try:
                return Map.from_iter(take_pairs(elements))
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
            if elem is None:
                raise Unclosed(opener, location_from(text))
            elements.append(elem)
        except Unmatched as unmatched:
            if unmatched.c == closing:
                return close_sequence(opener, elements), unmatched.remaining
            raise


def read_quoted_like(text, macro, prefix):
    to_quote, remaining = try_read(text)
    if to_quote is None:
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


class PackLangError(Exception):
    def __init__(self, msg, /, location=None):
        if location is not None:
            super().__init__(msg, location)
        else:
            super().__init__(msg)
        self.location = location


class SyntaxError(PackLangError):
    pass


class Unmatched(SyntaxError):
    "something ended that hadn't begun"
    def __init__(self, c, remaining, location=None):
        super().__init__(c, location)
        # c is the character that was unmatched
        self.c = c
        # We store the remaining text on the exception in case it's
        # possible to recover. (In fact that's how we read lists.)
        self.remaining = remaining


class Unclosed(SyntaxError):
    "something started but never ended"


def try_read(text):

    if text == '':
        return None, text
    c = text[0]

    # eat whitespace
    while c in WHITESPACE_OR_COMMENT_START:
        if c == ';':
            _, text = read_comment(text)
        else:
            text = text[1:]
        if text == '':
            return None, text
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
            case None, '':
                break
            case None, remaining:
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


# -------------
#  Interpreter
# -------------

class SemanticError(PackLangError):
    pass


class EvalError(PackLangError):
    pass


# We abuse the python exception system to implement tail recursion
class RecurError(PackLangError):
    def __init__(self, form, arg_vals):
        super().__init__("recur must be defined inside a loop or fn",
                         location_from(form))
        self.form = form
        self.arg_vals = arg_vals

    def __str__(self):
        return f"recur must be defined inside a loop or fn: {self.form}"


@dataclass(frozen=True)
class Namespace:
    name: str
    defs: Map | ArrayMap = ArrayMap.empty()  # str -> var

    def apply_defs(self, f):
        return dataclasses.replace(self, defs=f(self.defs))


def initial_pack_core():
    # Creates pack.core with some magic interpreter variables
    # This is not the full contents of pack.core. See pack/core.pack

    def initial_ns_macro(ns_sym):
        return Cons(Sym('pack.core', 'in-ns'),
                    Cons(Cons(Sym(None, 'quote'), Cons(ns_sym, nil)), nil))

    ns = Namespace(
        'pack.core',
        defs=ArrayMap.empty()
        .assoc('*ns*', Var(Sym('pack.core', '*ns*')))
        .assoc('*compile*', Var(Sym('pack.core', '*compile*'), False))
        .assoc('refer', Var(Sym('pack.core', 'refer'), rt_refer))
        .assoc('in-ns', Var(Sym('pack.core', 'in-ns'), rt_in_ns))
        .assoc('ns', Var(Sym('pack.core', 'ns'),
                         initial_ns_macro,
                         metadata=ArrayMap.empty().assoc(KW_MACRO, True)))
    )
    ns.defs['*ns*'].value = ns
    return ns


@dataclass(frozen=True)
class Interpreter:
    namespaces: Map | ArrayMap = dataclasses.field(
        default_factory=lambda: ArrayMap.empty().assoc(
            'pack.core', initial_pack_core()
        )
    )  # str -> Namespace

    def switch_namespace(self, name: str):
        assert isinstance(name, str)
        if name not in self.namespaces:
            ns = Namespace(name,
                           defs=ArrayMap.empty()
                           # always make sure we are able to switch
                           # namespace
                           .assoc('ns', self.namespaces['pack.core'].defs['ns']))
        else:
            ns = self.namespaces[name]

        namespaces = self.namespaces.assoc(name, ns)

        # Mutation!
        namespaces['pack.core'].defs['*ns*'].value = ns

        return dataclasses.replace(self, namespaces=namespaces)

    def update_ns(self, ns_name, f):
        updated_ns = f(self.namespaces[ns_name])
        namespaces = self.namespaces.assoc(ns_name, updated_ns)
        if ns_name == namespaces['pack.core'].defs['*ns*'].value.name:
            namespaces['pack.core'].defs['*ns*'].value = updated_ns
        return dataclasses.replace(self, namespaces=namespaces)

    @property
    def current_ns(self):
        return self.namespaces['pack.core'].defs['*ns*'].value

    def resolve_symbol(self, sym, env):
        if sym in env:
            return env[sym]
        ns_name = sym.ns or self.current_ns.name
        try:
            ns = self.namespaces[ns_name]
        except KeyError:
            raise EvalError(
                f"no such namespace: {ns_name}", location_from(sym)
            ) from None
        try:
            return ns.defs[sym.n]
        except KeyError:
            raise EvalError(
                f'could not resolve name in this context: {sym!r}', location_from(sym)
            ) from None

    @property
    def compilation_enabled(self):
        return self.namespaces['pack.core'].defs['*compile*'].value


@dataclass
class Var:
    "This is the mutable thing"
    symbol: Sym
    value: Any
    metadata: ArrayMap | Map

    def __init__(self, symbol, value=None, metadata=ArrayMap.empty()):
        assert isinstance(symbol, Sym)
        self.symbol = symbol
        self.value = value
        self.metadata = metadata

    def __repr__(self):
        return f"#'{self.symbol}"


KW_MACRO = Keyword(None, "macro")


def set_macro(var):
    var.metadata = var.metadata.assoc(KW_MACRO, True)


def is_macro(sym, interp):
    try:
        var = interp.resolve_symbol(sym, ArrayMap.empty())
    except PackLangError:
        return False
    try:
        return var.metadata.get(KW_MACRO)
    except AttributeError:
        return False


class Fn:
    def __init__(self, name, params, body, env, interp):
        assert isinstance(params, Vec)
        self.name = name
        self.params, self.restparam = self._split_params(params)
        self.body = body
        self.env = env
        self.interp = interp

    def _split_params(self, params: Vec):
        if Sym(None, '&') not in params:
            return params, None
        idx = params.index(Sym(None, '&'))
        new_params = [params[i] for i in range(0, idx)]
        restparam = params[idx + 1]
        return new_params, restparam

    def __call__(self, *args):
        if not self.restparam and len(args) != len(self.params):
            raise EvalError('incorrect number of arguments', location_from(args))
        env = self.env
        for p, arg in zip(self.params, args):
            env = env.assoc(p, arg)
        if self.restparam:
            restargs = args[len(self.params):]
            env = env.assoc(self.restparam, List.from_iter(restargs))
        return eval_expr(self.body, self.interp, env)

    def __repr__(self):
        return f'<Fn({self.name}) object at {hex(id(self))}>'


class RTFn:
    """
    An function that potentially affects interpreter state.
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, interp, *args) -> (Any, Interpreter):
        if not isinstance(interp, Interpreter):
            raise TypeError('first argument must be an Interpreter')
        return self.func(interp, *args)

    def __repr__(self):
        return f'<RTFn({self.func}) object at {hex(id(self))}>'


def is_interpreter_fn(sym, interp):
    try:
        var = interp.resolve_symbol(sym, ArrayMap.empty())
    except PackLangError:
        return False
    return isinstance(var.value, RTFn)


def rt_apply(f, args):
    if isinstance(f, RTFn):
        raise TypeError('apply was given a function that can alter the interpreter')
    return f(*args)


def _rt_eval(interp, form):
    [result], interp = expand_and_evaluate_forms([form], interp)
    return result, interp


def _rt_refer(interp, namespace_name):
    if not isinstance(namespace_name, Sym) or namespace_name.ns is not None:
        raise SemanticError(
            f'module must be a simple symbol (no /): invalid: {namespace_name}'
        )
    name = namespace_name.n

    def update_defs(defs):
        for key, var in interp.namespaces[name].defs.items():
            if var.symbol.ns == name:
                if key not in interp.current_ns.defs:
                    defs = defs.assoc(key, var)
        return defs

    return None, interp.update_ns(
        interp.current_ns.name,
        lambda ns: ns.apply_defs(update_defs)
    )


def _rt_in_ns(interp, namespace_name):
    if not isinstance(namespace_name, Sym) or namespace_name.ns is not None:
        raise SemanticError(
            f'namespace must be a simple symbol (no /): invalid: {namespace_name}'
        )
    name = namespace_name.n
    return None, interp.switch_namespace(name)


rt_eval = RTFn(_rt_eval)
"""eval - as to be imported and used from lisp programs. evaluates a single form"""
rt_refer = RTFn(_rt_refer)
rt_in_ns = RTFn(_rt_in_ns)


def extract_closure(body, params, interp, env):
    # To decide is, whether or not the values from the interp should be
    # captured as they are or not... I think the right answer is to
    # do so...
    # TODO
    m = ArrayMap.empty()  # closure. i.e. the result

    # TODO: combine params and lets into bound_vars and use a Set for this
    lets = ArrayMap.from_iter([(p, p) for p in params])

    def aux(expr, m, lets):
        match expr:
            case str() | int() | float() | bool() | Nil() | None | Keyword():
                return m
            case Sym(None, 'true' | 'false' | 'nil'):
                return m
            case Sym(None, name) as sym:
                # Maybe this is also the point to do more macro expanding?
                if sym in lets:
                    return m
                if sym in env:
                    return m.assoc(sym, env[sym])
                if name in interp.current_ns.defs:
                    return m.assoc(sym, interp.current_ns.defs[name])
                raise EvalError(
                    f'Could not resolve symbol: {sym!r} in this context',
                    location_from(sym)
                )
            case Sym(ns, name) if ns is not None:
                # Can be resolved during evaluation ...
                # TODO: think about whether it might be better to
                # always resolve now and include in the closure...
                return m
            # Special forms
            case Cons(Sym(None, '.'), Cons(obj, Cons(_, Nil()))):
                # only the obj is evaluated, not the attr
                return aux(obj, m, lets)
            case Cons(Sym(None, '.'), _):
                raise SyntaxError(
                    f'invald member expression: {expr}', location_from(expr)
                )
            case Cons(Sym(None, 'def'), Cons(Sym(), Cons(init, Nil()))):
                return aux(init, m, lets)
            # TODO: let* ?
            case Cons(Sym(None, 'let*' | 'loop'),
                      Cons(Vec() as bindings, Cons(body, Nil()))):
                for binding, init in take_pairs(bindings):
                    m = aux(init, m, lets)
                    lets = lets.assoc(binding, binding)
                return aux(body, m, lets)
            case Cons(Sym(None, 'fn'),
                      Cons(Sym(), Cons(Vec() as fn_params, Cons(body, Nil())))) \
                    | Cons(Sym(None, 'fn'),
                           Cons(Vec() as fn_params, Cons(body, Nil()))):
                lets = reduce(lambda xs, p: xs.assoc(p, p), fn_params, lets)
                return aux(body, m, lets)
            case Cons(Sym(None, 'fn'),
                      Cons(Sym(), Cons(Vec() as fn_params, Cons(body, Nil())))) \
                    | Cons(Sym(None, 'fn'),
                           Cons(Vec() as fn_params, Cons(body, Nil()))):
                lets = reduce(lambda xs, p: xs.assoc(p, p), fn_params, lets)
                return aux(body, m, lets)
            case Cons(Sym(None, 'quote'), _):
                return m
            case Cons(Sym(None, 'if' | 'raise' | 'var' | 'recur' | 'do'), args):
                return reduce(lambda m, sub_form: aux(sub_form, m, lets), args, m)
            case Cons(hd, _) as lst:
                assert hd not in Special.all
                return reduce(lambda m, sub_form: aux(sub_form, m, lets), lst, m)
            case Vec() as vec:
                return reduce(lambda m, sub_form: aux(sub_form, m, lets), vec, m)
            case ArrayMap() | Map() as m:
                for k, v in m.items():
                    m = aux(k, m, lets)
                    m = aux(v, m, lets)
                return m
            case _:
                raise NotImplementedError(expr)

    return aux(body, m, lets)


def to_module_path(module_name):
    if (len(module_name) == 0
            or '.' in (module_name[0], module_name[-1])
            or '/' in module_name):
        raise ValueError(f'invalid module name: {module_name}')

    with_replaced_seps = os.path.join(*module_name.split('.'))
    return f'{with_replaced_seps}.pack'


def find_module(module_name):
    module_path = to_module_path(module_name)
    for base in sys.path:
        full_path = os.path.join(base, module_path)
        if os.path.isfile(full_path):
            return full_path

    details = '\n'.join([
        f"searched {module_path!r} in the following directories:\n",
        *sys.path
    ])
    raise PackLangError(f'module not found: {details}')


def load_module(module_path, interp):
    with open(module_path) as f:
        forms = read_all_forms(
            FileString(f.read(), file=module_path, lineno=1, col=0)
        )
    _, interp = expand_and_evaluate_forms(forms, interp)
    return interp


# --------------
# Compiler start
# --------------

L = TypeVar('L')
R = TypeVar('R')


@dataclass(frozen=True)
class Left(Generic[L, R]):
    l: L


@dataclass(frozen=True)
class Right(Generic[L, R]):
    r: R


def fan_in(b_to_d, c_to_d):
    """
    (|||) ::: (b -> d) -> (c -> d) -> Either b c -> d
    (|||) = either
    """
    def aux(either_b_or_c):
        match either_b_or_c:
            case Left(b):
                return b_to_d(b)
            case Right(c):
                return c_to_d(c)
    return aux


def cata_f(fmap, unfix=lambda x: x):
    """
    generalised fold-right over any functor. takes the fmap for that functor

    cata alg = alg . fmap (cata alg) . unfix
    """
    def cata(alg):
        return lambda f: alg(fmap(cata(alg), unfix(f)))
    return cata


def ana_f(fmap, fix=lambda x: x):
    """
    generalised unfold.
    we can use it for top down traverals

    ana :: Functor f => (a -> f a) -> a -> Fix f
    ana coalg = Fix . fmap (ana coalg) . coalg
    """
    def ana(coalg):
        return lambda a: fix(fmap(ana(coalg), coalg(a)))
    return ana


def hylo_f(fmap):
    """
    unfold and then fold

    hylo :: Functor f => (f b -> b) -> (a -> f a) -> a -> b
    hylo g h = cata g . ana h
    <=>
    hylo f g = f . fmap (hylo f g) . g
    """
    # we implement the second, fused version, which is more space-efficient
    def hylo(alg, coalg):
        return lambda a: alg(fmap(hylo(alg, coalg), coalg(a)))
    return hylo


def apo_f(fmap, fix=lambda x: x):
    """
    shortcutting anamorphism

    apo :: Fixpoint f t => (a -> f (Either a t)) -> a -> t
    apo coa = inF . fmap (apo coa ||| id) . coa
    """
    identity = lambda x: x

    def apo(coa):
        return lambda a: fix(fmap(fan_in(apo(coa), identity), coa(a)))
    return apo


def compose(*fs):
    "compose functions of a single argument. compose(f, g)(x) == f(g(x))"
    def composed(x):
        for f in reversed(fs):
            x = f(x)
        return x
    return composed


def fmap(f, expr):
    "describes recursion into sub-expressions in our lisp grammar"
    # we assume all macros have been expanded already...?
    match expr:
        case x if x is nil:
            return nil
        case None | True | False | int() | float() | str() | Keyword():
            return expr
        case Cons(Sym(None, 'require' | 'import'), _):
            raise ValueError(f'not allowed within fn: {expr}')
        case Cons(Sym(None, '.') as s, Cons(obj, Cons(attr, Nil()))):
            # attr will always be a symbol
            return Cons(s, Cons(f(obj), Cons(attr, nil)))
        case Cons(Sym(None, 'do') as s, args):
            return Cons(s, List.from_iter(map(f, args)))
        case Cons(Sym(None, 'def'), _):
            raise ValueError(f'def is a statement not an expression: {expr}')

        # I think we have to assume that the lhs in the bindings
        # are symbols - ... so I think that means we only call f on the values
        case Cons(Sym(None, 'let*' | 'loop') as s,
                  Cons(Vec() as bindings, Cons(body, Nil() | None))):
            def map_seconds(f, vec):
                for binding, init in take_pairs(vec):
                    yield binding
                    yield f(init)
            return Cons(s,
                        Cons(Vec.from_iter(map_seconds(f, bindings)),
                             Cons(f(body), nil)))
        case Cons(Sym(None, 'recur') as s, args):
            return Cons(s, List.from_iter(map(f, args)))
        case Cons(Sym(None, 'if') as s,
                  Cons(pred, Cons(consequent, Cons(alternative, Nil())))):

            return Cons(s,
                        Cons(f(pred),
                             Cons(f(consequent),
                                  Cons(f(alternative), nil))))
        case Cons(Sym(None, 'fn') as s, Cons(Vec() as params, Cons(body, Nil()))):
            return Cons(s, Cons(params, Cons(f(body), nil)))
        case Cons(Sym(None, 'fn') as s,
                  Cons(Sym(None, _) as n, Cons(Vec() as params, Cons(body, Nil())))):
            return Cons(s, Cons(n, Cons(params, Cons(f(body), nil))))
        case Cons(Sym(None, 'raise') as s, Cons(r, Nil() | None)):
            # Raise is also a statement?
            return Cons(s, Cons(f(r), nil))
        case Cons(Sym(None, 'quote'), Cons(_, Nil())):
            return expr
        case Cons(Sym(None, 'var'), Cons(Sym(), Nil() | None)):
            return expr
        case Cons() as lst:
            return List.from_iter(map(f, lst))
        case Sym() as sym:
            return sym
        case Vec() as vec:
            return Vec.from_iter(map(f, vec))
        case ArrayMap() as m:
            return ArrayMap.from_iter((f(k), f(v)) for (k, v) in m.items())
        case Map() as m:
            return Map.from_iter((f(k), f(v)) for (k, v) in m.items())
        case _:
            raise NotImplementedError(expr)


cata = cata_f(fmap)
ana = ana_f(fmap)


def reduce_expr(zero, plus, expr):
    """
    reduce when a is a monoid in f a -> a

    e.g. (if a b c) -> a + b + c
         (let* [a b] c) -> a + c
         ...
    """
    match expr:
        case x if x is nil:
            return zero
        case None | True | False | int() | float() | str() | Keyword():
            return zero

        case Cons(Sym(None, '.'), Cons(obj, Cons(_, Nil()))):
            return obj
        case Cons(Sym(None, 'do'), args):
            return reduce(plus, args, zero)
        case Cons(Sym(None, 'let*' | 'loop'),
                  Cons(Vec() as bindings, Cons(body, Nil() | None))):
            every_second = islice(bindings, 1, None, 2)
            return plus(reduce(plus, every_second, zero), body)
        case Cons(Sym(None, 'recur'), args):
            return reduce(plus, args, zero)
        case Cons(Sym(None, 'if'),
                  Cons(pred, Cons(consequent, Cons(alternative, Nil())))):
            return plus(plus(pred, consequent), alternative)
        case Cons(Sym(None, 'fn'), Cons(Vec(), Cons(body, Nil()))):
            return body
        case Cons(Sym(None, 'fn'), Cons(Sym(None, _), Cons(Vec(), Cons(body, Nil())))):
            return body
        case Cons(Sym(None, 'raise'), Cons(r, Nil() | None)):
            return r
        case Cons(Sym(None, 'quote'), Cons(_, Nil())):
            return zero
        case Cons(Sym(None, 'var'), Cons(Sym(), Nil() | None)):
            return zero
        case Cons() as lst:
            return reduce(plus, iter(lst), zero)
        case Sym():
            return zero
        case Vec() as vec:
            return reduce(plus, vec, zero)
        case Map() | ArrayMap() as m:
            return reduce(plus, untake_pairs(m.items()), zero)
        case _:
            raise NotImplementedError(expr)


def fmap_datum(f, datum):
    match datum:
        case x if x is nil:
            return nil
        case None | True | False | int() | float() | str() | Keyword():
            return datum
        case Cons() as lst:
            return List.from_iter(map(f, lst))
        case Sym() as sym:
            return sym
        case Vec() as vec:
            return Vec.from_iter(map(f, vec))
        case ArrayMap() as m:
            return ArrayMap.from_iter((f(k), f(v)) for (k, v) in m.items())
        case Map() as m:
            return Map.from_iter((f(k), f(v)) for (k, v) in m.items())
        case _:
            raise NotImplementedError(datum)


def remove_vec_and_map_alg(expr):
    "replace map and vector literals with their constructors"
    match expr:
        case Vec() as vec:
            return Cons(Sym('pack.core', 'vector'), List.from_iter(vec))
        case ArrayMap() | Map() as m:
            return Cons(Sym('pack.core', 'hash-map'),
                        List.from_iter(untake_pairs(m.items())))
        case other:
            return other


def remove_quote_alg(datum):
    """
    remove complex quoted datum by replacing literals with their
    constructors. quoted symbols remain.
    '(a 1 [b 2 {c 3 :d 4}]) -> (list 'a 1 (vector 'b 2 (hash-map 'c 3 :d 4)))
    """
    match datum:
        case Sym() as sym:
            return Cons(Sym(None, 'quote'), Cons(sym, nil))
        case Cons() | Nil() as lst:
            return Cons(Sym('pack.core', 'list'), List.from_iter(lst))
        case other:
            return other


remove_quote = cata_f(fmap_datum)(compose(remove_vec_and_map_alg, remove_quote_alg))


# This is in ultra-draft idea mode at the moment
def compile_fn(fn: Fn, interp, *, mode='func'):
    """
    mode can be one of
    * func -> returns a python function
    * lines -> returns the lines of python that make up the body of the
               function
    """
    import keyword

    class Uncompilable(Exception):
        pass

    name = fn.name
    body = fn.body
    params = tuple(fn.params)
    restparam = fn.restparam
    closure = fn.env

    def mangle_name(name):
        if keyword.iskeyword(name):
            return name + '__'

        name = name.replace('.', '_DOT_')
        name = name.replace('?', '_QUESTION_')
        name = name.replace('+', '_PLUS_')
        name = name.replace('-', '_MINUS_')
        name = name.replace('*', '_STAR_')
        name = name.replace('/', '_SLASH_')

        if not name.isidentifier():
            raise NotImplementedError(name)
        return name

    def raise__(e: BaseException):
        raise e

    def create_fn(body_lines, resolved_qualifieds):
        # FIXME: we will almost definitely have to do some param name
        # conversion
        args = ', '.join(
            [sym.n for sym in params]
            + [f'*{sym.n}' for sym in filter(None, [restparam])]
        )
        fn_body = '\n'.join([f'  {b}' for b in body_lines])
        fn_name = mangle_name(
            name or f'fn_{hash((params, restparam, body)) & ((1 << 63) - 1)}'
        )

        txt = f' def {fn_name}({args}):\n{fn_body}'

        locals = {mangle_name(sym.n): v for (sym, v) in closure.items()} | {
            '__List_from_iter': List.from_iter,
            '__Sym': Sym,
            '__Keyword': Keyword,
            'raise__': raise__
        }
        local_vars = ', '.join(locals.keys())
        txt = f"def __create_fn__({local_vars}):\n{txt}\n return {fn_name}"
        globals = {
            mangle_name(str(var.symbol)): var for var in resolved_qualifieds
        }
        ns = {}
        exec(txt, globals, ns)
        return ns['__create_fn__'](**locals)

    # First stage should be to make all names unique, and put the values
    # in a single map... maybe

    # There will need to be a stage that converts expressions containing
    # raise, into a statement sequence? or... define a raise_ func

    def remove_true_false_and_nil_alg(expr):
        match expr:
            case Sym(None, 'false'):
                return False
            case Sym(None, 'true'):
                return True
            case Sym(None, 'nil'):
                return None
            case _:
                return expr

    def remove_complex_quote_alg(expr):
        match expr:
            case Cons(Sym(None, 'quote'), Cons(datum, Nil())):
                return remove_quote(datum)
            case _:
                return expr

    def compile_expr_alg(expr):
        """
        assume all subexpressions have already been compiled and then embedded
        into the structure of our AST
        """

        match expr:
            case str() | int() | float() | bool() | None as lit:
                return repr(lit)
            case Keyword(None, n):
                # because of the FileString instances, we have to convert to
                # normal strings
                return f'__Keyword(None, {str(n)!r})'
            case Keyword(ns, n):
                return f'__Keyword({str(ns)!r}, {str(n)!r})'
            case Sym(None, n) as sym:
                # TODO: find let-bound vars
                if sym == restparam:
                    return mangle_name(n)
                if sym in params:
                    return mangle_name(n)
                if sym in closure:
                    # Or... should this be doing a .value access when this
                    # is referring to a Var?
                    if isinstance(closure[sym], Var):
                        return mangle_name(n) + '.value'
                raise Uncompilable(f'unresolved: {n}')
            case Sym() as sym:
                return mangle_name(str(sym)) + '.value'
            case Cons(Sym(None, 'if'),
                      Cons(predicate,
                           Cons(consequent, Cons(alternative, Nil())))):
                return f'({consequent}) if ({predicate}) else ({alternative})'
            case Cons(Sym(None, '.'), Cons(obj, Cons(attr, Nil()))):
                return f'({obj}).{attr}'
            case Cons(Sym(None, 'raise'), Cons(raisable, Nil())):
                return f'raise__({raisable})'

            case Cons(Sym(None, 'quote'), Cons(Sym(None, n), Nil())):
                return f'__Sym(None, {str(n)!r})'
            case Cons(Sym(None, 'quote'), Cons(Sym(ns, n), Nil())):
                return f'__Sym({str(ns)!r}, {str(n)!r})'

            case Cons(proc_form, args):
                # TODO: deal with Fns that take an interpreter
                joined_args = ', '.join(args)
                return f'({proc_form})({joined_args})'
            case _:
                raise Uncompilable(expr)

    prog1 = cata(compose(
        remove_vec_and_map_alg,
        remove_complex_quote_alg,
        remove_true_false_and_nil_alg,
    ))(body)
    after_transforms = prog1

    def used_qualifieds_alg(expr):
        match expr:
            case Sym(ns, _) as sym if ns is not None:
                return (sym,)
            case other:
                return reduce_expr(zero=(), plus=operator.add, expr=other)

    used_qualifieds = cata(used_qualifieds_alg)(after_transforms)

    resolved_qualifieds = [
        interp.resolve_symbol(sym, ArrayMap.empty()) for sym in used_qualifieds
    ]

    body_lines = []
    try:
        if restparam is not None:
            mn = mangle_name(restparam.n)
            body_lines += [f'{mn} = __List_from_iter({mn})']
        body_lines += ['return ' + cata(compile_expr_alg)(after_transforms)]
        if mode == 'lines':
            return body_lines
        return create_fn(body_lines, resolved_qualifieds)
    except Uncompilable:
        traceback.print_exc()
        return None


# ------------------------
# Compiler End
# AKA: Interpreter Resumed
# ------------------------


def eval_top_level_form(form, interp):
    """
    Top Level Forms are the only kind of forms that mutate the interpreter
    i.e. make changes to namespaces

    - def
    - do
    - import
    - require
    - refer
    """
    match form:
        case Cons(Sym(None, 'require'), Cons(name_form, _)):
            name = eval_expr(name_form, interp, ArrayMap.empty())
            if name.ns is not None:
                raise SemanticError(
                    f'module must be a simple name (no /): invalid: {name}'
                )
            module_path = find_module(name.n)
            saved_namespace_name = interp.current_ns.name
            interp = load_module(module_path, interp)
            interp = interp.switch_namespace(saved_namespace_name)
            return None, interp

        case Cons(Sym(None, 'import'), Cons(Sym(None, name) as sym, _)):
            # TODO: check that the name is not already used for something else
            module = importlib.import_module(name)

            saved_namespace_name = interp.current_ns.name
            interp = interp.switch_namespace('py')
            _, interp = expand_and_evaluate_forms([
                Cons(Sym(None, 'def'), Cons(sym, nil))
            ], interp)

            var = interp.resolve_symbol(Sym("py", name), ArrayMap.empty())
            var.value = module

            interp = interp.switch_namespace(saved_namespace_name)

            if interp.current_ns.defs.get(name):
                # Don't overwrite current mapping
                return var, interp

            return var, interp.update_ns(
                interp.current_ns.name,
                lambda ns: ns.apply_defs(
                    lambda defs: defs.assoc(name, var)
                )
            )

        case Cons(Sym(None, 'def'), Cons(Sym(ns_name, name), Nil())) as form:
            return eval_top_level_form(form + Cons(None, nil), interp)
        case Cons(Sym(None, 'def'), Cons(Sym(ns_name, name), Cons(init, Nil()))):
            init_val = eval_expr(init, interp, ArrayMap.empty())
            var = interp.namespaces[ns_name or interp.current_ns.name].defs[name]
            # !Mutation!
            var.value = init_val
            return var, interp

        # Top Level do can contain other top level stuff, so that
        # macros are able to expand to multiple side-effects
        case Cons(Sym(None, 'do'), exprs):
            for expr in exprs:
                result, interp = eval_top_level_form(expr, interp)
            return result, interp

        case Cons(Sym() as sym, args) if is_interpreter_fn(sym, interp):
            proc = eval_expr(sym, interp, ArrayMap.empty())
            assert isinstance(proc, RTFn)

            arg_vals = [
                eval_expr(arg, interp, ArrayMap.empty()) for arg in args
            ]
            return proc(interp, *arg_vals)

        case other:
            return eval_expr(other, interp, ArrayMap.empty()), interp


def eval_expr(form, interp, env):
    match form:
        case x if x is nil:
            return nil
        case None | True | False | int() | float() | str() | Keyword():
            return form

        case Cons(Sym(None, '.'), Cons(obj_expr, Cons(Sym(None, attr), Nil()))):
            obj = eval_expr(obj_expr, interp, env)
            return getattr(obj, attr)
        case Cons(Sym(None, '.'), _):
            raise SyntaxError(f'invald member expression: {form}', location_from(form))

        case Cons(Sym(None, 'do'), exprs):
            for expr in exprs:
                result = eval_expr(expr, interp, env)
            return result

        case Cons(Sym(None, 'def'), _):
            raise SyntaxError(
                'def may only appear at top level or in top level do',
                location_from(form)
            )

        case Cons(Sym(None, 'let*'), Cons(Vec() as bindings, Cons(body, Nil() | None))):
            if len(bindings) % 2 != 0:
                raise SyntaxError(
                    'uneven number of forms in let bindings', location_from(bindings)
                )

            for binding, init in take_pairs(bindings):
                if not isinstance(binding, Sym) or binding.ns is not None:
                    raise SyntaxError(
                        f'let* does not support destructuring: problem: {binding}',
                        location_from(binding)
                    )
                init_val = eval_expr(init, interp, env)
                env = env.assoc(binding, init_val)
            return eval_expr(body, interp, env)
        case Cons(Sym(None, 'let*'), _):
            raise SyntaxError(f"invalid let: {form}", location_from(form))

        case Cons(Sym(None, 'loop'), Cons(Vec() as bindings, Cons(body, Nil() | None))):
            if len(bindings) % 2 != 0:
                raise SyntaxError(
                    'uneven number of forms in loop bindings', location_from(bindings)
                )

            for binding, init in take_pairs(bindings):
                if not isinstance(binding, Sym) or binding.ns is not None:
                    raise SyntaxError(
                        f'loop does not support destructuring: problem: {binding}',
                        location_from(binding)
                    )
                init_val = eval_expr(init, interp, env)
                env = env.assoc(binding, init_val)
            while True:
                try:
                    return eval_expr(body, interp, env)
                except RecurError as e:
                    for (binding, _), val in zip(take_pairs(bindings), e.arg_vals):
                        env = env.assoc(binding, val)
        case Cons(Sym(None, 'loop'), _):
            raise SyntaxError(f"invalid loop: {form}", location_from(form))

        case Cons(Sym(None, 'recur'), args):
            arg_vals = [
                eval_expr(arg, interp, env) for arg in args
            ]
            raise RecurError(form, arg_vals)

        case Cons(Sym(None, 'if'),
                  Cons(predicate,
                       Cons(consequent, Cons(alternative, Nil())))):
            p = eval_expr(predicate, interp, env)
            if p:
                return eval_expr(consequent, interp, env)
            return eval_expr(alternative, interp, env)
        case Cons(Sym(None, 'if'), Cons(predicate, Cons(consequent, Nil()))):
            p = eval_expr(predicate, interp, env)
            if p:
                return eval_expr(consequent, interp, env)
            return None
        case Cons(Sym(None, 'if')):
            raise SyntaxError(
                f'something is wrong with this if: {form}', location_from(form)
            )

        case Cons(Sym(None, 'fn'), Cons(Vec() as params, Cons(body, Nil()))):
            closure = extract_closure(body, params, interp, env)
            fn = Fn(None, params, body, closure, interp)
            if interp.compilation_enabled:
                compiled = compile_fn(fn, interp)
                if compiled is not None:
                    return compiled
            return fn
        case Cons(Sym(None, 'fn'),
                  Cons(Sym(None, name), Cons(Vec() as params, Cons(body, Nil())))):
            # TODO: create a var to store the name in?
            closure = extract_closure(body, params, interp, env)
            fn = Fn(name, params, body, closure, interp)
            if interp.compilation_enabled:
                compiled = compile_fn(fn, interp)
                if compiled is not None:
                    return compiled
            return fn
        case Cons(Sym(None, 'fn'), _):
            raise SyntaxError(f'invalid fn form: {form}', location_from(form))

        case Cons(Sym(None, 'raise'), Cons(raisable_form, Nil() | None)):
            raisable = eval_expr(raisable_form, interp, env)
            raise raisable
        case Cons(Sym(None, 'raise')):
            raise SyntaxError(
                f"invalid special form 'raise': {form}", location_from(form)
            )

        case Cons(Sym(None, 'quote'), Cons(arg, Nil())):
            return arg
        case Cons(Sym(None, 'quote'), args):
            raise SemanticError(
                f'wrong number of arguments to quote: {len(args)}'
            )

        case Cons(Sym(None, 'var'), Cons(Sym() as sym, Nil() | None)):
            return interp.resolve_symbol(sym, env)
        case Cons(Sym(None, 'var'), _):
            raise SemanticError(
                f'invalid special form var: var takes 1 symbol as argument: {form}'
            )

        case Cons(proc_form, args):
            proc = eval_expr(proc_form, interp, env)
            if isinstance(proc, RTFn):
                raise TypeError(
                    'attempted to call top level function outside top level'
                )
            arg_vals = []
            for arg in args:
                arg_val = eval_expr(arg, interp, env)
                arg_vals.append(arg_val)
            while True:
                try:
                    return proc(*arg_vals)
                except RecurError as e:
                    arg_vals = e.arg_vals

        case Sym(None, 'true'):
            return True
        case Sym(None, 'false'):
            return False
        case Sym(None, 'nil'):
            return None

        case Sym() as sym:
            resolved = interp.resolve_symbol(sym, env)
            if isinstance(resolved, Var):
                return resolved.value
            return resolved

        case ArrayMap() | Map() as m:
            result_m = ArrayMap.empty()
            for k, v in m.items():
                k_val = eval_expr(k, interp, env)
                v_val = eval_expr(v, interp, env)
                result_m = result_m.assoc(k_val, v_val)
            return result_m
        case Vec() as vec:
            values = []
            for sub_form in vec:
                value = eval_expr(sub_form, interp, env)
                values.append(value)
            return Vec.from_iter(values)

        case RTFn():
            return form
        # These have to go at the bottom because map, vec and keyword, etc
        # implement callable as well, but we are just wanting to deal with
        # python functions here
        case f if callable(f):
            return f

    raise NotImplementedError(form)


def create_defs(form, interp):
    # Notice that only defs at the top level are found
    match form:
        case Cons(Sym(None, 'def'), Cons(Sym(ns_name, name), (
            # (def xxx init) | (def blah/xxx init)
            Cons(_, Nil())
            # (def xxx) | (def blah/xxx)
            | Nil()
        ))):
            if ns_name is None:
                ns_name = interp.current_ns.name
            try:
                ns = interp.namespaces[ns_name]
            except KeyError:
                raise SemanticError(f'No such namespace {ns_name!r}') from None
            if ns_name != interp.current_ns.name:
                raise SemanticError(
                    'Cannot def symbol outside current ns'
                )

            if name in ns.defs:
                # TODO: Should this error if the name refers to an alias
                # already?
                return form, interp

            return form, interp.update_ns(
                ns_name,
                lambda ns: ns.apply_defs(
                    lambda defs: defs.assoc(name, Var(Sym(ns_name, name)))
                )
            )
        case Cons(Sym(None, 'def'), _):
            raise SemanticError('def expects a symbol as argument')

        case Cons(Sym(None, 'do') as s, stmts):
            # defs are also allowed inside 'do' s
            new_forms = []
            for subform in stmts:
                new_form, interp = create_defs(subform, interp)
                new_forms.append(new_form)
            return Cons(s, List.from_iter(new_forms)), interp

        case other:
            return other, interp


def macroexpand_1(form, interp):
    # We use ana (anamorphism) for a top-down expansion
    # ... actually we use apo (apomorphism), because ana doesn't
    # work if the macro expands directly to another macro call

    # This is no longer used, but it's a nice example of using apo

    def coalgebra(form):
        nonlocal interp
        # I think we may find that generating symbols won't
        # work unless we thread the interpreter through as well...
        match form:
            case Cons(Sym() as sym, args) if is_macro(sym, interp):
                proc = eval_expr(sym, interp, ArrayMap.empty())
                if isinstance(proc, RTFn):
                    expanded_form, interp = proc(interp, *tuple(args))
                else:
                    expanded_form = proc(*tuple(args))
                return fmap_datum(Right, expanded_form)
            case other:
                return fmap_datum(Left, other)

    return apo_f(fmap_datum)(coalgebra)(form), interp


def macroexpand_old(form, interp):
    while True:
        new_form, interp = macroexpand_1(form, interp)
        if new_form == form:
            return new_form, interp
        form = new_form


def macroexpand(form, interp):
    # expand all macros in a single pass

    # The coalgebra embeds the expanded form in a marked form
    # in order that the next iteration ana on the subform finds it
    # and further expands any macros found in the expansion

    def coalgebra(form):
        nonlocal interp

        # I think we may find that generating symbols won't
        # work unless we thread the interpreter through as well...
        match form:
            case Cons(Sym() as sym, args) if is_macro(sym, interp):
                proc = eval_expr(sym, interp, ArrayMap.empty())
                if isinstance(proc, RTFn):
                    expanded_form, interp = proc(interp, *tuple(args))
                else:
                    expanded_form = proc(*tuple(args))
                return Cons(Sym(None, '_MARKER_'), Cons(expanded_form, nil))
            case other:
                return other

    def algebra(form):
        match form:
            case Cons(Sym(None, '_MARKER_'), Cons(expanded_form, Nil())):
                return expanded_form
            case other:
                return other

    return hylo_f(fmap_datum)(algebra, coalgebra)(form), interp


# Roughly following
# https://www.cs.cmu.edu/Groups/AI/html/cltl/clm/node190.html
def expanding_quasi_quote(form, interp):
    def quote_form(form):
        return Cons(Sym(None, 'quote'), Cons(form, nil))
    match form:
        case Sym(None, 'true' | 'false' | 'nil'):
            return form
        case Sym(None, 'require' | 'import'):
            return quote_form(form)
        case Sym(None, '.' | 'def' | 'let*' | 'if' | 'fn' | 'raise' | 'quote'
                 | 'var' | 'recur' | 'loop' | 'do'):
            return quote_form(form)
        case Sym(None, name) as sym:
            try:
                var = interp.resolve_symbol(sym, ArrayMap.empty())
                if isinstance(var, Var):
                    return quote_form(var.symbol)
                raise NotImplementedError
            except EvalError:
                return quote_form(Sym(interp.current_ns.name, name))
        case Sym(_, _):
            return quote_form(form)
        case Cons(Sym(None, 'unquote'), Cons(x, Nil())):
            return x
        case Cons(Sym(None, 'unquote-splicing'), Cons(x, Nil())):
            raise SyntaxError('"~@" found not inside a sequence', location_from(form))
        case Nil():
            return nil
        case Cons() as lst:
            def splice(form):
                match form:
                    case Cons(Sym(None, 'unquote-splicing'), Cons(x, Nil())):
                        return x
                    case _:
                        return Cons(Sym('pack.core', 'list'),
                                    Cons(expanding_quasi_quote(form, interp), nil))
            return Cons(Sym('pack.core', 'concat'),
                        List.from_iter(map(splice, lst)))
        case Vec() as vec:
            elems = List.from_iter(vec)
            return Cons(Sym('pack.core', 'apply'),
                        Cons(Sym('pack.core', 'vector'),
                             Cons(expanding_quasi_quote(elems, interp), nil)))
        case ArrayMap() | Map() as m:
            elems = List.from_iter(untake_pairs(m.items()))
            return Cons(Sym('pack.core', 'apply'),
                        Cons(Sym('pack.core', 'hash-map'),
                             Cons(expanding_quasi_quote(elems, interp), nil)))
        case other:
            return other


def expand_quasi_quotes(form, interp):
    def alg(form):
        match form:
            case Cons(Sym(None, 'quasiquote'), Cons(expanded_subform, Nil())):
                return expanding_quasi_quote(expanded_subform, interp)
            case other:
                return other
    return cata_f(fmap_datum)(alg)(form)


def expand_and_evaluate_forms(forms, interp):

    results = []
    for form in forms:
        form0 = expand_quasi_quotes(form, interp)
        expanded, interp = macroexpand(form0, interp)
        # TODO: validate stage
        defined, interp = create_defs(expanded, interp)
        result, interp = eval_top_level_form(defined, interp)
        results.append(result)
    return results, interp


def main():

    interp = Interpreter()

    interp = load_module(find_module('pack.core'), interp)

    _, interp = expand_and_evaluate_forms(read_all_forms("(ns user)"), interp)

    if os.isatty(sys.stdin.fileno()):

        while True:
            try:
                forms = read_forms(prompt=interp.current_ns.name + '=> ')
            except EOFError:
                print()
                exit(0)
            except (NotImplementedError, SyntaxError) as e:
                print(repr(e))
                continue

            try:
                results, interp = expand_and_evaluate_forms(forms, interp)
                for result in results:
                    print(repr(result))
            except (NotImplementedError, PackLangError) as e:
                print(repr(e))
            except Exception:
                traceback.print_exc()

    else:

        forms = read_all_forms(
            FileString(sys.stdin.read(), file="<stdin>", lineno=1, col=0)
        )
        _, interp = expand_and_evaluate_forms(forms, interp)
