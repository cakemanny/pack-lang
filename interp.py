#!/usr/bin/env python3

import dataclasses
import os
import sys
from collections.abc import Sequence, Mapping
from dataclasses import dataclass
from itertools import islice
from typing import Any, Optional


# -------------
#  Reader
# -------------


WHITESPACE = (' ', '\n', '\t', '\v')


@dataclass(frozen=True)
class Sym:
    ns: Optional[str]
    n: str

    def __str__(self):
        if self.ns:
            return self.ns + '/' + self.n
        return self.n


@dataclass(frozen=True)
class Num:
    n: str

    def __str__(self):
        return self.n


@dataclass(frozen=True)
class List:
    xs: tuple

    def __str__(self):
        return '(' + ' '.join(map(str, self.xs)) + ')'


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


@dataclass
class Vec(Sequence):
    """
    A Trie with at most 32 elements in each node
    """
    xs: tuple[Any | 'Vec']
    height: int

    def __init__(self, xs: list | tuple, height=None):
        # Would be nice to implement a version that works for iterable
        self._len = len(xs)

        if height is None:
            height = 0
            len_ = len(xs)

            for i in range(9):
                if len_ > (1 << (5 * i)):
                    height = i
                else:
                    break

        self.height = height

        if height == 0:
            self.xs = tuple(xs)
        else:
            batch_size = 1 << (5 * height)
            self.xs = tuple(
                Vec(teil, self.height - 1) for teil in batched(xs, batch_size)
            )

    def is_leaf(self):
        return self.height == 0

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        if idx < 0:
            return self[self._len + idx]
        if idx >= self._len:
            raise IndexError('vector index out of range')

        if self.is_leaf():
            return self.xs[idx]

        subvec_idx = idx >> (5 * self.height)

        mask = (1 << (5 * self.height)) - 1

        return self.xs[subvec_idx][mask & idx]

    def __str__(self):
        return '[' + ' '.join(map(str, self)) + ']'

    def __repr__(self):
        if self.is_leaf():
            return str(self)
        return '[' + ' '.join(map(str, (
            self[0], self[1], self[2],
            '...',
            self[-3], self[-2], self[-1],
        ))) + ']'


@dataclass(frozen=True)
class ArrayMap(Mapping):
    kvs: tuple

    def __len__(self):
        return len(self.kvs) // 2

    def __iter__(self):
        def iterate(kvs):
            for i in range(0, len(kvs), 2):
                yield kvs[i]
        return iterate(self.kvs)

    def __getitem__(self, k):
        kvs = self.kvs
        for i in range(0, len(kvs), 2):
            if k == kvs[i]:
                return kvs[i + 1]
        raise KeyError(k)

    def __eq__(self, o):
        if not isinstance(o, Mapping):
            return False
        if len(o) != len(self):
            return False
        # inefficient nested loop join
        try:
            for k in self:
                if self[k] != o[k]:
                    return False
            for k in o:
                if self[k] != o[k]:
                    return False
        except KeyError:
            return False
        return True

    def __str__(self):
        return '{' + '  '.join(
           f'{k!r} {v!r}' for (k, v) in self.items()
        ) + '}'

    @classmethod
    def from_iter(cls, it):
        # NB: This is not dealing with duplicates currently
        def aux():
            for k, v in it:
                yield k
                yield v
        return cls(tuple(aux()))


@dataclass
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

    def __getitem__(self, k):
        h = hash(k) & ((1 << 32) - 1)
        return self._getitem_for_hash_and_key(h, k)

    def _getitem_for_hash_and_key(self, h, k):

        idx = (h >> (self.height * 5)) & 0b11111

        if self._is_leaf(idx):
            entry = self.xs[idx]
            if entry is None or entry[0] != k:
                raise KeyError(k)
            return entry[1]

        next_map = self.xs[idx]
        if next_map is None:
            raise KeyError(k)
        return next_map._getitem_for_hash_and_key(h, k)

    def __iter__(self):
        for i, x in enumerate(self.xs):
            if (self.kindset & (1 << i)):
                yield x[0]
            elif x is not None:
                for k in x:
                    yield k

    def _hash(self, k):
        return hash(k) & ((1 << 32) - 1)

    def _is_leaf(self, idx):
        return bool(self.kindset & (1 << idx))

    def assoc(self, k, v):
        h = self._hash(k)

        idx = (h >> (self.height * 5)) & 0b11111
        if self._is_leaf(idx):
            entry = self.xs[idx]
            if entry is not None and entry == (k, v):
                return self
            if entry is not None and entry[0] != k:
                existing_h = self._hash(entry[0])
                if h == existing_h:
                    raise NotImplementedError('hash collision')
                # If has is the same: need to append to the bucket
                # otherwise: need to create new subnode

                new_value = (
                    dataclasses.replace(
                        Map.empty(), height=(self.height - 1)
                    )
                    .assoc(entry[0], entry[1])
                    .assoc(k, v)
                )
                return Map(
                    tuple(
                        new_value if i == idx else x
                        for (i, x) in enumerate(self.xs)
                    ),
                    kindset=(self.kindset & (((1 << 32) - 1) ^ (1 << idx))),
                    _len=(self._len + 1),
                    height=self.height,
                )

            # This will break when we have to bucket things
            assert entry[0] == k
            # Replace a key
            return Map(
                tuple(
                    (k, v) if i == idx else x for (i, x) in enumerate(self.xs)
                ),
                kindset=(self.kindset | (1 << idx)),
                _len=self._len,
                height=self.height,
            )
        next_map = self.xs[idx]
        if next_map is None:
            # put (k, v) as a entry in this node
            return Map(
                xs=tuple(
                    (k, v) if i == idx else x for (i, x) in enumerate(self.xs)
                ),
                kindset=(self.kindset | (1 << idx)),
                _len=self._len + 1,
                height=self.height,
            )
        # assoc within submap and return that stuck into place in this one

        new_value = next_map.assoc(k, v)
        return Map(
            tuple(
                new_value if i == idx else x
                for (i, x) in enumerate(self.xs)
            ),
            kindset=self.kindset,
            _len=self._len - len(next_map) + len(new_value),
            height=self.height,
        )

    @classmethod
    def empty(cls):
        if not hasattr(cls, '_empty'):
            cls._empty = cls(
                tuple([None] * 32), kindset=0, _len=0, height=7
            )
        return cls._empty


def is_ident_start(c):
    return (
        'a' <= c <= 'z'
        or c in ('+', '-', '*', '/', '<', '>', '!', '=', '&', '^', '.')
        or '🌀' <= c <= '🫸'
    )


def is_ident(c):
    return is_ident_start(c) or '0' <= c <= '9'


def split_ident(n):
    if '/' in n[:-1]:
        match n.split('/', 1):
            case [ns, n]:
                return ns, n
            case [n]:
                return None, n
    return None, n


def read_ident(text):
    i = 0
    for c in text:
        if is_ident(c):
            i += 1
        else:
            break

    return Sym(*split_ident(text[:i])), text[i:]


def read_num(text, prefix=''):
    i = 0
    for c in text:
        if '0' <= c <= '9':
            i += 1
        else:
            break
    return Num(prefix + text[:i]), text[i:]


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
    "yield pairs from an even length iterable: ABCDEF -> AB CD EF"
    i = 0
    a = None
    for x in xs:
        if i == 0:
            a = x
            i = 1
        else:
            yield (a, x)
            i = 0
    if i != 0:
        raise ValueError('odd length input')


def close_sequence(opener, elements):
    match opener:
        case '(':
            return List(tuple(elements))
        case '[':
            return Vec(elements)
        case '{':
            try:
                return ArrayMap.from_iter(take_pairs(elements))
            except ValueError:
                raise SyntaxError(
                    'A map literal must contain an even number of forms'
                ) from None
    raise ValueError('unknown opener', opener)


def read_list(opener, text, closing):
    elements = []
    while True:
        try:
            elem, text = try_read(text)
            if elem is None:
                raise Unclosed(opener, text)
            elements.append(elem)
        except Unmatched as e:
            if e.args[0] == closing:
                return close_sequence(opener, elements), e.args[1]
            raise


def read_quoted(text):
    to_quote, remaining = try_read(text)
    if to_quote is None:
        raise Unclosed("'", remaining)
    return List((Sym(None, 'quote'), to_quote)), remaining


class SyntaxError(Exception):
    pass


class Unmatched(SyntaxError):
    pass


class Unclosed(SyntaxError):
    pass


def try_read(text):

    if text == '':
        return None, text
    c = text[0]

    # eat whitespace
    while c in WHITESPACE:
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
            raise Unmatched(c, text[1:])
        case "'":
            # quote next form
            return read_quoted(text[1:])
        case '-' | '+' if '0' <= c1 <= '9':
            return read_num(text[1:], c)
        case n if '0' <= n <= '9':
            return read_num(text)
        case '"':
            # TODO: strings
            raise NotImplementedError(c)
        case ';':
            return read_comment(text)
        case '\\':
            # TODO characters
            raise NotImplementedError(c)

        case s if is_ident(s):
            return read_ident(text)

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

    remaining = previous_lines + '\n' + line
    try:
        return read_all_forms(remaining)
    except Unclosed:
        return read_forms(remaining, input, prompt)


# -------------
#  Interpreter
# -------------

class SemanticError(Exception):
    pass


class Namespace:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.defs = {}


class Interpreter:
    def __init__(self):
        self.namespaces = {}
        self.current_ns = None

    def switch_namespace(self, name):
        if name not in self.namespaces:
            ns = Namespace(name)
        self.current_ns = ns


def expand_and_evaluate_forms(forms, interpreter):
    # TODO: macro expand

    # Expands forms...
    for form in forms:
        match form:
            case List((Sym(None, 'ns'), Sym(None, name), *_)):
                interpreter.switch_namespace(name)
            case List((Sym(None, 'ns'), *_)):
                raise SemanticError('ns expects a symbol as argument')
            case List((Sym(None, 'def'), Sym(None, name), *args)):
                interpreter.current_ns.defs[name] = {
                    'form': args
                }
            case List((Sym(None, 'def'), Sym(_, name), *args)):
                # TODO
                pass
            case List((Sym('def'), *_)):
                raise SemanticError('def expects a symbol as argument')
            case other:
                raise NotImplementedError(other)

    # TODO: Evaluate forms


def main():

    interpreter = Interpreter()

    with open('core.pack') as f:
        forms = read_all_forms(f.read())
    expand_and_evaluate_forms(forms, interpreter)

    interpreter.switch_namespace('user')

    if os.isatty(sys.stdin.fileno()):

        while True:
            try:
                forms = read_forms(prompt=interpreter.current_ns.name + '=> ')
            except EOFError:
                print()
                exit(0)
            except SyntaxError as e:
                print(repr(e))
                continue

            for form in forms:
                print(form)

            try:
                expand_and_evaluate_forms(forms, interpreter)
            except NotImplementedError as e:
                print(repr(e))

    else:

        forms = read_all_forms(sys.stdin.read())
        for form in forms:
            print(form)

        expand_and_evaluate_forms(forms, interpreter)


if __name__ == '__main__':
    main()
