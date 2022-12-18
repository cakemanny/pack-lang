#!/usr/bin/env python3

import os
import sys
from collections.abc import Sequence
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
class Map:
    xs: tuple[tuple]

    def __str__(self):
        return '{' + '  '.join(
            map(lambda x: ' '.join(map(str, x)), self.xs)
        ) + '}'


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
                return Map(tuple(take_pairs(elements)))
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
            # TODO: evaluate form
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
