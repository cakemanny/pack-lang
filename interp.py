#!/usr/bin/env python3

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any


# -------------
#  Reader
# -------------


WHITESPACE = (' ', '\n', '\t', '\v')


@dataclass
class Ident:
    i: str

    def __str__(self):
        return self.i


@dataclass
class Num:
    n: str

    def __str__(self):
        return self.n


@dataclass
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


@dataclass
class Map:
    xs: tuple[tuple]

    def __str__(self):
        return '{' + '  '.join(
            map(lambda x: ' '.join(map(str, x)), self.xs)
        ) + '}'


def is_ident_start(c):
    return (
        'a' <= c <= 'z'
        or c in ('+', '-', '*', '/', '<', '>', '!', '=', '&', '^')
        or '🌀' <= c <= '🫸'
    )


def is_ident(c):
    return is_ident_start(c) or '0' <= c <= '9'


def read_ident(text):
    i = 0
    for c in text:
        if is_ident(c):
            i += 1
        else:
            break
    return Ident(text[:i]), text[i:]


def read_num(text, prefix=''):
    i = 0
    for c in text:
        if '0' <= c <= '9':
            i += 1
        else:
            break
    return Num(prefix + text[:i]), text[i:]


def take_pairs(xs):
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
                if opener == '(':
                    return List(tuple(elements)), e.args[1]
                if opener == '[':
                    return Vec(elements), e.args[1]
                if opener == '{':
                    try:
                        return Map(tuple(take_pairs(elements))), e.args[1]
                    except ValueError:
                        raise SyntaxError(
                            'A map literal must contain an even number of'
                            ' forms'
                        ) from None

            raise


def read_quoted(text):
    to_quote, remaining = try_read(text)
    if to_quote is None:
        raise Unclosed("'", remaining)
    return List((Ident('quote'), to_quote)), remaining


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
        case '\\':
            # TODO characters
            raise NotImplementedError(c)
        case s if is_ident(s):
            return read_ident(text)

    raise NotImplementedError(c)


def read_forms(previous_lines='', input=input, forms=tuple()):

    line = input('=> ' if not previous_lines else '')
    try:
        form, remaining = try_read(previous_lines + '\n' + line)
    except Unclosed:
        return read_forms(previous_lines + '\n' + line, input, forms)

    if form is not None:
        forms = forms + (form,)
    while remaining:
        try:
            form, remaining = try_read(remaining)
            if form is not None:
                forms = forms + (form,)
        except Unclosed:
            return read_forms(remaining, input, forms)
    return forms


# -------------
#  Interpreter
# -------------


class Package:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent

        self.values = {}


class Interpreter:
    def __init__(self):
        self.packages = []
        self.package = None


def expand_and_evaluate_forms(forms, interpreter):
    # TODO: macro expand
    expanded_forms = forms


def main():

    interpreter = Interpreter()

    if os.isatty(sys.stdin.fileno()):
        while True:
            try:
                forms = read_forms()
            except EOFError:
                print()
                exit(0)
            # TODO: evaluate form
            for form in forms:
                print(form)

        expand_and_evaluate_forms(forms, interpreter)

    else:

        def _input(prompt):
            return sys.stdin.read()

        forms = read_forms(input=_input)
        for form in forms:
            print(form)

        expand_and_evaluate_forms(forms, interpreter)


if __name__ == '__main__':
    main()
