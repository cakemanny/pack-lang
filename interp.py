#!/usr/bin/env python3

import sys
import os
from dataclasses import dataclass


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


@dataclass
class Vec:
    xs: tuple

    def __str__(self):
        return '[' + ' '.join(map(str, self.xs)) + ']'


@dataclass
class Hash:
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
                    return Vec(tuple(elements)), e.args[1]
                if opener == '{':
                    try:
                        return Hash(tuple(take_pairs(elements))), e.args[1]
                    except ValueError:
                        raise SyntaxError(
                            'A map literal must contain an even number of'
                            ' forms'
                        ) from None

            raise


def read_quoted(text):
    to_quote, remaining = try_read(text)
    if not to_quote:
        raise Unclosed("'", remaining)
    return List((Ident('quote'), to_quote)), remaining


class SyntaxError(Exception):
    pass


class Unmatched(SyntaxError):
    pass


class Unclosed(SyntaxError):
    pass


def try_read(text):

    if not text:
        return None, text
    c = text[0]

    # eat whitespace
    while c in WHITESPACE:
        text = text[1:]
        if not text:
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

    if form:
        forms = forms + (form,)
    while remaining:
        try:
            form, remaining = try_read(remaining)
            if form:
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


def main():

    interpreter = Interpreter()

    if os.isatty(sys.stdin.fileno()):
        while True:
            forms = read_forms()
            # TODO: evaluate form
            for form in forms:
                print(form)

            # TODO: macro expand
            expanded_forms = forms

            match form:
                case List((Ident(i), s,)):
                    print('EVAL!')
    else:

        def _input(prompt):
            return sys.stdin.read()

        forms = read_forms(input=_input)
        # TODO: evaluate form
        for form in forms:
            print(form)

        # TODO: macro expand
        expanded_forms = forms

        match form:
            case List((Ident(i), s,)):
                print('EVAL!')
        # TODO
        # Read all of stdin at once
        pass


if __name__ == '__main__':
    main()
