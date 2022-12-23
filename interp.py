#!/usr/bin/env python3

import dataclasses
import os
import sys
from collections.abc import Sequence, Mapping, Set
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
        kvs = self.kvs
        for i in range(0, len(kvs), 2):
            yield kvs[i]

    def __getitem__(self, key):
        kvs = self.kvs
        for i in range(0, len(kvs), 2):
            k = kvs[i]
            if key is k or key == k:
                return kvs[i + 1]
        raise KeyError(key)

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

    def items(self):
        kvs = self.kvs

        class ItemsView(Set):
            def __iter__(self):
                for i in range(0, len(kvs), 2):
                    yield (kvs[i], kvs[i + 1])

            def __len__(self):
                return len(kvs) // 2

            def __contains__(self, item):
                for i in range(0, len(kvs), 2):
                    if item == (kvs[i], kvs[i + 1]):
                        return True
                return False

        return ItemsView()

    def values(self):
        kvs = self.kvs
        # Could be rewritten to return a view
        return [kvs[i + 1] for i in range(0, len(kvs), 2)]

    def __str__(self):
        return '{' + '  '.join(
           f'{k!r} {v!r}' for (k, v) in self.items()
        ) + '}'

    def __repr__(self):
        return str(self)

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

    @classmethod
    def empty(cls):
        return cls(tuple())

    @classmethod
    def from_iter(cls, it):
        # NB: This is not dealing with duplicates currently
        def aux():
            for k, v in it:
                yield k
                yield v
        return cls(tuple(aux()))


@dataclass(frozen=True)
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
            if (self.kindset & (1 << i)):
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

    def __str__(self):
        return '{' + '  '.join(
           f'{k!r} {v!r}' for (k, v) in self.items()
        ) + '}'

    def __repr__(self):
        return str(self)

    @classmethod
    def empty(cls):
        if not hasattr(cls, '_empty'):
            cls._empty = cls(
                tuple([None] * 32), kindset=0, _len=0, height=7
            )
        return cls._empty

    @classmethod
    def from_iter(self, it):
        m = Map.empty()
        for k, v in it:
            m = m.assoc(k, v)
        return m


def is_ident_start(c):
    return (
        'a' <= c <= 'z'
        or c in ('+', '-', '*', '/', '<', '>', '!', '=', '&', '^', '.', '?')
        or '🌀' <= c <= '🫸'
    )


def is_ident(c):
    return is_ident_start(c) or (c in ("'")) or '0' <= c <= '9'


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
    point = 0
    for c in text:
        if c == '.':
            point += 1
        if '0' <= c <= '9' or c == '.':
            i += 1
        else:
            break
    if point > 1:
        raise SyntaxError("invalid number literal: multiple '.'s")
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
                return Map.from_iter(take_pairs(elements))
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


class PackLangError(Exception):
    pass


class SyntaxError(PackLangError):
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
        case ':':
            # TODO: keywords
            raise NotImplementedError(c)
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

class SemanticError(PackLangError):
    pass


class EvalError(PackLangError):
    pass


@dataclass(frozen=True)
class Namespace:
    name: str
    defs: Mapping = ArrayMap.empty()

    def apply_defs(self, f):
        return Namespace(self.name, f(self.defs))


@dataclass(frozen=True)
class Interpreter:
    namespaces: Mapping  # str -> Namespace

    def switch_namespace(self, name: str):
        assert isinstance(name, str)
        if name not in self.namespaces:
            ns = Namespace(name)
        else:
            ns = self.namespaces[name]

        namespaces = self.namespaces.assoc(name, ns)
        if '*ns*' not in namespaces['pack.core'].defs:
            core = namespaces['pack.core'].apply_defs(
                lambda defs: defs.assoc('*ns*', Var(Sym('pack.core', '*ns*')))
            )
            namespaces = namespaces.assoc('pack.core', core)

        # Mutation!
        namespaces['pack.core'].defs['*ns*'].value = ns

        return Interpreter(namespaces)

    @property
    def current_ns(self):
        return self.namespaces['pack.core'].defs['*ns*'].value


@dataclass
class Var:
    "This is the mutable thing"
    symbol: Sym
    value: Any

    def __init__(self, symbol, value=None):
        assert isinstance(symbol, Sym)
        self.symbol = symbol
        self.value = value

    def __str__(self):
        return f"#'{self.symbol}"

    def __repr__(self):
        return str(self)


class Fn:
    def __init__(self, name, params, body, env):
        assert isinstance(params, Vec)
        self.name = name
        self.params = params
        self.body = body
        self.env = env

    def __call__(self, interp, *args):
        if len(args) != len(self.params):
            raise EvalError('incorrect number of arguments')
        env = self.env
        for p, arg in zip(self.params, args):
            env = env.assoc(p, arg)
        return eval_form(self.body, interp, env)

    def __repr__(self):
        return f'<Fn({self.name}) object at {hex(id(self))}>'


def extract_closure(body, params, interp, env):
    # To decide is, whether or not the values from the interp should be
    # captured as they are or not... I think the right answer is to
    # do so...
    # TODO
    return Map.empty()


def eval_form(form, interp, env):
    match form:
        case None | True | False:
            return form, interp
        case List((Sym(None, 'ns'), Sym(None, name), *_)):
            return None, interp.switch_namespace(name)
        case List((Sym(None, 'ns'), *_)):
            raise SemanticError('ns expects a symbol as argument')

        case List((Sym(None, 'def'), Sym(ns_name, name), init)):
            init_val, interp = eval_form(init, interp, env)
            var = interp.namespaces[ns_name].defs[name]
            # !Mutation!
            var.value = init_val
            return var, interp
        case List((Sym(None, 'if'), predicate, consequent, alternative)):
            p, interp = eval_form(predicate, interp, env)
            if p:
                return eval_form(consequent, interp, env)
            return eval_form(alternative, interp, env)
        case List((Sym(None, 'fn'), params, body)):
            closure = extract_closure(body, params, interp, env)
            return Fn(None, params, body, closure), interp
        case List((Sym(None, 'fn'), name, params, body)):
            # TODO: create a var to store the name in?
            closure = extract_closure(body, params, interp, env)
            return Fn(name, params, body, closure), interp
        case List((Sym(None, 'quote'), arg)):
            return arg, interp
        case List((Sym(None, 'quote'), *args)):
            raise SemanticError(
                f'wrong number of arguments to quote: {len(args)}'
            )
        case Num(n):
            if '.' in n:
                return float(n), interp
            return int(n), interp
        case Sym(ns_name, name) if ns_name is not None:
            var = interp.namespaces[ns_name].defs[name]
            assert isinstance(var, Var)
            return var.value, interp
        case Sym(None, 'true'):
            return True, interp
        case Sym(None, 'false'):
            return False, interp
        case Sym(None, 'nil'):
            return None, interp
        case Sym(None, name) as sym:
            if sym in env:
                return env[sym], interp
            if name in interp.current_ns.defs:
                return interp.current_ns.defs[name].value, interp
            raise EvalError(f'Could not resolve {sym} in this context')
        case List((proc_form, *args)):
            proc, interp = eval_form(proc_form, interp, env)

            arg_vals = []
            for arg in args:
                arg_val, interp = eval_form(arg, interp, env)
                arg_vals.append(arg_val)
            # Assume proc is an Fn
            return proc(interp, *arg_vals)
        case ArrayMap() | Map() as m:
            result_m = ArrayMap.empty()
            for k, v in m.items():
                k_val, interp = eval_form(k, interp, env)
                v_val, interp = eval_form(v, interp, env)
                result_m = result_m.assoc(k_val, v_val)
            return result_m, interp
        case Vec() as vec:
            values = []
            for sub_form in vec:
                value, interp = eval_form(sub_form, interp, env)
                values.append(value)
            return Vec(values), interp

    raise NotImplementedError(form)


def macroexpand_1(form, interp):
    match form:
        case None:
            return None, interp
        case List((Sym(None, 'ns'), Sym(None, name), *_)):
            return form, interp.switch_namespace(name)
        case List((Sym(None, 'ns'), *_)):
            raise SemanticError('ns expects a symbol as argument')

        case List((Sym(None, 'def') as deff, Sym(_, _) as sym)):
            return List((deff, sym, None)), interp
        case List((Sym(None, 'def'), Sym(None, name), init)):
            # Should make a note of whether this is a macro or not

            ns = interp.current_ns.name
            return List((Sym(None, 'def'), Sym(ns, name), init)), interp
        case List((Sym(None, 'def'), Sym(ns_name, name) as sym, init)):
            try:
                ns = interp.namespaces[ns_name]
            except KeyError:
                raise SemanticError(f'No such namespace {ns_name!r}')
            if ns_name != interp.current_ns.name:
                raise SemanticError(
                    'Cannot def symbol outside current ns'
                )

            return form, Interpreter(
                interp.namespaces.assoc(
                    ns_name, interp.namespaces[ns_name].apply_defs(
                        lambda defs: defs.assoc(name, Var(sym))
                    )
                )
            )
        case List((Sym('def'), *_)):
            raise SemanticError('def expects a symbol as argument')

        case List((Sym(None, 'if'), p, c)):
            return List((Sym(None, 'if'), p, c, None)), interp
        case List((Sym(None, 'if'), _, _, _) | (Sym(None, 'if'), _, _)):
            return form, interp
        case List((Sym(None, 'if'), *_)):
            raise SyntaxError('invalid special form if')
        # case ...
        case other:
            return other, interp


def macroexpand(form, interp):
    while True:
        new_form, interp = macroexpand_1(form, interp)
        if new_form == form:
            return new_form, interp
        form = new_form


def expand_and_evaluate_forms(forms, interp):
    # TODO: macro expand

    # Expands forms...
    expanded_forms = []
    for form in forms:
        expanded, interp = macroexpand(form, interp)
        expanded_forms.append(expanded)

    # TODO: Evaluate forms
    results = []
    for form in expanded_forms:
        result, interp = eval_form(form, interp, Map.empty())
        results.append(result)
    return results, interp


def main():

    interp = Interpreter(Map.empty())

    with open('core.pack') as f:
        forms = read_all_forms(f.read())
    _, interp = expand_and_evaluate_forms(forms, interp)

    interp = interp.switch_namespace('user')

    if os.isatty(sys.stdin.fileno()):

        while True:
            try:
                forms = read_forms(prompt=interp.current_ns.name + '=> ')
            except EOFError:
                print()
                exit(0)
            except SyntaxError as e:
                print(repr(e))
                continue

            for form in forms:
                print(form)

            try:
                results, interp = expand_and_evaluate_forms(forms, interp)
                for result in results:
                    print(result)
            except (NotImplementedError, SemanticError, EvalError) as e:
                print(repr(e))

    else:

        forms = read_all_forms(sys.stdin.read())
        _, interp = expand_and_evaluate_forms(forms, interp)


if __name__ == '__main__':
    main()
