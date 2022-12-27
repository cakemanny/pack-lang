#!/usr/bin/env python3

import dataclasses
import importlib
import os
import sys
import traceback
from collections.abc import Sequence, Mapping
from dataclasses import dataclass
from functools import reduce, partial
from itertools import islice
from typing import Any, Optional, Set, Collection


# ----------------
#  Data Structures
# ----------------


@dataclass(frozen=True, slots=True)
class Sym:
    ns: Optional[str]
    n: str

    def __str__(self):
        if self.ns:
            return self.ns + '/' + self.n
        return self.n

    def __repr__(self):
        return str(self)


@dataclass(frozen=True, slots=True)
class Keyword:
    ns: Optional[str]
    n: str

    def __str__(self):
        if self.ns:
            return f':{self.ns}/{self.n}'
        return f':{self.n}'

    def __repr__(self):
        return str(self)

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
        try:
            xs = nil
            for x in reversed(it):
                xs = Cons(x, xs)
            return xs
        except TypeError:
            # not reversible
            return List.from_iter(tuple(it))


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


@dataclass(frozen=True, slots=True)
class Cons(List):
    hd: Any
    tl: 'Optional[List]'

    def __post_init__(self):
        if self.tl is not None and self.tl is not nil:
            assert isinstance(self.tl, Cons)

    def __iter__(self):
        cons = self
        while cons is not None and cons is not nil:
            yield cons.hd
            cons = cons.tl

    def __reversed__(self):
        result = nil
        for x in self:
            result = Cons(x, result)
        return result

    def __str__(self):
        return '(' + ' '.join(map(str, self)) + ')'

    def __bool__(self):
        return True

    def __len__(self):
        return sum(map(lambda _: 1, self), 0)

    def __add__(self, o):
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


@dataclass
class Vec(Sequence):
    """
    A Trie with at most 32 elements in each node
    """
    xs: tuple[Any | 'Vec']
    height: int

    def __init__(self, xs: list | tuple = (), height=None):
        # TODO: implement a version that works for iterable
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

    def __call__(self, idx: int):
        return self[idx]

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

        class ItemsView:
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
        return cls(())

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

# -------------
#  Constants
# -------------


class Special:
    "A simple namespace for some constants that we use"

    DOT = Sym(None, '.')
    DEF = Sym(None, 'def')
    LETSTAR = Sym(None, 'let*')
    IF = Sym(None, 'if')
    FN = Sym(None, 'fn')
    RAISE = Sym(None, 'raise')
    QUOTE = Sym(None, 'quote')
    VAR = Sym(None, 'var')


# -------------
#  Reader
# -------------


WHITESPACE = (' ', '\n', '\t', '\v')

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
        raise SyntaxError("invalid number literal: multiple '.'s")
    if point == 1:
        return float(prefix + text[:i]), text[i:]
    return int(prefix + text[:i]), text[i:]


def read_str(text):
    assert text[0] == '"'
    text = text[1:]

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
                    raise SyntaxError(f'unknown escape sequence \\{c}')
                seg_start += 2
            i += 1
        if i == n:
            raise Unclosed('"')

    segments = list(aux(text))
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
    "yield pairs from an even length iterable: ABCDEF -> AB CD EF"
    # See https://docs.python.org/3/library/functions.html#zip Tips and tricks
    return zip(it := iter(xs), it, strict=True)


def close_sequence(opener, elements):
    match opener:
        case '(':
            return List.from_iter(elements)
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


def read_quoted_like(text, macro, prefix):
    to_quote, remaining = try_read(text)
    if to_quote is None:
        raise Unclosed(prefix, remaining)
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
    pass


class SyntaxError(PackLangError):
    pass


class Unmatched(SyntaxError):
    "something ended that hadn't begun"


class Unclosed(SyntaxError):
    "something started but never ended"


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
        case ';':
            return read_comment(text)
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
    defs: Map | ArrayMap = ArrayMap.empty()  # str -> var
    aliases: Map | ArrayMap = ArrayMap.empty()  # str -> var

    def apply_defs(self, f):
        return dataclasses.replace(self, defs=f(self.defs))

    def apply_aliases(self, f):
        return dataclasses.replace(self, aliases=f(self.aliases))


@dataclass(frozen=True)
class Interpreter:
    namespaces: Map | ArrayMap  # str -> Namespace

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

        return dataclasses.replace(self, namespaces=namespaces)

    def update_ns(self, ns_name, f):
        updated_ns = f(self.namespaces[ns_name])
        namespaces = self.namespaces.assoc(ns_name, updated_ns)
        if ns_name == namespaces['pack.core'].defs['*ns*'].value.name:
            namespaces['pack.core'].defs['*ns*'].value = updated_ns
        return dataclasses.replace(self, namespaces=namespaces)

    @property
    def default_ns(self):
        return self.namespaces['pack.core']

    @property
    def current_ns(self):
        return self.namespaces['pack.core'].defs['*ns*'].value

    def resolve_symbol(self, sym, env):
        if sym in env:
            if isinstance(env[sym], Var):
                return env[sym]
            raise SemanticError('{sym} is not a var')
        ns_name = sym.ns or self.current_ns.name
        try:
            ns = self.namespaces[ns_name]
        except KeyError:
            raise EvalError(f"no such namespace {ns_name}") from None
        try:
            return ns.defs[sym.n]
        except KeyError:
            raise EvalError(f'{sym!r} not found in this context') from None


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

    def __str__(self):
        return f"#'{self.symbol}"

    def __repr__(self):
        return str(self)


KW_MACRO = Keyword(None, "macro")


def set_macro(var):
    var.metadata = var.metadata.assoc(KW_MACRO, True)


def is_macro(sym, interp, env):
    try:
        var = interp.resolve_symbol(sym, env)
    except PackLangError:
        return False
    try:
        return var.metadata.get(KW_MACRO)
    except AttributeError:
        return False


class Fn:
    def __init__(self, name, params, body, env):
        assert isinstance(params, Vec)
        self.name = name
        self.params, self.restparam = self._split_params(params)
        self.body = body
        self.env = env

    def _split_params(self, params: Vec):
        if Sym(None, '&') not in params:
            return params, None
        idx = params.index(Sym(None, '&'))
        new_params = [params[i] for i in range(0, idx)]
        restparam = params[idx + 1]
        return new_params, restparam

    def __call__(self, interp, *args):
        if not self.restparam and len(args) != len(self.params):
            raise EvalError('incorrect number of arguments')
        env = self.env
        for p, arg in zip(self.params, args):
            env = env.assoc(p, arg)
        if self.restparam:
            restargs = args[len(self.params):]
            env = env.assoc(self.restparam, List.from_iter(restargs))
        return eval_form(self.body, interp, env)

    def __repr__(self):
        return f'<Fn({self.name}) object at {hex(id(self))}>'


def extract_closure(body, params, interp, env):
    # To decide is, whether or not the values from the interp should be
    # captured as they are or not... I think the right answer is to
    # do so...
    # TODO
    m = ArrayMap.empty()  # closure
    # TODO: combine params and lets into bound_vars and use a Set for this
    lets = ArrayMap.empty()

    def aux(expr, m, lets):
        match expr:
            case Sym(None, 'true' | 'false' | 'nil'):
                return m
            # The arguments for if and raise are evaluated, so
            # can just be treated like functions here
            case Sym(None, 'if' | 'raise' | 'var'):
                return m
            case Sym(None, name) as sym:
                # Maybe this is also the point to do more macro expanding?
                if sym in params or sym in lets:
                    return m
                if sym in env:
                    return m.assoc(sym, env[sym])
                if name in interp.current_ns.defs:
                    return m.assoc(sym, interp.current_ns.defs[name])
                if name in interp.current_ns.aliases:
                    return m.assoc(sym, interp.current_ns.aliases[name])
                if name in interp.default_ns.defs:
                    return m.assoc(sym, interp.default_ns.defs[name])
                raise EvalError(
                    f'Could not resolve symbol: {sym!r} in this context'
                )
            # Special forms
            case Cons(Sym(None, '.'), Cons(obj, Cons(_, Nil()))):
                # onlt the obj is evaluated, not the attr
                return aux(obj, m, lets)
            case Cons(Sym(None, 'def'), Cons(Sym(), Cons(init, Nil()))):
                return aux(init, m, lets)
            case Cons(Sym(None, 'fn'),
                      Cons(Sym(), Cons(Vec() as fn_params, Cons(body, Nil())))) \
                    | Cons(Sym(None, 'fn'),
                           Cons(Vec() as fn_params, Cons(body, Nil()))):
                lets = reduce(lambda xs, p: xs.assoc(p, None), fn_params, lets)
                return aux(body, m, lets)
            case Cons(Sym(None, 'quote'), _):
                return m

            case Cons(hd, tl):
                m = aux(hd, m, lets)
                return aux(tl, m, lets)
            case Vec() as vec:
                return reduce(lambda m, sub_form: aux(sub_form, m, lets), vec, m)
            case ArrayMap() | Map() as m:
                for k, v in m.items():
                    m = aux(k, m, lets)
                    m = aux(v, m, lets)
                return m
            case str() | int() | float() | bool() | Nil() | None:
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
        forms = read_all_forms(f.read())
    _, interp = expand_and_evaluate_forms(forms, interp)
    return interp


def eval_form(form, interp, env):
    match form:
        case x if x is nil:
            return nil, interp
        case None | True | False | int() | float() | str() | Keyword():
            return form, interp
        # TODO: change this to in-ns
        case Cons(Sym(None, 'ns'), Cons(Sym(None, name), _)):
            return None, interp.switch_namespace(name)
        case Cons(Sym(None, 'ns'), _):
            raise SemanticError('ns expects a symbol as argument')

        case Cons(Sym(None, 'require'), Cons(name_form, _)):
            name, interp = eval_form(name_form, interp, env)
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
            module = importlib.import_module(name)

            ns_name = interp.current_ns.name
            if name in interp.namespaces[ns_name].aliases:
                var = interp.namespaces[ns_name].aliases[name]
                # !Mutation!
                var.value = module
            else:
                var = Var(sym, module)
                interp = interp.update_ns(
                    ns_name,
                    lambda ns: ns.apply_aliases(
                        lambda aliases: aliases.assoc(name, var)
                    )
                )
            return var, interp

        case Cons(Sym(None, '.'), Cons(obj_expr, Cons(Sym(None, attr), Nil()))):
            obj, interp = eval_form(obj_expr, interp, env)
            return getattr(obj, attr), interp
        case Cons(Sym(None, '.'), _):
            raise SyntaxError(f'invald member expression: {form}')

        case Cons(Sym(None, 'def'), Cons(Sym(ns_name, name), Cons(init, Nil()))):
            init_val, interp = eval_form(init, interp, env)
            var = interp.namespaces[ns_name].defs[name]
            # !Mutation!
            var.value = init_val
            return var, interp

        case Cons(Sym(None, 'let*'), Cons(Vec() as bindings, Cons(body))):
            if len(bindings) % 2 != 0:
                raise SyntaxError('uneven number of forms in let bindings')

            for binding, init in take_pairs(bindings):
                if not isinstance(binding, Sym) or binding.ns is not None:
                    raise SyntaxError(
                        f'let* does not support destructuring: problem: {binding}'
                    )
                init_val, interp = eval_form(init, interp, env)
                env = env.assoc(binding, init_val)
            return eval_form(body, interp, env)
        case Cons(Sym(None, 'let*'), _):
            raise SyntaxError(f"invalid let: {form}")

        case Cons(Sym(None, 'if'),
                  Cons(predicate,
                       Cons(consequent, Cons(alternative, Nil())))):
            p, interp = eval_form(predicate, interp, env)
            if p:
                return eval_form(consequent, interp, env)
            return eval_form(alternative, interp, env)
        case Cons(Sym(None, 'if'), Cons(predicate, Cons(consequent, Nil()))):
            p, interp = eval_form(predicate, interp, env)
            if p:
                return eval_form(consequent, interp, env)
            return None, interp
        case Cons(Sym(None, 'if')):
            raise SyntaxError(f'something is wrong with this if: {form}')

        case Cons(Sym(None, 'fn'), Cons(Vec() as params, Cons(body, Nil()))):
            closure = extract_closure(body, params, interp, env)
            return Fn(None, params, body, closure), interp
        case Cons(Sym(None, 'fn'),
                  Cons(Sym(None, name), Cons(Vec() as params, Cons(body, Nil())))):
            # TODO: create a var to store the name in?
            closure = extract_closure(body, params, interp, env)
            return Fn(name, params, body, closure), interp
        case Cons(Sym(None, 'fn'), _):
            raise SyntaxError(f'invalid fn form: {form}')

        case Cons(Sym(None, 'raise'), Cons(raisable_form, Nil() | None)):
            # FIXME: this loses the effects of the evaluation on the interp
            raisable, interp = eval_form(raisable_form, interp, env)
            raise raisable
        case Cons(Sym(None, 'raise')):
            raise SyntaxError(f"invalid special form 'raise': {form}")

        case Cons(Sym(None, 'quote'), Cons(arg, Nil())):
            return arg, interp
        case Cons(Sym(None, 'quote'), args):
            raise SemanticError(
                f'wrong number of arguments to quote: {len(args)}'
            )

        case Cons(Sym(None, 'var'), Cons(Sym() as sym, Nil() | None)):
            var = interp.resolve_symbol(sym, env)
            return var, interp
        case Cons(Sym(None, 'var'), _):
            raise SemanticError(
                f'invalid special form var: var takes 1 symbol as argument: {form}'
            )

        case Cons(Sym() as sym, args) if is_macro(sym, interp, env):
            # TODO: maybe this is the point we should be calling macroexpand...
            proc, interp = eval_form(sym, interp, env)
            assert isinstance(proc, Fn)
            expanded_form, interp = proc(interp, *list(args))
            return eval_form(expanded_form, interp, env)
        case Cons(proc_form, args):
            proc, interp = eval_form(proc_form, interp, env)

            arg_vals = []
            for arg in args:
                arg_val, interp = eval_form(arg, interp, env)
                arg_vals.append(arg_val)
            # Assume proc is an Fn
            if isinstance(proc, Fn):
                return proc(interp, *arg_vals)
            # python function
            return proc(*arg_vals), interp

        case Sym(None, 'true'):
            return True, interp
        case Sym(None, 'false'):
            return False, interp
        case Sym(None, 'nil'):
            return None, interp

        case Sym(None, name) as sym:
            if sym in env:
                if isinstance(env[sym], Var):
                    return env[sym].value, interp
                return env[sym], interp

            # This is actually broken for fns defined in one module
            # but called in another
            if name in interp.current_ns.defs:
                return interp.current_ns.defs[name].value, interp
            if name in interp.current_ns.aliases:
                return interp.current_ns.aliases[name].value, interp
            # TODO: put stuff from pack.core into aliases
            if name in interp.default_ns.defs:
                return interp.default_ns.defs[name].value, interp
            raise EvalError(f'Could not resolve symbol: {sym} in this context')
        case Sym(ns_name, name) as sym:
            var = interp.resolve_symbol(sym, env)
            if isinstance(var, Var):
                return var.value, interp
            return var, interp

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


# Currently this is doing too much!
# identifying definitions should be done in a step before expanding
# and then form simplifying should come later
def macroexpand_1(form, interp):
    match form:

        case Cons(Sym(None, 'def') as deff, Cons(Sym(_, _) as sym, Nil())):
            return Cons(deff, Cons(sym, Cons(None, nil))), interp
        case Cons(Sym(None, 'def'), Cons(Sym(None, name), Cons(init, _))):
            # Should make a note of whether this is a macro or not

            ns = interp.current_ns.name
            return Cons(Sym(None, 'def'), Cons(Sym(ns, name), Cons(init, nil))), interp
        case Cons(Sym(None, 'def'), Cons(Sym(ns_name, name) as sym, Cons(init, Nil()))):
            try:
                ns = interp.namespaces[ns_name]
            except KeyError:
                raise SemanticError(f'No such namespace {ns_name!r}') from None
            if ns_name != interp.current_ns.name:
                raise SemanticError(
                    'Cannot def symbol outside current ns'
                )

            if name in interp.namespaces[ns_name].defs:
                return form, interp

            return form, interp.update_ns(
                ns_name,
                lambda ns: ns.apply_defs(
                    lambda defs: defs.assoc(name, Var(sym))
                )
            )
        case Cons(Sym(None, 'def'), _):
            raise SemanticError('def expects a symbol as argument')

        # TODO: check if definition is macro
        # case ...
        case other:
            return other, interp


def macroexpand(form, interp):
    while True:
        new_form, interp = macroexpand_1(form, interp)
        if new_form == form:
            return new_form, interp
        form = new_form


def expanding_quasi_quotes(form, interp):
    def quote_form(form):
        return Cons(Sym(None, 'quote'), Cons(form, nil))
    match form:
        case Sym(None, 'ns' | 'require' | 'import'):
            return quote_form(form)
        case Sym(None, '.' | 'def' | 'let*' | 'if' | 'fn' | 'raise' | 'quote' | 'var'):
            return quote_form(form)
        case Sym(None, name) as sym:
            try:
                var = interp.resolve_symbol(sym, ArrayMap.empty())
                if isinstance(var, Var):
                    return quote_form(var.symbol)
                raise NotImplementedError
            except EvalError:
                return quote_form(Sym(interp.current_ns.name, name))
        case Cons(Sym(None, 'unquote'), Cons(x, Nil())):
            return x
        case Nil():
            return nil
        case Cons() as lst:
            def f(form):
                match form:
                    case Cons(Sym(None, 'unquote-splicing'), Cons(x, Nil())):
                        return x
                return Cons(Sym('pack.core', 'list'),
                            Cons(expanding_quasi_quotes(form, interp), nil))
            return Cons(Sym('pack.core', 'concat'),
                        List.from_iter(map(f, lst)))
        # TODO: vectors and maps too
        case other:
            return other


def expand_quasi_quotes(form, interp):
    match form:
        case Cons(Sym(None, 'quasiquote'), Cons(quoted_form, Nil())):
            return expanding_quasi_quotes(quoted_form, interp)
        case Cons() as lst:
            f = partial(expand_quasi_quotes, interp=interp)
            return List.from_iter(map(f, lst))
        # TODO: vectors and maps too
        case other:
            return other


def expand_and_evaluate_forms(forms, interp):

    results = []
    for form in forms:
        form0 = expand_quasi_quotes(form, interp)
        expanded, interp = macroexpand(form0, interp)
        result, interp = eval_form(expanded, interp, Map.empty())
        results.append(result)
    return results, interp


def main():

    interp = Interpreter(Map.empty())

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
                    print(result)
            except (NotImplementedError, PackLangError) as e:
                print(repr(e))
            except Exception:
                traceback.print_exc()

    else:

        forms = read_all_forms(sys.stdin.read())
        _, interp = expand_and_evaluate_forms(forms, interp)


if __name__ == '__main__':
    main()
