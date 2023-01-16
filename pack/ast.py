"""
pack.ast

This contains functions for some but not all of the in-memory syntax structures.
"""
from functools import reduce
from itertools import islice

from pack.data import Sym, Keyword, Vec, List, Cons, Nil, nil, Map, ArrayMap
from pack.util import take_pairs, untake_pairs


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
    # Add set! ?

    all = {DOT, DO, DEF, LETSTAR, LOOP, RECUR, IF, FN, RAISE, QUOTE, VAR}


def split_params(params: Vec):
    if Fn.PARAM_SEP not in params:
        return params, None
    idx = params.index(Fn.PARAM_SEP)
    new_params = tuple([params[i] for i in range(0, idx)])
    restparam = params[idx + 1]
    return new_params, restparam


class Fn:
    PARAM_SEP = Sym(None, '&')

    def __init__(self, name, params, body, env):
        assert isinstance(params, Vec)
        self.name = name
        self.params, self.restparam = split_params(params)
        self.body = body
        self.env = env

    def __repr__(self):
        name = ''
        if self.name:
            name = f' {self.name}'
        restparam = ''
        if self.restparam:
            restparam = f' & {self.restparam}'
        address = hex(id(self))
        return f'<(fn{name} [{self.params}{restparam}] {self.body}) at {address}>'


# ---------------
#  AST Traversal
# ---------------


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
            raise ValueError(f'def is only allowed at the top level: {expr}')
        case Cons(Sym(None, 'let*' | 'loop') as s,
                  Cons(Vec() as bindings, Cons(body, Nil()))):
            def map_seconds(f, vec):
                for binding, init in take_pairs(vec):
                    yield binding
                    yield f(init)
            new_bindings = Vec.from_iter(map_seconds(f, bindings))
            return Cons(s, Cons(new_bindings, Cons(f(body), nil)))
        case Cons(Sym(None, 'recur') as s, args):
            return Cons(s, List.from_iter(map(f, args)))
        # (if predicate consequent alternative)
        case Cons(Sym(None, 'if') as s, Cons(pred, Cons(con, Cons(alt, Nil())))):
            return Cons(s, Cons(f(pred), Cons(f(con), Cons(f(alt), nil))))
        case Cons(Sym(None, 'if') as s, Cons(pred, Cons(con, Nil()))):
            return Cons(s, Cons(f(pred), Cons(f(con), nil)))
        case Cons(Sym(None, 'fn') as s, Cons(Vec() as params, Cons(body, Nil()))):
            return Cons(s, Cons(params, Cons(f(body), nil)))
        case Cons(Sym(None, 'fn') as s,
                  Cons(Sym(None, _) as n, Cons(Vec() as params, Cons(body, Nil())))):
            return Cons(s, Cons(n, Cons(params, Cons(f(body), nil))))
        case Cons(Sym(None, 'raise') as s, Cons(r, Nil())):
            return Cons(s, Cons(f(r), nil))
        case Cons(Sym(None, 'quote'), Cons(_, Nil())):
            return expr
        case Cons(Sym(None, 'var'), Cons(Sym(), Nil())):
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
                  Cons(Vec() as bindings, Cons(body, Nil()))):
            every_second = islice(bindings, 1, None, 2)
            return plus(reduce(plus, every_second, zero), body)
        case Cons(Sym(None, 'recur'), args):
            return reduce(plus, args, zero)
        case Cons(Sym(None, 'if'),
                  Cons(pred, Cons(consequent, Cons(alternative, Nil())))):
            return plus(plus(pred, consequent), alternative)
        case Cons(Sym(None, 'if'),
                  Cons(pred, Cons(consequent, Nil()))):
            return plus(pred, consequent)
        case Cons(Sym(None, 'fn'), Cons(Vec(), Cons(body, Nil()))):
            return body
        case Cons(Sym(None, 'fn'), Cons(Sym(None, _), Cons(Vec(), Cons(body, Nil())))):
            return body
        case Cons(Sym(None, 'raise'), Cons(r, Nil())):
            return r
        case Cons(Sym(None, 'quote'), Cons(_, Nil())):
            return zero
        case Cons(Sym(None, 'var'), Cons(Sym(), Nil())):
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
