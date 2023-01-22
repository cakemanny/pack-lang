from dataclasses import dataclass
from functools import reduce
from typing import TypeVar, Optional
import keyword
import operator
import traceback

from pack.ast import Special, fmap, fmap_datum, reduce_expr, split_params
from pack.ast import Fn as InterpFn
from pack.data import Sym, Keyword, Vec, List, Cons, Nil, nil, Map, ArrayMap
from pack.reader import location_from
from pack.recursion import cata_f, ana_f, zygo_f, compose
from pack.runtime import Var
from pack.util import take_pairs, untake_pairs


# --------------
# Compiler
# --------------


class IR1:
    SETBANG = Sym('pack.core', 'set!')
    WHILE_TRUE = Sym('pack.core', 'while-true')
    BREAK = Sym('pack.core', 'break')
    CONTINUE = Sym('pack.core', 'continue')
    # This does not really occur in IR1, but it's useful for
    # being able to conver from, to make tests more readable
    IF_STMT = Sym('pack.core', 'if-stmt')


def fmap_setbang(f, expr):
    match expr:
        case Cons(IR1.SETBANG, Cons(name_sym, Cons(init, Nil()))):
            return Cons(IR1.SETBANG, Cons(name_sym, Cons(f(init), Nil())))
        case Cons(Sym(None, 'let*'), _):
            raise ValueError('unexpected let*')
        case Cons(IR1.WHILE_TRUE, Cons(action, Nil())):
            return Cons(IR1.WHILE_TRUE, Cons(f(action), nil))
        case Cons(IR1.BREAK | IR1.CONTINUE, Nil()):
            return expr
        case Cons(IR1.IF_STMT, Cons(pred, Cons(con, Cons(alt, Nil())))):
            return Cons(IR1.IF_STMT, Cons(f(pred), Cons(f(con), Cons(f(alt), nil))))
        case other:
            return fmap(f, other)


def reduce_expr_setbang(zero, plus, expr):
    match expr:
        case Cons(IR1.SETBANG, Cons(_, Cons(init, Nil()))):
            return init
        case Cons(IR1.WHILE_TRUE, Cons(action, Nil())):
            return action
        case Cons(IR1.BREAK | IR1.CONTINUE, Nil()):
            return zero
        case Cons(IR1.IF_STMT, Cons(pred, Cons(con, Cons(alt, Nil())))):
            return plus(plus(pred, con), alt)
        case other:
            return reduce_expr(zero, plus, other)


def fmap_tail(f, expr):
    """
    A variant of fmap that only recurses into tail positions.
    """
    match expr:
        case Cons(Special.DO as s, args):
            *non_tail, tail = args  # can this be put up into the thing?
            return Cons(s, List.from_iter(non_tail) + Cons(f(tail), nil))
        case Cons(Special.LETSTAR as s, Cons(Vec() as bnds, Cons(body, Nil()))):
            return Cons(s, Cons(bnds, Cons(f(body), nil)))
        case Cons(Special.IF as s, Cons(pred, Cons(con, Cons(alt, Nil())))):
            return Cons(s, Cons(pred, Cons(f(con), Cons(f(alt), nil))))
        case Cons(Special.IF as s, Cons(pred, Cons(con, Nil()))):
            return Cons(s, Cons(pred, Cons(f(con), nil)))
        case other:
            return other


def reduce_expr_tail(zero, plus, expr):
    match expr:
        case Cons(Special.DO, args):
            *non_tail, tail = args  # can this be put up into the thing?
            return tail
        case Cons(Special.LETSTAR, Cons(Vec(), Cons(body, Nil()))):
            return body
        case Cons(Special.IF, Cons(_, Cons(con, Cons(alt, Nil())))):
            return plus(con, alt)
        case Cons(Special.IF, Cons(_, Cons(con, Nil()))):
            return con
        case _:
            return zero


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
    remove complex quoted data by replacing literals with their
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


def remove_complex_quote_alg(expr):
    """
    an algebra to find quoted data within our program and rewrite it
    """
    match expr:
        case Cons(Sym(None, 'quote'), Cons(datum, Nil())):
            return remove_quote(datum)
        case _:
            return expr


def create_deduce_scope_coalg(i=0):
    """
    Replace all bound variable names with ones with IDs appended
    so that we no longer have to worry about scoping
    """
    def genid():
        nonlocal i
        i += 1
        return i

    def mk_new_sym(sym):
        assert sym.ns is None
        return Sym(None, f'{sym.n}.{genid()}')

    def number_params(new_params_and_mapping, p):
        "to be used in a reduce"
        new_params, m = new_params_and_mapping  # Vec, Map
        if p == InterpFn.PARAM_SEP:
            return new_params.conj(p), m
        else:
            new_p = mk_new_sym(p)  # !
            return new_params.conj(new_p), m.assoc(p, new_p)

    def deduce_scope_coalg(expr_and_new_bnds):
        """
        This algebra takes a form and threads through a mapping of
        old names to new names

        (ExprF[Expr], Map) -> ExprF[(Expr, Map)]
        """
        match expr_and_new_bnds:
            case (Cons(Sym(None, 'let*' | 'loop') as s,
                       Cons(Vec() as bindings, Cons(body, Nil()))),
                  m):
                new_bindings = []
                for name, init in take_pairs(bindings):
                    new_name = mk_new_sym(name)  # !
                    new_bindings += [new_name, (init, m)]
                    m = m.assoc(name, new_name)
                return Cons(s,
                            Cons(Vec.from_iter(new_bindings),
                                 Cons((body, m), nil)))
            case (Cons(Sym(None, 'fn') as s,
                       Cons(Sym() as fn_name_sym,
                            Cons(Vec() as fn_params, Cons(body, Nil())))),
                  m):
                new_fn_name = mk_new_sym(fn_name_sym)
                m = m.assoc(fn_name_sym, new_fn_name)
                new_params, m = reduce(number_params, fn_params, (Vec.empty(), m))
                return Cons(s,
                            Cons(new_fn_name,
                                 Cons(new_params,
                                      Cons((body, m), nil))))
            case (Cons(Sym(None, 'fn') as s,
                       Cons(Vec() as fn_params, Cons(body, Nil()))),
                  m):
                new_params, m = reduce(number_params, fn_params, (Vec.empty(), m))
                return Cons(s, Cons(Vec.from_iter(new_params), Cons((body, m), nil)))
            case (Sym(None, _) as sym,
                  m):
                if sym in m:
                    return m[sym]
                return sym
            # thread the new bindings into the rest of the structure
            case (other, m):
                return fmap(lambda x: (x, m), other)
        raise NotImplementedError(expr_and_new_bnds)

    return deduce_scope_coalg


def create_hoist_lambda_alg(i=0):
    """
    We hoist lambda forms into let* bindings so that we can
    write them as defs

    (fn [x] ...) -> (let* [t.1 (fn [x] ...)]
                      t.1)
    ->

    def t_DOT_1(x):
        ...
    return t_DOT_1
    """
    def next_temp():
        nonlocal i
        i += 1
        return Sym(None, f'__t.{i}')

    def hoist_lambda_alg(expr):
        match expr:
            case Cons(Sym(None, 'fn'), _) as fn:
                t = next_temp()
                return Cons(Sym(None, 'let*'),
                            Cons(Vec.from_iter([t, fn]),
                                 Cons(t, nil)))
            case other:
                return other
    return hoist_lambda_alg


def replace_letstar_alg(expr):
    """
    replace let* bindings with 'set!' assignment
    (let* [x 5 y 9] ...) -> (do (set! x 5) (set! y 9) ...)
    """
    match expr:
        case Cons(Sym(None, 'let*'),
                  Cons(Vec() as bindings,
                       Cons(body, Nil()))):
            set_bang_forms = [
                Cons(IR1.SETBANG, Cons(name_sym, Cons(init, nil)))
                for name_sym, init in take_pairs(bindings)
            ]
            return Cons(Sym(None, 'do'),
                        List.from_iter(set_bang_forms + [body]))
        case other:
            return other


# In the future it might be nice to have a look at these kind of attributes and
# see if it is possible to calculate them earlier, in an initial pass of the
# tree.
#
def contains_recur_alg(expr):
    match expr:
        case Cons(Sym(None, 'recur'), _):
            return True
        case other:
            return reduce_expr_tail(
                zero=False, plus=operator.or_, expr=other
            )


# by using fmap_tail, we won't accidently traverse past
# deeper recurs, or into other fns
contains_recur = cata_f(fmap_tail)(contains_recur_alg)


def nest_loop_in_body_of_recursive_fn(params, body):
    """
    This is a variation on the function beneath. It exists because we
    threw away the outer fn form in compile_fn :(
    """
    if contains_recur(body):
        return Cons(Sym(None, 'loop'),
                    Cons(Vec.from_iter(untake_pairs(zip(params, params))),
                         Cons(body, nil)))
    return body


def nest_loop_in_recursive_fn_alg(expr):
    match expr:
        case Cons(Sym(None, 'fn'),
                  Cons(params, Cons(body, Nil()))) if contains_recur(body):
            loop = Cons(Sym(None, 'loop'),
                        Cons(Vec.from_iter(untake_pairs(zip(params, params))),
                             Cons(body, nil)))
            return Cons(Sym(None, 'fn'),
                        Cons(params,
                             Cons(loop, nil)))
        case other:
            return other


def create_replace_loop_recur_alg(i=0):
    def next_temp(prefix=""):
        nonlocal i
        i += 1
        return Sym(None, f'{prefix}__t.{i}')

    def replace_tails_alg(binding_names, return_value):
        def alg(expr):
            match expr:
                case Cons(Special.RECUR, args):
                    # We have to separate out evaluation and rebinding
                    # because the expressions may refer to each other
                    temp_names = list(map(next_temp, binding_names))
                    evaluation = [
                        Cons(IR1.SETBANG, Cons(name_sym, Cons(init, nil)))
                        for name_sym, init in zip(temp_names, args)
                    ]
                    rebinding = [
                        Cons(IR1.SETBANG, Cons(name_sym, Cons(init, nil)))
                        for name_sym, init in zip(binding_names, temp_names)
                    ]
                    continue_ = Cons(IR1.CONTINUE, nil)
                    return Cons(Special.DO,
                                List.from_iter(evaluation + rebinding + [continue_]))

                # Cases where the forms contain deeper tail expressions
                case Cons(Special.DO | Special.LETSTAR | Special.IF, _):
                    # recur and other tail expressions have been replaced
                    # thanks to fmap_tail
                    return expr

                # Tail Reached: replace the result with a set!
                case other:
                    eval_and_assign = Cons(IR1.SETBANG,
                                           Cons(return_value,
                                                Cons(other, nil)))
                    break_ = Cons(IR1.BREAK, nil)
                    return Cons(Special.DO,
                                Cons(eval_and_assign, Cons(break_, nil)))
        return alg

    def replace_loop_recur_alg(expr):
        match expr:
            case Cons(Sym(None, 'loop'),
                      Cons(Vec() as bindings,
                           Cons(body, Nil()))):
                set_bang_forms = [
                    Cons(IR1.SETBANG, Cons(name_sym, Cons(init, nil)))
                    for name_sym, init in take_pairs(bindings)
                ]

                binding_names = bindings[::2]
                return_value = next_temp()

                alg = replace_tails_alg(binding_names, return_value)
                new_body = Cons(IR1.WHILE_TRUE,
                                Cons(cata_f(fmap_tail)(alg)(body),
                                     nil))

                return Cons(Sym(None, 'do'),
                            List.from_iter(set_bang_forms + [new_body, return_value]))
            case other:
                return other

    return replace_loop_recur_alg


#
# An Intermediate Representation
#
# It's come to the point where I am making too many mistakes, continuing
# with these raw lisp forms. So we create concrete constructors for
# each syntax element
#
E = TypeVar('E')
S = TypeVar('S')


@dataclass(frozen=True)
class Attr:
    obj: E
    name: Sym


@dataclass(frozen=True)
class Do:
    stmts: tuple[S]
    expr: E


@dataclass(frozen=True)
class DoS:
    stmts: tuple[S]


@dataclass(frozen=True)
class IfExpr:
    pred: E
    con: E
    alt: E


@dataclass(frozen=True)
class IfStmt:
    pred: E
    con: S
    alt: S


@dataclass(frozen=True)
class Fn:
    name: Optional[Sym]
    params: tuple[Sym]
    restparam: Optional[Sym]
    body: S | E


@dataclass(frozen=True)
class Raise:
    e: E


@dataclass(frozen=True)
class Quote:
    name: Sym


@dataclass(frozen=True)
class VarE:
    name: Sym


@dataclass(frozen=True)
class SetBang:
    name: Sym
    init: E


@dataclass(frozen=True)
class Break:
    pass


@dataclass(frozen=True)
class Continue:
    pass


@dataclass(frozen=True)
class WhileTrue:
    action: S


@dataclass(frozen=True)
class Return:
    value: E


@dataclass(frozen=True)
class Call:
    proc: E
    args: tuple[E]


@dataclass(frozen=True)
class Lit:
    value: None | bool | int | float | str | Keyword


def is_stmt(expr):
    """
    being a statement is not the same as containing statements
    """
    match expr:
        case Raise() | SetBang() | WhileTrue() | Break() | Continue() | Return():
            return True
        case DoS() | IfStmt():
            return True
        case _: return False


def convert_to_intermediate_alg(expr):
    match expr:
        case x if x is nil:
            return Lit(nil)
        case None | True | False | int() | float() | str() | Keyword():
            return Lit(expr)
        case Cons(Sym(None, '.') as s, Cons(obj, Cons(Sym() as attr, Nil()))):
            return Attr(obj, attr)
        case Cons(Sym(None, 'do') as s, args):
            *stmts, final = args
            if is_stmt(final):
                return DoS(tuple(args))
            return Do(tuple(stmts), final)
        case Cons(Sym(None, 'if') as s, Cons(pred, Cons(con, Cons(alt, Nil())))):
            return IfExpr(pred, con, alt)
        case Cons(Sym(None, 'if') as s, Cons(pred, Cons(con, Nil()))):
            return IfExpr(pred, con, Lit(None))
        case Cons(Sym(None, 'fn') as s, Cons(Vec() as params, Cons(body, Nil()))):
            positional_params, restparam = split_params(params)
            return Fn(None, positional_params, restparam, body)
        case Cons(Sym(None, 'fn') as s,
                  Cons(Sym(None, _) as n, Cons(Vec() as params, Cons(body, Nil())))):
            positional_params, restparam = split_params(params)
            return Fn(n, positional_params, restparam, body)
        case Cons(Sym(None, 'raise'), Cons(r, Nil())):
            return Raise(r)
        case Cons(Sym(None, 'quote'), Cons(Sym() as s, Nil())):
            return Quote(s)
        case Cons(Sym(None, 'var'), Cons(Sym() as s, Nil())):
            return VarE(s)
        case Cons(IR1.SETBANG, Cons(Sym() as name, Cons(init, Nil()))):
            return SetBang(name, init)
        case Cons(IR1.BREAK, Nil()):
            return Break()
        case Cons(IR1.CONTINUE, Nil()):
            return Continue()
        case Cons(IR1.WHILE_TRUE, Cons(action, Nil())):
            return WhileTrue(action)
        case Cons(IR1.IF_STMT, Cons(pred, Cons(con, Cons(alt, Nil())))):
            return IfStmt(pred, con, alt)
        case Cons(hd, tl):
            return Call(hd, tuple(tl))
        case Sym() as sym:
            return sym
        case _:
            raise NotImplementedError(expr)


def fmap_ir(f, expr):
    match expr:
        case Lit() | Quote() | VarE() | Break() | Continue() | Sym() as terminal:
            return terminal
        case Attr(obj, attr):
            return Attr(f(obj), attr)
        case Do(stmts, final):
            return Do(tuple(map(f, stmts)), f(final))
        case DoS(stmts):
            return DoS(tuple(map(f, stmts)))
        case IfExpr(pred, con, alt):
            return IfExpr(f(pred), f(con), f(alt))
        case IfStmt(pred, con, alt):
            return IfStmt(f(pred), f(con), f(alt))
        case Fn(n, params, restparam, body):
            return Fn(n, params, restparam, f(body))
        case Raise(r):
            return Raise(f(r))
        case SetBang(name, init):
            return SetBang(name, f(init))
        case WhileTrue(action):
            return WhileTrue(f(action))
        case Call(hd, tl):
            return Call(f(hd), tuple(map(f, tl)))
        case Return(e):
            return Return(f(e))
        case _:
            raise NotImplementedError(expr)


def reduce_ir(zero, plus, expr):
    match expr:
        case Lit() | Quote() | VarE() | Break() | Continue() | Sym():
            return zero
        case Attr(obj, _):
            return obj
        case Do(stmts, final):
            return plus(reduce(plus, stmts, zero), final)
        case DoS(stmts):
            return reduce(plus, stmts, zero)
        case IfExpr(pred, con, alt) | IfStmt(pred, con, alt):
            return plus(plus(pred, con), alt)
        case Fn(_, _, _, body):
            return body
        case Raise(r):
            return r
        case SetBang(_, init):
            return init
        case WhileTrue(action):
            return action
        case Call(proc, args):
            return plus(proc, reduce(plus, args, zero))
        case Return(e):
            return e
        case _:
            raise NotImplementedError(expr)


def contains_stmt_alg(expr):
    "if expr or any subexpression of expr is a statement"
    match expr:
        case x if is_stmt(x):
            return True
        case other:
            return reduce_ir(False, operator.or_, other)


def convert_if_expr_to_stmt(i=0):
    def next_temp(prefix=""):
        nonlocal i
        i += 1
        return Sym(None, f'{prefix}__t.{i}')

    fst = lambda pair: pair[0]

    def alg(expr):
        """
        ExprF[(ExprF, contains_stmt: Bool)] -> ExprF
        """
        match expr:
            # c1 and c2 are whether those arms of the if expression
            # contain any statements
            case IfExpr((pred, _), (con, c1), (alt, c2)) if c1 or c2:
                t = next_temp()
                # statement hoisting will clean this up
                return Do((
                    IfStmt(pred,
                           con if is_stmt(con) else SetBang(t, con),
                           alt if is_stmt(alt) else SetBang(t, alt)),
                ), t)
            case other:
                return fmap_ir(fst, other)
        assert False
    return alg


def create_hoist_statements(i=0):
    """
    (1) (do s1 (do s2 e)) -> (do s1 s2 e)
    (2) ((do s e1) e2) -> (do s (e1 e2))  # procedure call
        (. (do s e1) n) -> (do s (. e1 n))
        (if (do s e1) s1 s2) -> (do s (if e2 s1 s2))
        (raise (do s e)) -> (do s (raise e))
        (set! n (do s e)) -> (do s (set! n e))
    (3) (e1 (do s e2)) -> (do (set! t* e1) s (t* e2))  <- applies to further
                                                          args too
    (4) (e1 (do s e2)) -> (do s (e1 e2)) if s,e1 commutes
                                         e.g. e1 is a symbol or literal

    """
    # Analysis
    # (. e x) <- e +
    # (do s... e) <- s +
    # (do e) <- e -> e
    # (if e e1 e2) <- e
    # (if-stmt e s1 s2) <- s
    # (fn [...] s) <- e
    # (raise e) <- s
    # (set! n e) <- s
    # (break) <- s
    # (continue) <- s
    # (while-true s) <- s
    # (e1 e2 ...) <- e
    # e <- e

    def next_temp(prefix=""):
        nonlocal i
        i += 1
        return Sym(None, f'{prefix}__t.{i}')

    def commutes(stmt, expr):
        match stmt:
            case DoS(()): return True
        match expr:
            case Sym() | Lit() | VarE() | Quote(): return True
        return False

    def seq(stmt1, stmt2):
        match (stmt1, stmt2):
            case (DoS(()), _): return stmt2
            case (_, DoS(())): return stmt1
            case (DoS(ss1), DoS(ss2)): return DoS(ss1 + ss2)
            case (DoS(ss1), stmt2): return DoS(ss1 + (stmt2,))
            case (stmt1, DoS(ss2)): return DoS((stmt1,) + ss2)
            case (stmt1, stmt2): return DoS((stmt1, stmt2,))
        assert False

    def reorder(exps):
        match exps:
            case Nil():
                return DoS(()), nil
            case Cons(Do(stmts1_, e1), rest):
                stmts1 = DoS(stmts1_)
            case Cons(e1, rest):
                stmts1 = DoS(())

        rest_stmts, rest_exprs = reorder(rest)

        if commutes(rest_stmts, e1):
            return (seq(stmts1, rest_stmts),
                    Cons(e1, rest_exprs))
        else:
            t1 = next_temp()
            return (seq(stmts1,
                        seq(SetBang(t1, e1),
                            rest_stmts)),
                    Cons(t1, rest_exprs))

    def reorder_expr(exps, reconstruct):
        stmt, exprs = reorder(exps)
        if stmt == DoS(()):
            return reconstruct(exprs)
        if isinstance(stmt, DoS):
            return Do(stmt.stmts, reconstruct(exprs))
        return Do((stmt,), reconstruct(exprs))

    def reorder_stmt(exps, reconstruct):
        stmt, exprs = reorder(exps)
        return seq(stmt, reconstruct(exprs))

    def reorder_e1(e, reconstruct):
        return reorder_expr(Cons(e, nil), lambda el: reconstruct(el.hd))

    def hoist_statements_alg(expr):
        match expr:
            case Attr(e1, attr):
                return reorder_e1(e1, lambda e: Attr(e, attr))
            # This one is a bit special
            case Do(stmts, Do(stmts2, e)):
                return Do(stmts + stmts2, e)
            case IfExpr(pred, con, alt):
                # We have already ensured that there are no statements in
                # the arms
                return reorder_e1(pred, lambda p: IfExpr(p, con, alt))

            case IfStmt(pred, con, alt):
                return reorder_stmt(Cons(pred, nil), lambda el: IfStmt(el.hd, con, alt))
            case Raise(e):
                return reorder_stmt(Cons(e, nil), lambda el: Raise(el.hd))
            case Return(e):
                return reorder_stmt(Cons(e, nil), lambda el: Return(el.hd))
            case SetBang(name, init):
                return reorder_stmt(Cons(init, nil), lambda el: SetBang(name, el.hd))

            case Break() | Continue() | WhileTrue(_):
                # I think these are correct, given that they only are
                # or contain statements
                return expr
            case VarE() | Quote() | Lit():
                return expr
            case Call(proc, args):
                def reconstruct(expr_list):
                    proc, *args = expr_list
                    return Call(proc, tuple(args))
                return reorder_expr(Cons(proc, List.from_iter(args)), reconstruct)
            case other:
                return other
    return hoist_statements_alg


def place_return_alg(expr):
    match expr:
        case Fn(name, params, restparam, body):
            return Fn(name, params, restparam, place_return_outer(body))
        case other:
            return other


def place_return_outer(fn_body):
    match fn_body:
        case Do(stmts, e):
            return DoS(stmts + (Return(e),))
        case body if not is_stmt(body):
            return Return(body)
        case other:
            return other


# This is in ultra-draft idea mode at the moment
def compile_fn(fn: InterpFn, interp, *, mode='func'):
    """
    mode can be one of
    * func -> returns a python function
    * lines -> returns the lines of python that make up the body of the
               function
    * txt -> returns the full program text
    """

    class Uncompilable(Exception):
        pass

    name = fn.name.n if fn.name else None
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
        name = name.replace('=', '_EQUAL_')
        name = name.replace('<', '_LESS_')
        name = name.replace('>', '_GREATER_')

        if not name.isidentifier():
            raise NotImplementedError(name)
        return name

    def create_fn(body_lines, resolved_qualifieds, mode):
        args = ', '.join(
            [mangle_name(sym.n) for sym in params]
            + [f'*{mangle_name(sym.n)}' for sym in filter(None, [restparam])]
        )
        fn_body = '\n'.join([f'  {b}' for b in body_lines])
        fn_name = mangle_name(
            name or f'fn_{hash((params, restparam, body)) & ((1 << 63) - 1)}'
        )

        # Note: using fn_name here is equivalent
        txt = f' def {fn_name}({args}):\n{fn_body}'

        locals = {mangle_name(sym.n): v for (sym, v) in closure.items()} | {
            '__List_from_iter': List.from_iter,
            '__Sym': Sym,
            '__Keyword': Keyword,
        }
        local_vars = ', '.join(locals.keys())
        txt = f"def __create_fn__({local_vars}):\n{txt}\n return {fn_name}"
        globals = {
            mangle_name(str(var.symbol)): var for var in resolved_qualifieds
        }
        ns = {}
        if mode == 'txt':
            return txt
        exec(txt, globals, ns)
        return ns['__create_fn__'](**locals)

    def compile_expr_alg(expr):
        """
        assume all subexpressions have already been compiled and then embedded
        into the structure of our AST
        """

        match expr:
            case Lit(str() | int() | float() | bool() | None as lit):
                return repr(lit)
            case Nil():
                return 'None'
            case Lit(Keyword(None, n)):
                # because of the FileString instances, we have to convert to
                # normal strings
                return f'__Keyword(None, {str(n)!r})'
            case Lit(Keyword(ns, n)):
                return f'__Keyword({str(ns)!r}, {str(n)!r})'
            case Sym(None, n) as sym:
                if sym == fn.name or sym in params or sym == restparam:
                    return mangle_name(n)
                if sym in closure:
                    if isinstance(closure[sym], Var):
                        return mangle_name(n) + '.value'
                # must be a bound var
                return mangle_name(n)
            case Sym() as sym:
                return mangle_name(str(sym)) + '.value'
            case IfExpr(predicate, consequent, alternative):
                return f'({consequent}) if ({predicate}) else ({alternative})'
            case Attr(obj, attr):
                return f'({obj}).{attr.n}'

            # fn will always be in a set!
            case Fn(_, _params, _restparam, _body):
                return expr
            case SetBang(Sym(None, n), Fn(_, _params, _restparam, _body)):
                args = ', '.join(mangle_name(p.n) for p in _params)
                if _restparam is not None:
                    args = f'{args}, *{mangle_name(_restparam.n)}'
                body_lines = [f'  {line}' for line in _body]
                return [f'def {mangle_name(n)}({args}):', *body_lines]

            case Quote(Sym(None, n)):
                return f'__Sym(None, {str(n)!r})'
            case Quote(Sym(ns, n)):
                return f'__Sym({str(ns)!r}, {str(n)!r})'

            case VarE(Sym(None, n) as sym):
                if sym in closure:
                    if isinstance(closure[sym], Var):
                        return mangle_name(sym.n)
                raise Uncompilable(f'unresolved: {n}', location_from(n))
            case VarE(Sym(ns, n) as sym):
                if sym in closure:
                    if isinstance(closure[sym], Var):
                        return mangle_name(str(sym))
                raise Uncompilable(f'unresolved: {sym}', location_from(sym))

            case Call(proc_form, args):
                joined_args = ', '.join(args)
                return f'({proc_form})({joined_args})'

            # Statements
            case IfStmt(pred, con, alt):
                return ([f'if {pred}:']
                        + [f'  {line}' for line in con]
                        + ['else:']
                        + [f'  {line}' for line in alt])
            case SetBang(Sym(None, n), init):
                return [f'{mangle_name(n)} = ({init})']
            case Raise(raisable):
                return [f'raise ({raisable})']
            case Return(e):
                return [f'return {e}']
            case WhileTrue(action_lines):
                return ['while True:',
                        *[f'  {line}' for line in action_lines]]
            case Break():
                return ['break']
            case Continue():
                return ['continue']
            case DoS(stmts):
                return reduce(operator.add, stmts, [])
            case Do(stmts, final):  # this is kinda wrong
                return reduce(operator.add, stmts, []) + [final]
            case _:
                raise Uncompilable(expr, location_from(expr))

    class Counter:
        def __init__(self, i=0):
            self.i = i

        def __iadd__(self, j):
            if not isinstance(j, int):
                return NotImplemented
            self.i += j
            return self

        def __str__(self):
            return str(self.i)

    var_id_counter = Counter()

    # I don't think it's possible to combine this ana with the below cata
    # due to the differing carrier type
    ana = ana_f(fmap)
    prog0 = ana(create_deduce_scope_coalg(var_id_counter))((body, ArrayMap.empty()))

    # These transforms inside this compose run last to first
    prog1 = cata_f(fmap)(compose(
        replace_letstar_alg,
        create_hoist_lambda_alg(var_id_counter),
        remove_vec_and_map_alg,
        remove_complex_quote_alg,
    ))(prog0)

    # After now adding set!, we need a different fmap!!

    cata_sb = cata_f(fmap_setbang)

    def used_qualifieds_alg(expr):
        match expr:
            case Sym(ns, _) as sym if ns is not None:
                return (sym,)
            case other:
                return reduce_expr_setbang(zero=(), plus=operator.add, expr=other)

    used_qualifieds = cata_sb(used_qualifieds_alg)(prog1)

    resolved_qualifieds = [
        interp.resolve_symbol(sym, ArrayMap.empty()) for sym in used_qualifieds
    ]

    prog2 = nest_loop_in_body_of_recursive_fn(params, prog1)

    prog3 = compose(
        cata_sb(convert_to_intermediate_alg),
        cata_sb(create_replace_loop_recur_alg(var_id_counter)),
        cata_sb(nest_loop_in_recursive_fn_alg),
    )(prog2)

    # TODO: work out how to efficiently apply the catamorphism
    # fusion law
    cata_ir = cata_f(fmap_ir)
    prog4 = compose(
        cata_ir(place_return_alg),
        cata_ir(create_hoist_statements(var_id_counter)),
        zygo_f(fmap_ir)(contains_stmt_alg, convert_if_expr_to_stmt(var_id_counter)),
    )(prog3)

    after_transforms = place_return_outer(prog4)

    body_lines = []
    try:
        if restparam is not None:
            mn = mangle_name(restparam.n)
            body_lines += [f'{mn} = __List_from_iter({mn})']
        match cata_ir(compile_expr_alg)(after_transforms):
            case str(result):
                body_lines += [result]
            case list(lines) if lines:
                body_lines += lines
            case _: assert False
        if mode == 'lines':
            return body_lines
        return create_fn(body_lines, resolved_qualifieds, mode)
    except Uncompilable:
        traceback.print_exc()
        return None
