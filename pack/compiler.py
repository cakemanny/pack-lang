import operator
import traceback
from functools import reduce

from pack.ast import Special, Fn, fmap, fmap_datum, reduce_expr
from pack.data import Sym, Keyword, Vec, List, Cons, Nil, nil, Map, ArrayMap
from pack.reader import location_from
from pack.recursion import cata_f, ana_f, compose
from pack.runtime import Var
from pack.util import take_pairs, untake_pairs


# --------------
# Compiler
# --------------

def fmap_setbang(f, expr):
    match expr:
        case Cons(Sym(None, 'set!') as s, Cons(name_sym, Cons(init, Nil()))):
            return Cons(s, Cons(name_sym, Cons(f(init), Nil())))
        case Cons(Sym(None, 'let*'), _):
            raise ValueError('unexpected let*')
        case other:
            return fmap(f, other)


def reduce_expr_setbang(zero, plus, expr):
    match expr:
        case Cons(Sym(None, 'set!'), Cons(_, Cons(init, Nil()))):
            return init
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
            return Cons(s, Cons(pred, Cons(f(con), Cons(f(alt), Nil()))))
        case Cons(Special.IF as s, Cons(pred, Cons(f(con), Nil()))):
            return Cons(s, Cons(pred, Cons(f(con), Nil())))
        case other:
            return other


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


# FIXME: This is running bottom up.
# broken might be (quote (something (quote something)))
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
        if p == Fn.PARAM_SEP:
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
                Cons(Sym(None, 'set!'), Cons(name_sym, Cons(init, nil)))
                for name_sym, init in take_pairs(bindings)
            ]
            return Cons(Sym(None, 'do'),
                        List.from_iter(set_bang_forms + [body]))
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
                        Cons(Sym(None, 'set!'), Cons(name_sym, Cons(init, nil)))
                        for name_sym, init in zip(temp_names, args)
                    ]
                    rebinding = [
                        Cons(Sym(None, 'set!'), Cons(name_sym, Cons(init, nil)))
                        for name_sym, init in zip(binding_names, temp_names)
                    ]
                    continue_ = Cons(Sym(None, 'continue'), nil)
                    return Cons(Special.DO,
                                List.from_iter(evaluation + rebinding + [continue_]))

                # Cases where the forms contain deeper tail expressions
                case Cons(Special.DO | Special.LETSTAR | Special.IF, _):
                    # recur and other tail expressions have been replaced
                    # thanks to fmap_tail
                    return expr

                # Tail Reached: replace the result with a set!
                case other:
                    eval_and_assign = Cons(Sym(None, 'set!'),
                                           Cons(return_value,
                                                Cons(other, nil)))
                    break_ = Cons(Sym(None, 'break'), nil)
                    return Cons(Special.DO,
                                Cons(eval_and_assign, Cons(break_, nil)))
        return alg

    def replace_loop_recur_alg(expr):
        match expr:
            case Cons(Sym(None, 'loop'),
                      Cons(Vec() as bindings,
                           Cons(body, Nil()))):
                set_bang_forms = [
                    Cons(Sym(None, 'set!'), Cons(name_sym, Cons(init, nil)))
                    for name_sym, init in take_pairs(bindings)
                ]

                binding_names = bindings[::2]
                return_value = next_temp()

                alg = replace_tails_alg(binding_names, return_value)
                new_body = Cons(Sym(None, 'while-true'),
                                Cons(cata_f(fmap_tail)(alg)(body),
                                     nil))

                return Cons(Sym(None, 'do'),
                            List.from_iter(set_bang_forms + [new_body, return_value]))
            case other:
                return other

    return replace_loop_recur_alg


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
    # (if e s1 s2) <- s
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

    def is_do(expr):
        match expr:
            case Cons(Sym(None, 'do'), _): return True
            case _: return False

    def is_stmt(expr):
        match expr:
            case Cons(Sym(None, 'do'), _): return True
            case Cons(Sym(None, 'raise'), _): return True
            case Cons(Sym(None, 'set!'), _): return True
            case Cons(Sym(None, 'break'), _): return True
            case Cons(Sym(None, 'continue'), _): return True
            case Cons(Sym(None, 'while-true'), _): return True
            case Cons(Sym(None, 'if'), Cons(_, Cons(con, Cons(alt, Nil())))):
                return is_stmt(con) or is_stmt(alt)
            case Cons(Sym(None, 'if'), Cons(_, Cons(con, Nil()))):
                return is_stmt(con)
            case _: return False

    def reorder(exp, reconstruct):
        # Pulls statements out of expr and puts them on the outside
        # and then reconstructs expr only containing the nested expr
        # into the place where it was broken
        match exp:
            case Cons(Sym(None, 'do'), args):
                *stmts, e = args
                return Cons(Sym(None, 'do'),
                            List.from_iter(stmts)
                            + Cons(reconstruct(e), nil))
            case other:
                return reconstruct(other)

    def commutes(stmt, expr):
        if stmt is None:
            return True
        match expr:
            case Sym(): return True
            case str() | int() | float() | bool() | None | Nil(): return True
            case Keyword(): return True
            case Cons(Sym(None, 'var'), _): return True
            case Cons(Sym(None, 'quote'), _): return True
        return False

    def reorder2(exps, reconstruct):
        def aux(exps):
            match exps:
                case Nil():
                    return None, nil
                case Cons(e1, rest):
                    if is_do(e1):
                        *stmts1, e1_ = e1.tl
                        stmts1 = List.from_iter(stmts1)
                    else:
                        e1_ = e1
                        stmts1 = nil

                    rest_stmts, rest_exprs = aux(rest)

                    if commutes(rest_stmts, e1_):
                        return (Cons(Sym(None, 'do'),
                                     List.from_iter(stmts1)
                                     + List.from_iter([rest_stmts])),
                                Cons(e1_, rest_exprs))
                    else:
                        t1 = next_temp()
                        return (Cons(Sym(None, 'do'),
                                     List.from_iter([
                                         stmts1,
                                         Cons(Sym(None, 'set!'),
                                              Cons(t1, Cons(e1_, nil))),
                                         rest_stmts,
                                     ])),
                                Cons(t1, rest_exprs))
        stmt, exprs = aux(exps)
        if stmt is None:
            return reconstruct(exprs)
        if is_do(stmt):
            return Cons(stmt.hd, stmt.tl + Cons(reconstruct(exprs), nil))
        return Cons(Sym(None, 'do'),
                    List.from_iter([stmt, reconstruct(exprs)]))

    def hoist_statements_alg(expr):
        match expr:
            case Cons(Sym(None, '.') as s,
                      Cons(e1, Cons(attr, Nil()))) if is_do(e1):
                return reorder(e1, lambda e: Cons(s, Cons(e, Cons(attr, nil))))
            case Cons(Sym(None, '.') as s,
                      Cons(e1, Cons(attr, Nil()))):
                return expr
            # This one is a bit special
            case Cons(Sym(None, 'do') as s, args):
                *stmts, e = args
                if is_do(e):
                    return Cons(s, List.from_iter(stmts) + e.tl)
                return expr
            case Cons(Sym(None, 'if') as s, Cons(pred, stmts)) if is_do(pred):
                return reorder(pred, lambda pred: Cons(s, Cons(pred, stmts)))
            case Cons(Sym(None, 'if') as s, Cons(pred, stmts)):
                return expr
            case Cons(Sym(None, 'raise') as s, Cons(e, Nil())) if is_do(e):
                return reorder(e, lambda e: Cons(s, Cons(e, nil)))
            case Cons(Sym(None, 'raise') as s, Cons(e, Nil())):
                return expr
            case Cons(Sym(None, 'set!') as s, Cons(name_sym, Cons(e, Nil()))) if is_do(e):
                return reorder(e, lambda e: Cons(s, Cons(name_sym, Cons(e, Nil()))))
            case Cons(Sym(None, 'break' | 'continue' | 'while-true'), _):
                # TODO
                return expr
            case Cons(Sym(None, 'var' | 'quote'), _):
                return expr
            case Cons() as lst:
                return reorder2(lst, lambda lst: lst)
            case Cons(proc, args) if is_do(proc):
                return reorder(proc, lambda e: Cons(e, args))
            case other:
                return other
    return hoist_statements_alg


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
        args = ', '.join(
            [mangle_name(sym.n) for sym in params]
            + [f'*{mangle_name(sym.n)}' for sym in filter(None, [restparam])]
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

    # There will need to be a stage that converts expressions containing
    # raise, into a statement sequence? or... define a raise_ func

    def compile_statement_alg(expr):
        match expr:
            case Cons(Sym(None, 'set!'), Cons(name, Cons(init, Nil()))):
                return [f'{name} = {init}']
        assert False, "TODO"

    def compile_expr_alg(expr):
        """
        assume all subexpressions have already been compiled and then embedded
        into the structure of our AST

        TODO: do , loop, recur, complex if, fn,
        """

        match expr:
            case str() | int() | float() | bool() | None as lit:
                return repr(lit)
            case Nil():
                return 'None'
            case Keyword(None, n):
                # because of the FileString instances, we have to convert to
                # normal strings
                return f'__Keyword(None, {str(n)!r})'
            case Keyword(ns, n):
                return f'__Keyword({str(ns)!r}, {str(n)!r})'
            case Sym(None, n) as sym:
                if sym == restparam:
                    return mangle_name(n)
                if sym in params:
                    return mangle_name(n)
                if sym in closure:
                    if isinstance(closure[sym], Var):
                        return mangle_name(n) + '.value'
                # must be a bound var
                return mangle_name(n)
                raise Uncompilable(f'unresolved: {n}', location_from(n))
            case Sym() as sym:
                return mangle_name(str(sym)) + '.value'

            case Cons(Sym(None, 'if'),
                      Cons(predicate,
                           Cons(consequent, Cons(alternative, Nil())))):
                return f'({consequent}) if ({predicate}) else ({alternative})'
            case Cons(Sym(None, 'if'),
                      Cons(predicate,
                           Cons(consequent, Nil()))):
                return f'({consequent}) if ({predicate}) else None'
            case Cons(Sym(None, '.'), Cons(obj, Cons(attr, Nil()))):
                return f'({obj}).{attr.n}'
            case Cons(Sym(None, 'raise'), Cons(raisable, Nil())):
                return f'raise__({raisable})'

            case Cons(Sym(None, 'quote'), Cons(Sym(None, n), Nil())):
                return f'__Sym(None, {str(n)!r})'
            case Cons(Sym(None, 'quote'), Cons(Sym(ns, n), Nil())):
                return f'__Sym({str(ns)!r}, {str(n)!r})'

            case Cons(Sym(None, 'var'), Cons(Sym(None, n) as sym, Nil())):
                if sym in closure:
                    if isinstance(closure[sym], Var):
                        return mangle_name(sym.n)
                raise Uncompilable(f'unresolved: {n}', location_from(n))
            case Cons(Sym(None, 'var'), Cons(Sym(ns, n) as sym, Nil())):
                if sym in closure:
                    if isinstance(closure[sym], Var):
                        return mangle_name(str(sym))
                raise Uncompilable(f'unresolved: {sym}', location_from(sym))

            case Cons(Sym(None, 'set!'), _):
                raise Uncompilable(expr)
            case Cons(Sym(None, _) as s, args) if s in Special.all:
                # Not yet implemented
                raise Uncompilable(expr, location_from(expr))

            case Cons(proc_form, args):
                joined_args = ', '.join(args)
                return f'({proc_form})({joined_args})'
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

    prog2 = cata_f(fmap_setbang)(create_replace_loop_recur_alg(var_id_counter))(prog1)

    after_transforms = prog2

    body_lines = []
    try:
        if restparam is not None:
            mn = mangle_name(restparam.n)
            body_lines += [f'{mn} = __List_from_iter({mn})']
        body_lines += ['return ' + cata_sb(compile_expr_alg)(after_transforms)]
        if mode == 'lines':
            return body_lines
        return create_fn(body_lines, resolved_qualifieds)
    except Uncompilable:
        traceback.print_exc()
        return None
