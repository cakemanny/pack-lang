import dataclasses
import importlib
import operator
import os
import sys
import traceback
from dataclasses import dataclass
from functools import reduce
from itertools import islice
from typing import Any

import pack.ast
from pack.ast import Special, fmap_datum, fmap, reduce_expr
from pack.compiler import compile_fn
from pack.data import Sym, Keyword, Vec, List, Cons, Nil, nil, Map, ArrayMap
from pack.exceptions import PackLangError, EvalError, SemanticError, SyntaxError
from pack.reader import read_all_forms, location_from, FileString, read_forms
from pack.recursion import cata_f, ana_f, hylo_f, apo_f, Left, Right
from pack.runtime import Var
from pack.util import take_pairs, untake_pairs


# -------------
#  Interpreter
# -------------

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


KW_MACRO = Keyword(None, "macro")


def set_macro(var: Var):
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


class Fn(pack.ast.Fn):
    def __init__(self, name, params, body, env, interp):
        # TODO: change this to use composition, if possible
        super().__init__(name, params, body, env)
        self.interp = interp

    def __call__(self, *args):
        if not self.restparam and len(args) != len(self.params):
            raise EvalError('incorrect number of arguments', location_from(args))
        env = self.env
        if self.name:
            env = env.assoc(self.name, self)
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

    def __call__(self, interp, *args) -> tuple[Any, Interpreter]:
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


def extract_closure(body, params, interp, env, name=None):
    """
    We write a bottom up traversal that builds a set of symbols that
    are unresolved. As we encounter them in binding forms, we remove
    them from the set.
    """
    def dissoc_all(the_map, keys):
        return reduce(lambda m, k: m.dissoc(k), keys, the_map)

    def find_frees_alg(expr):
        match expr:
            case Sym(None, _) as sym:
                return ArrayMap.empty().assoc(sym, sym)
            case Sym(ns, _) if ns is not None:
                return ArrayMap.empty()
            case Cons(Sym(None, 'let*' | 'loop'),
                      Cons(Vec() as bindings, Cons(body, Nil()))):
                all_free_vars = body
                for binding, init in reversed(tuple(take_pairs(bindings))):
                    all_free_vars = all_free_vars.dissoc(binding)
                    all_free_vars |= init
                return all_free_vars
            case Cons(Sym(None, 'fn'),
                      Cons(Sym() as sym, Cons(Vec() as fn_params, Cons(body, Nil())))):
                return dissoc_all(body.dissoc(sym), fn_params)
            case Cons(Sym(None, 'fn'),
                      Cons(Vec() as fn_params, Cons(body, Nil()))):
                return dissoc_all(body, fn_params)
            case Cons(Sym(None, 'var'), Cons(Sym() as sym, Nil())):
                return ArrayMap.empty().assoc(sym, sym)
            case other:
                return reduce_expr(zero=ArrayMap.empty(),
                                   plus=operator.or_,
                                   expr=other)

    cata = cata_f(fmap)
    bound = params.conj(name) if name else params
    unresolved_frees = dissoc_all(cata(find_frees_alg)(body), bound)

    def resolve(closure, unresolved_free):
        return closure.assoc(
            unresolved_free, interp.resolve_symbol(unresolved_free, env)
        )
    return reduce(resolve, unresolved_frees, ArrayMap.empty())


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
            try:
                interp = load_module(module_path, interp)
            except PackLangError as e:
                raise EvalError(
                    f"Failed to load module {name.n}", location_from(form)
                ) from e
            finally:
                interp = interp.switch_namespace(saved_namespace_name)
            return None, interp

        case Cons(Sym(None, 'import'), Cons(Sym(None, name) as sym, _)):
            # TODO: check that the name is not already used for something else
            try:
                module = importlib.import_module(str(name))
            except ModuleNotFoundError as e:
                raise EvalError(
                    f"No python module named '{name}'", location_from(sym),
                ) from e

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

        case Cons(Sym(None, '.'), Cons(obj_expr, Cons(Sym(_, attr), Nil()))):
            obj = eval_expr(obj_expr, interp, env)
            try:
                return getattr(obj, attr)
            except AttributeError as e:
                raise EvalError(str(e), location_from(form)) from e

        case Cons(Sym(None, 'do'), exprs):
            for expr in exprs:
                result = eval_expr(expr, interp, env)
            return result

        case Cons(Sym(None, 'def'), _):
            raise SyntaxError(
                'def may only appear at top level or in top level do',
                location_from(form)
            )

        case Cons(Sym(None, 'let*'), Cons(Vec() as bindings, Cons(body, Nil()))):
            for binding, init in take_pairs(bindings):
                init_val = eval_expr(init, interp, env)
                env = env.assoc(binding, init_val)
            return eval_expr(body, interp, env)

        case Cons(Sym(None, 'loop'), Cons(Vec() as bindings, Cons(body, Nil()))):
            for binding, init in take_pairs(bindings):
                init_val = eval_expr(init, interp, env)
                env = env.assoc(binding, init_val)
            while True:
                try:
                    return eval_expr(body, interp, env)
                except RecurError as e:
                    bind_names = bindings[::2]
                    if len(e.arg_vals) != len(bind_names):
                        raise SemanticError(
                            ('not enough operands given to recur: '
                             f'expected {len(bind_names)}, got {len(e.arg_vals)}: ',
                             f'{e.form} in {form}'),
                            e.location
                        )
                    for bind_name, val in zip(bind_names, e.arg_vals):
                        env = env.assoc(bind_name, val)

        case Cons(Sym(None, 'recur'), args):
            arg_vals = [
                eval_expr(arg, interp, env) for arg in args
            ]
            raise RecurError(form, arg_vals)

        case Cons(Sym(None, 'if'),
                  Cons(predicate,
                       Cons(consequent, Cons(alternative, Nil())))):
            if eval_expr(predicate, interp, env):
                return eval_expr(consequent, interp, env)
            return eval_expr(alternative, interp, env)
        case Cons(Sym(None, 'if'), Cons(predicate, Cons(consequent, Nil()))):
            if eval_expr(predicate, interp, env):
                return eval_expr(consequent, interp, env)
            return None

        case Cons(Sym(None, 'fn'), Cons(Vec() as params, Cons(body, Nil()))):
            closure = extract_closure(body, params, interp, env)
            fn = Fn(None, params, body, closure, interp)
            if interp.compilation_enabled:
                compiled = compile_fn(fn, interp)
                if compiled is not None:
                    return compiled
            return fn
        case Cons(Sym(None, 'fn'),
                  Cons(Sym(None, _) as name,
                       Cons(Vec() as params, Cons(body, Nil())))):
            closure = extract_closure(body, params, interp, env, name=name)
            fn = Fn(name, params, body, closure, interp)
            if interp.compilation_enabled:
                compiled = compile_fn(fn, interp)
                if compiled is not None:
                    return compiled
            return fn

        case Cons(Sym(None, 'raise'), Cons(raisable_form, Nil())):
            raisable = eval_expr(raisable_form, interp, env)
            raise raisable

        case Cons(Sym(None, 'quote'), Cons(arg, Nil())):
            return arg

        case Cons(Sym(None, 'var'), Cons(Sym() as sym, Nil())):
            return interp.resolve_symbol(sym, env)

        case Cons(proc_form, args):
            proc = eval_expr(proc_form, interp, env)
            if isinstance(proc, RTFn):
                raise TypeError(
                    'attempted to call top level function outside top level'
                )
            arg_vals = [eval_expr(arg, interp, env) for arg in args]
            while True:
                try:
                    return proc(*arg_vals)
                except RecurError as e:
                    arg_vals = e.arg_vals

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
            return Vec.from_iter(
                eval_expr(sub_form, interp, env) for sub_form in vec
            )

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


def validate_recur_tail(form):
    """
    check that recur occurs only in tail position
    """
    # Analysis
    # (. non-tail x)
    # (do non-tail... tail) <- if in tail position
    # (let* [x non-tail y non-tail ... ...] tail) <- if in tail pos
    # (loop [x non-tail y non-tail ... ...] tail) <- resolved
    # (recur non-tail...)
    # (if non-tail tail tail) <- if in tail pos
    # (fn [...] tail) <- resolved
    # (raise non-tail)
    # (non-tail non-tail...)
    # [non-tail...]
    # {non-tail non-tail...}

    def is_recur(form):
        match form:
            case Cons(Special.RECUR, _): return True
            case _: return False

    def first_recur_or_none(iterable):
        return next(filter(is_recur, iterable), None)

    def err(bad_recur):
        return SyntaxError(
            f'recur in non-tail position: {form}', location_from(bad_recur)
        )

    # Returns 'form' if it contains a recur in a tail position
    # This bubbles up recursively
    match form:
        case Cons(Special.DOT, Cons(obj, Cons(_, Nil()))):
            if is_recur(obj):
                raise err(obj)
            return None
        case Cons(Special.DO, subforms):  # TODO: can we put [*non_tail, tail]
            *non_tail, tail = subforms
            if bad_recur := first_recur_or_none(non_tail):
                raise err(bad_recur)
            return tail
        case Cons(Special.LETSTAR, Cons(Vec() as bindings, Cons(body, Nil()))):
            if bad_recur := first_recur_or_none(islice(bindings, 1, None, 2)):
                raise err(bad_recur)
            return body
        case Cons(Special.LOOP, Cons(Vec() as bindings, Cons(_, Nil()))):
            if bad_recur := first_recur_or_none(islice(bindings, 1, None, 2)):
                raise err(bad_recur)
            # loop resolves a recur, so we return none, regardless of
            # whether there was a recur in the body.
            return None
        case Cons(Special.RECUR, args):
            if bad_recur := first_recur_or_none(args):
                raise err(bad_recur)
            # recur is itself a recur :)
            return form
        case Cons(Special.IF, Cons(pred, Cons(consequent, Cons(alt, Nil())))):
            if is_recur(pred):
                raise err(pred)
            return consequent or alt
        case Cons(Special.IF, Cons(pred, Cons(consequent, Nil()))):
            if is_recur(pred):
                raise err(pred)
            return consequent
        case Cons(Special.FN, Cons(Vec(), Cons(_, Nil()))):
            return None  # fn is a another recursion point. i.e. the recur resolves
        case Cons(Special.FN, Cons(Sym(), Cons(Vec(), Cons(_, Nil())))):
            return None
        case Cons() | Vec() as seq:
            if bad_recur := first_recur_or_none(seq):
                raise err(bad_recur)
        case ArrayMap() | Map() as m:
            if bad_recur := first_recur_or_none(untake_pairs(m.items())):
                raise err(bad_recur)
        case other:
            return other


# And simplify?
def validate_syntax_coalg(form):
    # Maybe we could consider simplifying two armed if here as well
    match form:
        case Cons(Sym(None, '.'), Cons(_, Cons(Sym(_, _), Nil()))):
            return form
        case Cons(Sym(None, '.'), _):
            raise SyntaxError(f'invald member expression: {form}', location_from(form))
        case Cons(Sym(None, 'def'), Cons(Sym(), Nil())):
            return form
        case Cons(Sym(None, 'def'), Cons(Sym(), Cons(_, Nil()))):
            return form
        case Cons(Sym(None, 'def'), _):
            raise SyntaxError(f'invalid form def: {form}', location_from(form))
        case Cons(Sym(None, 'let*'), Cons(Vec() as bindings, Cons(_, Nil()))):
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
            return form
        case Cons(Sym(None, 'let*'), _):
            raise SyntaxError(f"invalid let: {form}", location_from(form))
        case Cons(Sym(None, 'loop'), Cons(Vec() as bindings, Cons(_, Nil()))):
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
            return form
        case Cons(Sym(None, 'loop'), _):
            raise SyntaxError(f"invalid loop: {form}", location_from(form))
        case Cons(Sym(None, 'if'), Cons(_, Cons(_, Cons(_, Nil())))):
            return form  # two armed if
        case Cons(Sym(None, 'if'), Cons(_, Cons(_, Nil()))):
            return form  # one armed if
        case Cons(Sym(None, 'if')):
            raise SyntaxError(f'invalid if form: {form}', location_from(form))
        case Cons(Sym(None, 'fn'), Cons(Vec() as params, Cons(_, Nil()))):
            for param in params:
                match param:
                    case Sym(None, _): pass
                    case _:
                        raise SyntaxError(
                            'fn params must be simple names', location_from(form)
                        )
            return form
        case Cons(Sym(None, 'fn'),
                  Cons(Sym() as name_sym, Cons(Vec() as params, Cons(_, Nil())))):
            if name_sym.ns is not None:
                raise SyntaxError(
                    f'fn name not simple name: {name_sym}', location_from(name_sym)
                )
            for param in params:
                match param:
                    case Sym(None, _): pass
                    case _:
                        raise SyntaxError(
                            'fn params must be simple names', location_from(form)
                        )
            return form
        case Cons(Sym(None, 'fn'), _):
            raise SyntaxError(f'invalid fn form: {form}', location_from(form))
        case Cons(Sym(None, 'raise'), Cons(_, Nil())):
            return form
        case Cons(Sym(None, 'raise'), _):
            raise SyntaxError(
                f"invalid special form 'raise': {form}", location_from(form)
            )
        case Cons(Sym(None, 'quote'), Cons(arg, Nil())):
            return form
        case Cons(Sym(None, 'quote'), args):
            raise SyntaxError(
                f'wrong number of arguments to quote: {len(args)}',
                location_from(form)
            )
        case Cons(Sym(None, 'var'), Cons(Sym(), Nil())):
            return form
        case Cons(Sym(None, 'var'), Cons(arg, Nil())):
            arg_type = type(arg).__name__
            raise SyntaxError(
                f'invalid var, expected symbol, got {arg_type}: {form}',
                location_from(form),
            )
        case Cons(Sym(None, 'var'), _):
            raise SyntaxError(
                f'invalid special form var: {form}', location_from(form)
            )
    return form


def fmap_with_def(f, expr):
    # The main fmap is only used in the compiler to compile functions
    # where def, require and import are not allowed.
    # Here we write a version that handles those cases.
    match expr:
        case Cons(Sym(None, 'def') as s, Cons(name, Cons(init, Nil()))):
            return Cons(s, Cons(name, Cons(f(init), nil)))
        case Cons(Sym(None, 'def'), Cons(name, Nil())):
            return expr
        case Cons(Sym(None, 'require' | 'import'), _):
            return List.from_iter(tuple(map(f, expr)))
        case other:
            return fmap(f, other)


# Not Used
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


# Not Used
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
        validated = ana_f(fmap_with_def)(validate_syntax_coalg)(expanded)
        _ = cata_f(fmap_with_def)(validate_recur_tail)(validated)
        defined, interp = create_defs(validated, interp)
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
            except PackLangError as e:
                print(repr(e))
                continue
            except Exception:
                traceback.print_exc()
                continue

            try:
                results, interp = expand_and_evaluate_forms(forms, interp)
                for result in results:
                    print(repr(result))
            except PackLangError as e:
                print(repr(e))
                while hasattr(e, '__cause__') and e.__cause__ is not None:
                    e = e.__cause__
                    print(repr(e))
            except Exception:
                traceback.print_exc()

    else:

        forms = read_all_forms(
            FileString(sys.stdin.read(), file="<stdin>", lineno=1, col=0)
        )
        _, interp = expand_and_evaluate_forms(forms, interp)
