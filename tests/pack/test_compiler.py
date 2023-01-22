import pytest

from pack.data import Sym, Vec, ArrayMap, Cons, nil, Nil
from pack.interp import Interpreter
from pack.interp import Var, Fn, expand_and_evaluate_forms
from pack.reader import read_all_forms


class Any:
    "makes some assertions a little nicer"
    def __init__(self, type=None):
        self.type = type

    def __eq__(self, o):
        if self.type is not None:
            return isinstance(o, self.type)
        return True


@pytest.fixture
def initial_interpreter():
    return Interpreter()


# -------------
#  Compiler
# -------------


def test_deduce_scope():
    from pack.ast import fmap
    from pack.compiler import create_deduce_scope_coalg
    from pack.recursion import ana_f
    ana = ana_f(fmap)

    form = read_all_forms("""\
    (do
        (let* [x 5 y 9] (* x y)))
    """)[0]

    assert ana(create_deduce_scope_coalg())((form, ArrayMap.empty())) \
        == read_all_forms("""
    (do
        (let* [x.1 5 y.2 9] (* x.1 y.2)))
    """)[0]

    form = read_all_forms("""\
    (fn [x y z & rest]
        (let* [u 5 w 9]
            (some-f x y rest w)))
    """)[0]

    assert ana(create_deduce_scope_coalg())((form, ArrayMap.empty())) \
        == read_all_forms("""
    (fn [x.1 y.2 z.3 & rest.4]
        (let* [u.5 5 w.6 9]
            (some-f x.1 y.2 rest.4 w.6)))
    """)[0]


def test_hoist_lambda():
    from pack.ast import fmap
    from pack.recursion import cata_f
    from pack.compiler import create_hoist_lambda_alg
    cata = cata_f(fmap)

    form = read_all_forms("""\
    (do
        (fn [x y z] (y x z)))
    """)[0]

    assert cata(create_hoist_lambda_alg())(form) == read_all_forms("""
    (do
        (let* [__t.1 (fn [x y z] (y x z))]
            __t.1))
    """)[0]


def test_replace_letstar():
    from pack.ast import fmap
    from pack.recursion import cata_f
    from pack.compiler import replace_letstar_alg
    cata = cata_f(fmap)

    form = read_all_forms("""\
    (do
        (let* [x 5 y 9] (* x y)))
    """)[0]

    assert cata(replace_letstar_alg)(form) == read_all_forms("""
    (do
        (do
            (pack.core/set! x 5)
            (pack.core/set! y 9)
            (* x y)))
    """)[0]


def test_nest_loop_in_recursive_fn():
    from pack.recursion import cata_f
    from pack.compiler import fmap_setbang, nest_loop_in_recursive_fn_alg

    form = read_all_forms("""\
    (fn [x y]
        (if (= x 0)
            y
            (recur (- x 1) (* x y))))
    """)[0]

    assert cata_f(fmap_setbang)(nest_loop_in_recursive_fn_alg)(form) == \
        read_all_forms("""\
    (fn [x y]
        (loop [x x y y]
            (if (= x 0)
                y
                (recur (- x 1) (* x y)))))
    """)[0]

    # Idempotency
    form = read_all_forms("""\
    (fn [x y]
        (loop [x x y y]
            (if (= x 0)
                y
                (recur (- x 1) (* x y)))))
    """)[0]
    assert cata_f(fmap_setbang)(nest_loop_in_recursive_fn_alg)(form) == form


def test_replace_loop_recur():
    from pack.recursion import cata_f
    from pack.compiler import fmap_setbang, create_replace_loop_recur_alg

    form = read_all_forms("""\
    (loop [x 2 y 3]
        (if (= x 0)
            y
            (recur (- x 1) (* x y))))
    """)[0]

    assert cata_f(fmap_setbang)(create_replace_loop_recur_alg())(form) == \
        read_all_forms("""\
    (do
        (pack.core/set! x 2)
        (pack.core/set! y 3)
        (pack.core/while-true
            (if (= x 0)
                (do
                    (pack.core/set! __t.1 y)
                    (pack.core/break))
                (do
                    (pack.core/set! x__t.2 (- x 1))
                    (pack.core/set! y__t.3 (* x y))
                    (pack.core/set! x x__t.2)
                    (pack.core/set! y y__t.3)
                    (pack.core/continue))))
        __t.1)
    """)[0]


def read_and_convert(text):
    from pack.compiler import fmap_setbang, convert_to_intermediate_alg, cata_f

    form = read_all_forms(text)[0]
    return cata_f(fmap_setbang)(convert_to_intermediate_alg)(form)


def test_convert_if_expr_to_stmt():
    # Idea, if an if expression contains statements, we create
    # a new temp, and then assign to it in the arms of the if
    # If the value in the arm is a statement, we elide the assignment

    from pack.recursion import zygo_f
    from pack.compiler import fmap_ir, convert_if_expr_to_stmt, contains_stmt_alg

    def run(form):
        return zygo_f(fmap_ir)(contains_stmt_alg, convert_if_expr_to_stmt())(form)

    form = read_and_convert("""\
    (do
        (pack.core/set! y (if (< x 0) (raise "x") x))
        y)
    """)

    assert run(form) == read_and_convert("""\
    (do
        (pack.core/set! y
            (do
                (pack.core/if-stmt (< x 0)
                    (raise "x")
                    (pack.core/set! __t.1 x))
                __t.1))
        y)
    """)

    # Does nothing on expression only if
    form = read_and_convert("""\
    (do
        (pack.core/set! y (if (< x 0) 0 x))
        y)
    """)

    assert run(form) == read_and_convert("""\
    (do
        (pack.core/set! y (if (< x 0) 0 x))
        y)
    """)


def test_hoist_statements():
    from pack.recursion import cata_f
    from pack.compiler import fmap_ir, create_hoist_statements

    form = read_and_convert("""\
    (. (do (raise s1) (raise s2) e) n)
    """)

    assert cata_f(fmap_ir)(create_hoist_statements())(form) == \
        read_and_convert("""\
    (do (raise s1) (raise s2) (. e n))
    """)

    form = read_and_convert("""\
    (do (raise s1) (do (raise s2) e))
    """)

    assert cata_f(fmap_ir)(create_hoist_statements())(form) == \
        read_and_convert("""\
    (do (raise s1) (raise s2) e)
    """)

    form = read_and_convert("""\
    (if (do (raise e) true) nil)
    """)

    assert cata_f(fmap_ir)(create_hoist_statements())(form) == \
        read_and_convert("""\
    (do
        (raise e)
        (if true nil))
    """)

    # Goes from do expr inside to do statement outside
    form = read_and_convert("""\
    (raise (do (raise e) e2))
    """)
    assert cata_f(fmap_ir)(create_hoist_statements())(form) == \
        read_and_convert("""\
    (do (raise e) (raise e2))
    """)

    form = read_and_convert("""\
    (pack.core/set! n (do (raise e) nil))
    """)
    assert cata_f(fmap_ir)(create_hoist_statements())(form) == \
        read_and_convert("""\
    (do (raise e) (pack.core/set! n nil))
    """)

    # Mess mess mess
    form = read_and_convert("""\
    ((do (raise e) f) 1 2)
    """)
    assert cata_f(fmap_ir)(create_hoist_statements())(form) == \
        read_and_convert("""\
    (do (raise e) (f 1 2))
    """)

    form = read_and_convert("""\
    (f (do (raise e) 1) 2)
    """)

    # we will clean this up, the (do nil), etc
    assert cata_f(fmap_ir)(create_hoist_statements())(form) == \
        read_and_convert("""\
    (do (raise e) (f 1 2))
    """)

    # non-commuting statements
    # TODO


def test_compiler__0(initial_interpreter):
    from pack.interp import compile_fn

    text = """\
    (ns pack.core)
    (def *compile* false)
    (fn not [x] (if x false true))
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)
    assert results[-1] == Any(Fn)

    fn = results[-1]
    lines = compile_fn(fn, interp, mode='lines')
    assert lines == ['return (False) if (x) else (True)']


def test_compiler__1(initial_interpreter):
    from typing import Callable

    text = """\
    (ns pack.core)
    (def *compile* true)

    (def not (fn not [x] (if x false true)))
    (not false)
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'pack.core'
    assert interp.namespaces['pack.core'].defs['not']
    assert results == [
        None,
        Var(Sym('pack.core', '*compile*'), True),
        Var(Sym('pack.core', 'not'), Any(Callable)),
        True
    ]


@pytest.mark.parametrize('fn_txt,expected_lines', [
    ('(fn f [x] (. x name))', ['return (x).name']),
    ('(fn f [] "a string")', ["return 'a string'"]),
    ('(fn f [] 5)', ["return 5"]),
    ('(fn f [] 5.9)', ["return 5.9"]),
    ('(fn f [] :a-key)', ["return __Keyword(None, 'a-key')"]),
    ('(fn f [] ((. "hi" islower)))', ["return (('hi').islower)()"]),
    ('(fn f [] ((. "hi" index) "i"))', ["return (('hi').index)('i')"]),
    ('(fn f [] (var *compile*))', ["return _STAR_compile_STAR_"]),
    # xfail: ('(fn f [] (raise "x"))', ["raise ('x')"]),
])
def test_compiler__simple_expressions(
        fn_txt, expected_lines, initial_interpreter
):
    from pack.interp import compile_fn

    text = f"""\
    (ns pack.core)
    (def *compile* false)
    {fn_txt}
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)
    assert results[-1] == Any(Fn)

    fn = results[-1]
    lines = compile_fn(fn, interp, mode='lines')

    assert lines == expected_lines


def test_compiler__references(initial_interpreter):
    from pack.interp import compile_fn

    text = """\
    (ns pack.core)
    (def *compile* false)
    (def not (fn not [x] (if x false true)))
    (ns user)
    (pack.core/refer 'pack.core)
    ; refer to not
    (fn rest [lst] (if (not lst) nil (. lst tl)))
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)
    assert results[-1] == Any(Fn)

    fn = results[-1]
    lines = compile_fn(fn, interp, mode='lines')
    assert lines == ['return (None) if ((not__.value)(lst)) else ((lst).tl)']


def test_compiler__qualified_references(initial_interpreter):
    from pack.interp import compile_fn

    text = """\
    (ns pack.core)
    (def *compile* false)
    (def not (fn not [x] (if x false true)))
    ; refer to not
    (fn rest [lst] (if (pack.core/not lst) nil (. lst tl)))
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)
    assert results[-1] == Any(Fn)

    fn = results[-1]
    lines = compile_fn(fn, interp, mode='lines')
    assert lines == [
        'return (None) if ((pack_DOT_core_SLASH_not.value)(lst)) else ((lst).tl)'
    ]


def test_compiler__remove_quote():
    from pack.compiler import remove_quote

    text = """\
    '(a b c)
    """
    form = read_all_forms(text)[0].tl.hd
    assert remove_quote(form) == read_all_forms("(pack.core/list 'a 'b 'c)")[0]

    text = """\
    '(a 1 [b 2 {c 3 :d 4}])
    """
    form = read_all_forms(text)[0].tl.hd
    # There are two possibilities due to the "random" iteration order of
    # hash-maps
    assert remove_quote(form) == read_all_forms("""\
    (pack.core/list 'a 1 (pack.core/vector 'b 2 (pack.core/hash-map 'c 3 :d 4)))
    """)[0] or remove_quote(form) == read_all_forms("""\
    (pack.core/list 'a 1 (pack.core/vector 'b 2 (pack.core/hash-map :d 4 'c 3)))
    """)[0]

    text = """\
    '()
    """
    form = read_all_forms(text)[0].tl.hd
    assert remove_quote(form) == read_all_forms("(pack.core/list)")[0]

    text = """\
    '(1 '(1 a))
    """
    form = read_all_forms(text)[0].tl.hd
    assert remove_quote(form) == read_all_forms(
        "(pack.core/list 1 (pack.core/list 'quote (pack.core/list 1 'a)))"
    )[0]


@pytest.mark.parametrize('fn_txt,expected_lines', [
    ('(fn f [x] (quote [1 a :d]))',
     ["return (pack_DOT_core_SLASH_vector.value)(1, __Sym(None, 'a'), __Keyword(None, 'd'))"]),  # noqa

    ('(fn f [x] [1 2])',
     ["return (pack_DOT_core_SLASH_vector.value)(1, 2)"]),
])
def test_compiler__quoted_data(
        fn_txt, expected_lines, initial_interpreter
):
    from pack.interp import compile_fn

    text = f"""\
    (require 'pack.core)
    (def *compile* false)
    (ns user)
    {fn_txt}
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)
    assert results[-1] == Any(Fn)

    fn = results[-1]
    lines = compile_fn(fn, interp, mode='lines')

    assert lines == expected_lines


@pytest.mark.parametrize('fn_txt,expected_lines', [
    ('(fn [] (fn [] 1))',
     [
         "def __t_DOT_1():",
         "  return 1",
         "return __t_DOT_1",
     ]),
    ('(fn [] (fn [x] x))',
     [
         "def __t_DOT_2(x_DOT_1):",
         "  return x_DOT_1",
         "return __t_DOT_2",
     ]),
])
def test_compiler__nested_functions(
        fn_txt, expected_lines, initial_interpreter
):
    from pack.interp import compile_fn

    text = f"""\
    (require 'pack.core)
    (def *compile* false)
    (ns user)
    {fn_txt}
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)
    assert results[-1] == Any(Fn)

    fn = results[-1]
    lines = compile_fn(fn, interp, mode='lines')

    assert lines == expected_lines


def test_compiler__recursive_function(initial_interpreter):
    from pack.interp import compile_fn

    text = """\
    (require 'pack.core)
    (def *compile* false)
    (ns user)
    (def y (fn f [] (if false (f))))
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    fn = interp.resolve_symbol(Sym('user', 'y'), ArrayMap.empty()).value
    lines = compile_fn(fn, interp, mode='lines')

    assert lines == ['return ((f)()) if (False) else (None)']


@pytest.mark.parametrize('fn_txt,expected_result', [
    ('((fn f [] [1 2 3 4]))',
     Vec.from_iter((1, 2, 3, 4,))),
])
def test_compiler__evaluate_vector(
        fn_txt, expected_result, initial_interpreter
):
    text = f"""\
    (ns pack.core)
    (require 'pack.core)
    (def *compile* true)
    (ns user)
    {fn_txt}
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)
    assert results[-1] == expected_result


def test_ana():
    from pack.ast import fmap
    from pack.compiler import take_pairs
    from pack.recursion import ana_f
    ana = ana_f(fmap)

    def build_sth(expr):
        match expr:
            case Cons(Sym(None, 'let*') as s,
                      Cons(Vec() as bindings, Cons(body, Nil()))):
                result = body
                for k, v in reversed(tuple(take_pairs(bindings))):
                    result = Cons(s, Cons(Vec.from_iter([k, v]), Cons(result, nil)))
                return result
            case other:
                return other

    form = read_all_forms("""\
    (do
        (let* [x 5 y 9] (* x y)))
    """)[0]

    assert ana(build_sth)(form) == read_all_forms("""
    (do
        (let* [x 5] (let* [y 9] (* x y))))
    """)[0]
