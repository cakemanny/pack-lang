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
            (set! x 5)
            (set! y 9)
            (* x y)))
    """)[0]


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
        (set! x 2)
        (set! y 3)
        (while-true
            (if (= x 0)
                (do
                    (set! __t.1 y)
                    (break))
                (do
                    (set! x__t.2 (- x 1))
                    (set! y__t.3 (* x y))
                    (set! x x__t.2)
                    (set! y y__t.3)
                    (continue))))
        __t.1)
    """)[0]


def test_hoist_statements():
    from pack.recursion import cata_f
    from pack.compiler import fmap_setbang, create_hoist_statements

    form = read_all_forms("""\
    (. (do (raise s1) (raise s2) e) n)
    """)[0]

    assert cata_f(fmap_setbang)(create_hoist_statements())(form) == \
        read_all_forms("""\
    (do (raise s1) (raise s2) (. e n))
    """)[0]

    form = read_all_forms("""\
    (do (raise s1) (do (raise s2) e))
    """)[0]

    assert cata_f(fmap_setbang)(create_hoist_statements())(form) == \
        read_all_forms("""\
    (do (raise s1) (raise s2) e)
    """)[0]

    form = read_all_forms("""\
    (if (do (raise e) true) nil)
    """)[0]

    assert cata_f(fmap_setbang)(create_hoist_statements())(form) == \
        read_all_forms("""\
    (do
        (raise e)
        (if true nil))
    """)[0]

    form = read_all_forms("""\
    (raise (do (raise e) e2))
    """)[0]

    assert cata_f(fmap_setbang)(create_hoist_statements())(form) == \
        read_all_forms("""\
    (do (raise e) (raise e2))
    """)[0]

    form = read_all_forms("""\
    (set! n (do (raise e) nil))
    """)[0]

    assert cata_f(fmap_setbang)(create_hoist_statements())(form) == \
        read_all_forms("""\
    (do (raise e) (set! n nil))
    """)[0]

    form = read_all_forms("""\
    ((do (raise e) f) 1 2)
    """)[0]

    assert cata_f(fmap_setbang)(create_hoist_statements())(form) == \
        read_all_forms("""\
    (do (raise e) (do (do nil)) (f 1 2))
    """)[0]

    form = read_all_forms("""\
    (f (do (raise e) 1) 2)
    """)[0]

    # we will clean this up, the (do nil), etc
    assert cata_f(fmap_setbang)(create_hoist_statements())(form) == \
        read_all_forms("""\
    (do (do (raise e) (do nil)) (f 1 2))
    """)[0]

    if False:
        # non-commuting statements

        form = read_all_forms("""\
        (if (if x true (do (raise e) nil)) nil)
        """)[0]

        assert cata_f(fmap_setbang)(create_hoist_statements())(form) == \
            read_all_forms("""\
        (do
            (set! __t.1 (if x true (do (raise e) nil)))
            (if __t.1
                nil))
        """)[0]


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
    (ns user)
    {fn_txt}
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)
    assert results[-1] == Any(Fn)

    fn = results[-1]
    lines = compile_fn(fn, interp, mode='lines')

    assert lines == expected_lines


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