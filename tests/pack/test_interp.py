import pytest

from pack.data import Sym, Keyword, Vec, ArrayMap, Map
from pack.exceptions import SemanticError, EvalError, SyntaxError
from pack.interp import RecurError
from pack.reader import FileString, read_all_forms
from pack.interp import Interpreter, Var, Fn, expand_and_evaluate_forms


class Any:
    "makes some assertions a little nicer"
    def __init__(self, type=None):
        self.type = type

    def __eq__(self, o):
        if self.type is not None:
            return isinstance(o, self.type)
        return True


# -------------
#  Interpreter
# -------------


def test_to_module_path():
    from pack.interp import to_module_path
    assert to_module_path('example') == 'example.pack'
    assert to_module_path('user') == 'user.pack'
    assert to_module_path('pack.core') == 'pack/core.pack'

    with pytest.raises(ValueError):
        assert to_module_path('.hi')
    with pytest.raises(ValueError):
        assert to_module_path('hi.')
    with pytest.raises(ValueError):
        assert to_module_path('')
    with pytest.raises(ValueError):
        assert to_module_path('user/example')


@pytest.fixture
def initial_interpreter():
    return Interpreter()


def test_expand_and_evaluate__ns(initial_interpreter):
    text = """\
    (ns pack.core)
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'pack.core'
    assert results == [None]


def test_expand_and_evaluate__ns_error(initial_interpreter):
    text = """\
    (ns "pack.core")
    """
    forms = read_all_forms(text)

    with pytest.raises(SemanticError) as exc_info:
        expand_and_evaluate_forms(forms, initial_interpreter)

    assert "must be a simple symbol" in str(exc_info.value)


def test_expand_and_evaluate__syntax(initial_interpreter):
    text = """\
    (ns pack.core)
    :a-keyword
    'a-symbol
    "a string"
    -1
    +1
    -1.2
    +1.2
    [1 2 3]
    {:a 1 :b 2}
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'pack.core'
    assert results == [
        None,
        Keyword(None, 'a-keyword'),
        Sym(None, 'a-symbol'),
        "a string",
        -1, 1, -1.2, 1.2, Vec.from_iter([1, 2, 3]),
        Map.empty().assoc(Keyword(None, 'a'), 1).assoc(Keyword(None, 'b'), 2)
    ]


def test_expand_and_evaluate__functionality(initial_interpreter):
    text = """\
    (ns pack.core)
    (:a {:a 1 :b 2})
    ({:a 1 :b 2} :a)
    ({:a 1 :b 2} :c 3)
    ([1 2 7] 2)
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'pack.core'
    assert results == [
        None,
        1,
        1,
        3,
        7,
    ]


def test_expand_and_evaluate__if(initial_interpreter):
    text = """\
    (ns pack.core)
    (if 1 2)
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'pack.core'
    assert results == [None, 2]


def test_expand_and_evaluate__def_var_fn(initial_interpreter):
    text = """\
    (ns pack.core)

    (def not (fn not [x] (if x false true)))
    (not false)
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'pack.core'
    assert interp.namespaces['pack.core'].defs['not']
    assert results == [None, Var(Sym('pack.core', 'not'), Any(Fn)), True]

    assert results[1].value.env == Map.empty()


@pytest.mark.parametrize('prog_txt', [
    '(fn [1] nil)',  # non name
    '(fn f [1] nil)',  # non name
    '(fn user/f [] nil)',  # non-simple name fn
    '(fn [user/x] nil)',  # non-simple param
    '(fn f [user/x] nil)',  # non-simple param
])
def test_expand_and_evaluate__fn__syntax(initial_interpreter, prog_txt):

    forms = read_all_forms(f"""\
    (ns pack.core)
    {prog_txt}
    """)

    with pytest.raises(SyntaxError):
        expand_and_evaluate_forms(
            forms, initial_interpreter
        )


@pytest.mark.parametrize('prog_txt', [
    '(def)',  # obvs
    '(def x nil nil)',  # too many init values
    '(if 1 (def x))',  # also non top level
    # '(fn [] (def x))',  # inside fn
])
def test_expand_and_evaluate__def_error(initial_interpreter, prog_txt):
    forms = read_all_forms(f"""\
    (ns pack.core)
    {prog_txt}
    """)
    with pytest.raises(SyntaxError):
        expand_and_evaluate_forms(
            forms, initial_interpreter
        )


def test_expand_and_evaluate__def_error_with_fn(initial_interpreter):
    # Ideally this would return a SyntaxError instead
    # but we will come back to that
    forms = read_all_forms("""\
    (ns pack.core)
    (fn [] (def x))
    """)
    with pytest.raises(ValueError):
        expand_and_evaluate_forms(forms, initial_interpreter)


def test_expand_and_evaluate__resolve_error(initial_interpreter):
    import textwrap
    text = textwrap.dedent("""\
    (ns pack.core)
    does-not-exist-yet
    """)
    print(text.splitlines())
    forms = read_all_forms(FileString(text, "fake.pack", 1, 0))

    with pytest.raises(EvalError) as exc_info:
        expand_and_evaluate_forms(forms, initial_interpreter)

    assert "does-not-exist-yet" in str(exc_info.value)
    assert exc_info.value.location == ("fake.pack", 2, 0)

    text = textwrap.dedent("""\
    (ns pack.core)
    no-such-ns/does-not-exist-yet
    """)
    forms = read_all_forms(FileString(text, "fake.pack", 1, 0))

    with pytest.raises(EvalError) as exc_info:
        expand_and_evaluate_forms(forms, initial_interpreter)

    assert "no-such-ns" in str(exc_info.value)
    assert exc_info.value.location == ("fake.pack", 2, 0)


def test_expand_and_evaluate__fn(initial_interpreter):
    text = """\
    (ns pack.core)
    ((fn [x] (if x false true)) false)
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'pack.core'
    assert results == [None, True]


def test_expand_and_evaluate__fn_rest_args(initial_interpreter):
    text = """\
    (ns pack.core)

    (def first (fn [xs] (. xs hd)))
    (def rest (fn [xs] (. xs tl)))
    (def null? (fn [lst] (if lst false true)))
    (def foldl
        (fn [func accum lst]
            (if (null? lst)
                accum
                (recur func (func accum (first lst)) (rest lst)))))

    (import operator)

    (def +
        (fn [& numbers]
            (foldl (. operator add) 0 numbers)))
    (+ 1 2 3 4)
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)
    assert results[-1] == 10


def test_expand_and_evaluate__redefining(initial_interpreter):
    # Being able to redefine stuff in the repl can be useful
    text = """\
    (ns pack.core)

    (def x 20)
    (def y (fn [] x))
    (y) ; 20
    (def x 40)
    (y) ; 40
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert results[-3] == 20
    assert results[-1] == 40


def test_expand_and_evaluate__4(initial_interpreter):
    text = """\
    (ns pack.core)

    (def not (fn not [x] (if x false true)))
    {(not false) 1 (not (not false)) 0}
    [1 2 3 (not 3)]
    (ns user)
    (pack.core/not false)
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'user'
    assert interp.namespaces['pack.core'].defs['not']
    assert results == [
        None,
        Var(Sym('pack.core', 'not'), Any(Fn)),
        Map.empty().assoc(True, 1).assoc(False, 0),
        Vec.from_iter([1, 2, 3, False]),
        None,
        True,
    ]


def test_expand_and_evaluate__recur(initial_interpreter):
    text = """\
    (ns pack.core)
    (import operator)
    (def + (. operator add))

    (def first (fn [xs] (. xs hd)))
    (def rest (fn [xs] (. xs tl)))
    (def null? (fn [lst] (if lst false true)))
    (def foldl
        (fn [func accum lst]
            (if (null? lst)
                accum
                (recur func (func accum (first lst)) (rest lst)))))
    (foldl + 0 '(1 2 3 4))
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'pack.core'
    assert results[-1:] == [
        10
    ]


def test_expand_and_evaluate__loop_recur(initial_interpreter):
    text = """\
    (ns pack.core)
    (import operator)
    (def = (. operator eq))
    (def - (. operator sub))
    (def * (. operator mul))
    (def zero? (fn [n] (= n 0)))
    (def dec (fn [n] (- n 1)))

    (def factorial
      (fn [n]
        (loop [cnt n acc 1]
           (if (zero? cnt)
                acc
              (recur (dec cnt) (* acc cnt))))))
    (factorial 5)
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert results[-1:] == [
        5 * 4 * 3 * 2
    ]


def test_expand_and_evaluate__recur_error(initial_interpreter):
    text = """\
    (ns pack.core)
    (recur 1 2)
    """
    forms = read_all_forms(text)

    with pytest.raises(RecurError):
        expand_and_evaluate_forms(forms, initial_interpreter)


def test_expand_and_evaluate__recur__non_tail(initial_interpreter):
    text = """\
    (ns pack.core)
    (import operator)
    (def = (. operator eq))
    (def - (. operator sub))
    (def + (. operator add))
    (loop [x 10 y 0]
        (if (= x 0)
            y
            (+ (- x 1) (recur y x))))
    """
    forms = read_all_forms(text)

    with pytest.raises(SyntaxError) as exc_info:
        expand_and_evaluate_forms(forms, initial_interpreter)
    assert 'recur in non-tail' in str(exc_info.value)


def test_expand_and_evaluate__loop_recur__incorrect_num_args(initial_interpreter):
    text = """\
    (ns pack.core)
    (import operator)
    (def = (. operator eq))
    (def - (. operator sub))
    (def * (. operator mul))
    (def zero? (fn [n] (= n 0)))
    (def dec (fn [n] (- n 1)))

    (def factorial
      (fn [n]
        (loop [cnt n acc 1]
           (if (zero? cnt)
                acc
              (recur (dec cnt))))))
    (factorial 5)
    """
    forms = read_all_forms(text)

    with pytest.raises(SemanticError) as exc_info:
        expand_and_evaluate_forms(forms, initial_interpreter)

    assert "not enough operands given to recur: expected 2, got 1" in str(exc_info.value)


def test_expand_and_evaluate__eval(initial_interpreter):
    text = """\
    (ns pack.core)
    (import pack.interp)
    (def eval (. pack.interp rt_eval))
    (import operator)
    (def + (. operator add))

    (eval '(+ 1 2))
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert results[-1:] == [
        3
    ]


def test_expand_quasi_quotes(initial_interpreter):
    from pack.interp import expand_quasi_quotes

    forms = read_all_forms("""(ns pack.core)""")
    _, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    form = read_all_forms(" `~a ")[0]
    expanded = expand_quasi_quotes(form, interp)

    assert expanded == read_all_forms("a")[0]

    form = read_all_forms(" `(a b) ")[0]
    expanded = expand_quasi_quotes(form, interp)

    assert expanded == read_all_forms(
        """
        (pack.core/concat
            (pack.core/list (quote pack.core/a))
            (pack.core/list (quote pack.core/b)))
        """
    )[0]

    form = read_all_forms(" `[a b] ")[0]
    expanded = expand_quasi_quotes(form, interp)

    assert expanded == read_all_forms(
        """
        (pack.core/apply pack.core/vector
            (pack.core/concat
                (pack.core/list (quote pack.core/a))
                (pack.core/list (quote pack.core/b))))
        """
    )[0]

    form = read_all_forms(" `{:a a} ")[0]
    expanded = expand_quasi_quotes(form, interp)
    assert expanded == read_all_forms(
        """
        (pack.core/apply pack.core/hash-map
            (pack.core/concat
                (pack.core/list :a)
                (pack.core/list (quote pack.core/a))))
        """
    )[0]

    # Check that we expand withing data forms
    form = read_all_forms(" [`a `b] ")[0]
    expanded = expand_quasi_quotes(form, interp)
    assert expanded == read_all_forms(
        """
        [(quote pack.core/a) (quote pack.core/b)]
        """
    )[0]

    form = read_all_forms(" {`a `b} ")[0]
    expanded = expand_quasi_quotes(form, interp)
    assert expanded == read_all_forms(
        """
        {(quote pack.core/a) (quote pack.core/b)}
        """
    )[0]


def test_expand_and_evaluate__quoting(initial_interpreter):
    text = """\
    (ns pack.core)
    (import operator)
    (def + (. operator add))
    (def not (. operator not_))
    (def first (fn [xs]
        (if xs (. xs hd) nil)))
    (def rest (fn [xs]
        (if xs (. xs tl) nil)))
    (def list (fn [& elems] elems))

    (import pack.interp)
    (def apply (. pack.interp rt_apply))

    (def concat
        (fn concat [& elems]
            (if elems
                (if (rest elems)
                    (+ (first elems) (apply concat (rest elems)))
                    (first elems))
                nil)))
    ;; above is just machinery for the test

    (def c 3)
    (def zz '(1 2 3))
    `a
    `(a b c)
    `(a b ~c)
    `(a b ~@zz)
    ``a
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert results[-5:] == [
        read_all_forms("pack.core/a")[0],
        read_all_forms("(pack.core/a pack.core/b pack.core/c)")[0],
        read_all_forms("(pack.core/a pack.core/b 3)")[0],
        read_all_forms("(pack.core/a pack.core/b 1 2 3)")[0],
        read_all_forms("(quote pack.core/a)")[0],
    ]


def test_expand_and_evaluate__import_nested_fn(initial_interpreter):
    # I'm not really sure what is being tested in this test.
    # I think this is an old one from when they were just numbered
    text = """\
    (ns pack.core)
    (import builtins)
    (def str (fn str [arg] ((. builtins str) arg)))

    (ns user)
    (pack.core/refer 'pack.core)
    (str 'example)
    (def f (fn [x] (fn [y] (str y x))))
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    import builtins
    assert results == [
        None,
        Var(Sym('py', 'builtins'), builtins),
        Var(Sym('pack.core', 'str'), Any(Fn)),
        None,
        None,
        "example",
        Var(Sym('user', 'f'), Any(Fn)),
    ]
    assert results[2].value.env == ArrayMap.empty().assoc(
        Sym(None, 'builtins'), Any(Var)
    )
    assert results[6].value.env == ArrayMap.empty().assoc(
        Sym(None, 'str'), Any(Var)
    )


def test_expand_and_evaluate__require(initial_interpreter):
    text = """\
    (ns pack.core)
    (ns user)
    (require 'tests.example)
    tests.example/hello
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'user'
    assert results
    assert results[-2:] == [None, 92]


def test_expand_and_evaluate__require__broken_module(initial_interpreter):
    # It's important that the interpreter continues to run when
    # a required module breaks, and moreover, it's important that the
    # namespace doesn't change
    text1 = """\
    (ns pack.core)
    (ns user)
    """
    initial_forms = read_all_forms(text1)
    results, interp = expand_and_evaluate_forms(initial_forms, initial_interpreter)

    text2 = """\
    (require 'tests.broken-1)
    """
    with pytest.raises(EvalError):
        results, interp = expand_and_evaluate_forms(read_all_forms(text2), interp)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'user'
    assert 'tests.broken-1' not in interp.namespaces


def test_expand_and_evaluate__require__broken_module2(initial_interpreter):
    text1 = """\
    (ns pack.core)
    (ns user)
    """
    initial_forms = read_all_forms(text1)
    results, interp = expand_and_evaluate_forms(initial_forms, initial_interpreter)

    text2 = """\
    (require 'tests.broken-2)
    """
    with pytest.raises(EvalError):
        results, interp = expand_and_evaluate_forms(read_all_forms(text2), interp)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'user'
    assert 'tests.broken-2' not in interp.namespaces


def test_expand_and_evaluate__raise(initial_interpreter):
    text = """\
    (ns pack.core)
    (ns user)
    (import builtins)
    (raise ((. builtins Exception) "hurray"))
    """
    forms = read_all_forms(text)

    with pytest.raises(Exception) as exc_info:
        expand_and_evaluate_forms(forms, initial_interpreter)
    assert 'hurray' in str(exc_info.value)


def test_expand_and_evaluate__do(initial_interpreter):
    text = """\
    (ns pack.core)
    (ns user)
    (import builtins)
    (do
        ((. builtins print) "I'm debugging")
        (def x 5)
        ((. builtins print) "x =" x))
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'user'
    import builtins
    assert results == [
        None,
        None,
        Var(Sym('py', 'builtins'), builtins),
        None
    ]


def test_expand_and_evaluate__raise__error(initial_interpreter):
    text = """\
    (ns pack.core)
    (ns user)
    (raise)
    """
    forms = read_all_forms(text)

    with pytest.raises(SyntaxError):
        results, interp = expand_and_evaluate_forms(forms, initial_interpreter)


def test_expand_and_evaluate__let(initial_interpreter):
    text = """\
    (ns pack.core)
    (let* [x 1] x)
    (let* [x 1 y 2] [x y])
    (let* [x 1 y ((. x __mul__) 2)] [x y])
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'pack.core'
    assert results == [
        None,
        1,
        Vec.from_iter([1, 2]),
        Vec.from_iter([1, 2]),
    ]


def test_expand_and_evaluate__let__fn(initial_interpreter):
    text = """\
    (ns pack.core)
    (import operator)
    (def * (. operator mul))
    (def + (. operator add))
    (def f
        (fn [x]
            (let* [y (* x 2)
                   z (+ y 1)]
                (+ z 4))))
    (f 3)
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert results[-1] == 11


@pytest.mark.parametrize('prog_txt', [
    '(let* [x 1 y] [x y])',  # uneven forms
    '(let* [{:a a} {:a 1}] [a a])',  # destructuring
    '(loop [x 1 y] [x y])',  # uneven forms
    '(loop [{:a a} {:a 1}] [a a])',  # destructuring
])
def test_expand_and_evaluate__let__error(initial_interpreter, prog_txt):

    with pytest.raises(SyntaxError):
        results, interp = expand_and_evaluate_forms(
            read_all_forms(f"""\
            (ns pack.core)
            {prog_txt}
            """),
            initial_interpreter
        )


def test_defmacro(initial_interpreter):
    text = """\
    (ns pack.core)
    (import builtins)

    (def error
        (fn [msg]
            (raise ((. builtins Exception) msg))))
    (import operator)
    (def + (. operator add))

    (import pack.interp)
    (def List (. pack.interp List))

    (def list?
        (fn
            [x]
            ((. builtins isinstance) x List)))
    (def list (fn [& elements] elements))
    (def first
        (fn [xs]
            (do (if (list? xs)
                (. xs hd)))))
    (def rest
        (fn [xs]
            (if (list? xs)
                (. xs tl))))

    (ns user)
    (pack.core/refer 'pack.core)

    (def ->
        (fn
            [value proc]
            (if (list? proc)
                (list (first proc) value (first (rest proc)))
                (error "second argument to -> must be an s-expr"))))
    (import pack.interp)
    ((. pack.interp set_macro) (var ->))

    (-> 7
        (+ 7))
    """
    forms = read_all_forms(FileString(text, "fake.pack", 1, 0))

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'user'
    assert results
    assert results[-2:] == [None, 14]


def test_macro_expansion(initial_interpreter):
    text = """\
    (require 'pack.core)
    (ns user)
    (import pack.interp)

    (def m1 (fn m1 [arg-form] (str arg-form)))
    ((. pack.interp set_macro) (var m1))

    (def m2 (fn m2 [arg-form] (str "m2-" arg-form)))
    ((. pack.interp set_macro) (var m2))

    ; m2 shall not be expanded because m1 converts it to a string
    (m1 (m2 hey))
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'user'

    assert results[-1:] == [
        "(m2 hey)"
    ]


def test_macro_expansion__repeated(initial_interpreter):
    text = """\
    (require 'pack.core)
    (ns user)
    (import pack.interp)

    (def m1 (fn m1 [arg-form] `(m2 ~arg-form)))
    ((. pack.interp set_macro) (var m1))

    (def m2 (fn m2 [arg-form] (str "m2-" arg-form)))
    ((. pack.interp set_macro) (var m2))

    (def m3 (fn m3 [arg-form] (str "m3-" arg-form)))
    ((. pack.interp set_macro) (var m3))

    ; m3 shall not be expanded because m2 converts m3 to a string
    (m1 (m3 hey))
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'user'

    assert results[-1:] == [
        "m2-(m3 hey)"
    ]


def test_expand_and_evaluate__refer(initial_interpreter):
    text = """\
    (ns pack.core)
    (ns example)
    (def xxx 5)
    (ns user)
    (pack.core/refer 'example)
    xxx
    (var xxx)
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert results[-2:] == [
        5,
        Var(Sym('example', 'xxx'), 5)
    ]


def test_expand_and_evaluate__import__error(initial_interpreter):
    text1 = """\
    (ns pack.core)
    (ns user)
    """
    initial_forms = read_all_forms(text1)
    results, interp = expand_and_evaluate_forms(initial_forms, initial_interpreter)

    text2 = """\
    (import i_hope_they_dont_add_this_module_to_python)
    """
    with pytest.raises(EvalError):
        expand_and_evaluate_forms(read_all_forms(text2), interp)

    assert interp.current_ns.name == 'user'
