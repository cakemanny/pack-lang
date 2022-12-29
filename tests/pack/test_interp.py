import pytest

from pack.interp import (
    try_read, read_sym, read_num, read_str, read_forms, read_all_forms
)
from pack.interp import Unmatched, SyntaxError, Unclosed
from pack.interp import Sym, Keyword, Vec, ArrayMap, Map, List, Cons, nil
from pack.interp import Interpreter, Var, Fn, expand_and_evaluate_forms


# ----------------
#  Data Structures
# ----------------


def test_nil():
    assert bool(nil) is False
    assert bool(Cons(None, nil)) is True
    assert bool(Cons(None, None)) is True


def test_list():
    assert len(nil) == 0
    assert len(Cons(1, nil)) == 1
    assert len(Cons(2, Cons(1, nil))) == 2
    assert len(Cons(2, Cons(1, None))) == 2

    assert List.from_iter([1, 2, 3]) == Cons(1, Cons(2, Cons(3, nil)))
    assert List.from_iter(range(3)) == Cons(0, Cons(1, Cons(2, nil)))

    def aux():
        yield 0
        yield 1
        yield 2
    assert List.from_iter(aux()) == Cons(0, Cons(1, Cons(2, nil)))


def test_vec():
    assert len(Vec.from_iter([])) == 0

    assert len(Vec.from_iter(range(31))) == 31
    assert Vec.from_iter(range(31))[30] == 30
    assert Vec.from_iter(range(31))[-1] == 30
    assert Vec.from_iter(range(31))[-2] == 29

    assert len(Vec.from_iter(range(32))) == 32

    # Two level Vector
    assert len(Vec.from_iter(range(33))) == 33
    assert Vec.from_iter(range(33))[32] == 32
    assert Vec.from_iter(range(33))[2] == 2


def test_vec_3():
    # Three level vector
    v = Vec.from_iter(range(1030))
    assert len(v) == 1030
    assert v.height == 2
    assert len(v.xs) == 2
    assert len(v.xs[0].xs) == 32
    assert len(v.xs[0].xs[0]) == 32
    assert len(v.xs[1].xs) == 1
    assert len(v.xs[1].xs[0]) == 6
    assert v[2] == 2
    assert v[33] == 33
    assert v[600] == 600
    assert v[1024] == 1024
    assert v[1028] == 1028
    assert v[-1] == 1029
    assert v[-2] == 1028


def test_arraymap():
    m = ArrayMap.from_iter((('a', 1), ('b', 2)))

    assert len(m) == 2
    assert list(iter(m)) == ['a', 'b']
    assert str(m) == "{'a' 1  'b' 2}"
    assert list(m.keys()) == ['a', 'b']
    assert 'a' in m.keys() and 'b' in m.keys()

    assert list(m.items()) == [('a', 1), ('b', 2)]
    assert ('a', 1) in m.items() and ('b', 2) in m.items()

    assert list(m.values()) == [1, 2]
    assert 1 in m.values() and 2 in m.values()

    m2 = ArrayMap.from_iter((('b', 2), ('a', 1)))
    assert m == m2
    assert m2 == m

    m3 = ArrayMap.from_iter((('a', 1), ('b', 3)))
    assert m != m3
    assert m3 != m
    assert len(m3) == 2

    match m3:
        case {'a': a, 'b': b}:
            assert a == 1 and b == 3
        case _:
            assert False, "match fail"


def test_arraymap_assoc():
    m = ArrayMap.from_iter((('a', 1), ('b', 2)))

    assert m.assoc('c', 3) == ArrayMap.from_iter(
        (('a', 1), ('b', 2), ('c', 3))
    )

    assert ArrayMap.empty().assoc('a', 1).assoc('b', 2) == m
    assert ArrayMap.empty().assoc('b', 2).assoc('a', 1) == m


def test_arraymap_dissoc():
    m = ArrayMap.from_iter((('a', 1), ('b', 2)))

    m4 = m.dissoc('a')
    assert len(m4) == 1
    assert list(iter(m4)) == ['b']


def test_hamt():
    nil = Map.empty()

    assert 'a' not in nil
    assert len(nil) == 0
    assert list(nil) == []

    m = nil.assoc('a', 1)
    assert 'a' in m
    assert m['a'] == 1
    assert len(m) == 1
    assert list(m) == ['a']

    m2 = m.assoc('a', 2)
    assert 'a' in m2
    assert m2['a'] == 2
    assert len(m2) == 1
    assert list(m2) == ['a']

    m3 = m.assoc('b', 2)
    assert 'a' in m3 and 'b' in m3
    assert m3['a'] == 1
    assert m3['b'] == 2
    assert len(m3) == 2
    assert set(m3) == {'a', 'b'}


def test_hamt_dissoc():
    m = Map.from_iter((('a', 1), ('b', 2)))

    m4 = m.dissoc('a')
    assert len(m4) == 1
    assert list(iter(m4)) == ['b']


def test_hamt_2():

    m = Map.empty()
    for i in range(0, 100):
        m = m.assoc(i, i)

    assert len(m) == 100
    for i in range(0, 100):
        assert m[i] == i

    m = Map.empty()
    for i in range(0, 100):
        m = m.assoc(chr(i), i)

    assert len(m) == 100
    for i in range(0, 100):
        assert m[chr(i)] == i

    for i in range(0, 100):
        m = m.dissoc(chr(i))
        assert len(m) == 99 - i


# -------------
#  Reader
# -------------


def test_read_sym():
    assert read_sym('a') == (Sym(None, 'a'), '')
    assert read_sym('z') == (Sym(None, 'z'), '')
    assert read_sym('hi ') == (Sym(None, 'hi'), ' ')
    assert read_sym('h9 ') == (Sym(None, 'h9'), ' ')
    assert read_sym('🫣🫒 😈') == (Sym(None, '🫣🫒'), ' 😈')
    assert read_sym('ns/n') == (Sym('ns', 'n'), '')
    assert read_sym('ns//') == (Sym('ns', '/'), '')

    assert read_sym('ns/') == (Sym(None, 'ns/'), '')

    assert read_sym('py/sys.stdin') == (Sym('py', 'sys.stdin'), '')

    # lambda-list keywords are symbols
    assert read_sym('&rest') == (Sym(None, '&rest'), '')


def test_read_num():
    assert read_num('1') == (1, '')
    assert read_num('1 ') == (1, ' ')
    assert read_num('123') == (123, '')
    assert read_num('123 ') == (123, ' ')
    assert read_num('123.3 ') == (123.3, ' ')

    with pytest.raises(SyntaxError):
        read_num('12.3.3 ')


def test_read_str():
    assert read_str('""') == ('', '')
    assert read_str('"hallo"') == ('hallo', '')
    assert read_str('"hallo"  some "extra"') == ('hallo', '  some "extra"')

    with pytest.raises(Unclosed):
        read_str('"hallo')

    assert read_str(r'"ha\nllo"') == ('ha\nllo', '')
    assert read_str(r'"ha\"llo"') == ('ha"llo', '')


def test_try_read():
    assert try_read('()') == (nil, '')
    assert try_read('( )') == (nil, '')
    assert try_read('(  )') == (nil, '')
    assert try_read('[]') == (Vec.empty(), '')
    assert try_read('[  ]') == (Vec.empty(), '')
    assert try_read('[1 2 3]') == (Vec.from_iter([1, 2, 3]), '')
    assert try_read('{  }') == (ArrayMap.from_iter(()), '')
    assert try_read('{1 2 3 4 5 6}') == (
        ArrayMap.from_iter((
            (1, 2),
            (3, 4),
            (5, 6)
        )),
        ''
    )

    with pytest.raises(Unmatched):
        try_read('[  )')

    # incomplete form
    with pytest.raises(Unclosed):
        assert try_read('(') == (None, '')

    assert try_read('(1)') == (Cons(1, nil), '')

    assert try_read('(\n1\n2\n3\n)') == (
        Cons(1, Cons(2, Cons(3, nil))), ''
    )

    with pytest.raises(SyntaxError) as exc_info:
        try_read('{ 1 }')
    assert 'even number of forms' in str(exc_info.value)

    assert try_read('-1')[0] == -1
    assert try_read('+1')[0] == +1


def test_try_read__reader_macros():
    assert try_read("'(1 2)") == try_read("(quote (1 2))")

    assert try_read("`(1 2)") == try_read("(quasiquote (1 2))")
    assert try_read("`(a ~b)") == try_read("(quasiquote (a (unquote b)))")
    assert try_read("`(a ~@b)") == try_read("(quasiquote (a (unquote-splicing b)))")


def test_try_read__comments():
    assert try_read(';hi\n1') == (1, '')
    assert try_read(';hi\n') == (None, '')
    assert try_read(';hi') == (None, '')


def test_read_all_forms__comments():

    assert read_all_forms("""\
    1
    ; hi
    """) == (1,)

    assert read_all_forms("""\
    ; hi
    1
    """) == (1,)
    assert read_all_forms("""\
    1 ; hi
    """) == (1,)

    assert read_all_forms("""\
    (
    1 ; my favourite number
    9 ; my least favourite number
    )
    """) == (Cons(1, Cons(9, nil)),)


@pytest.mark.parametrize('line_values,expected_forms', [
    (['1'], (1,)),
    (['[1', '2]'], (Vec.from_iter([1, 2,]),)),
    (['(', ')'], (nil,)),
    (['(', ') []'], (nil, Vec.empty(),)),
    (['()', '[]'], (nil,)),  # second line doesn't happen yet
    (['(', ') [', ']'], (nil, Vec.empty(),)),
])
def test_read_forms(line_values, expected_forms):

    lines = iter(line_values)

    def _input(prompt=''):
        return next(lines, '')
    assert read_forms(input=_input) == expected_forms


def test_read_all_forms__unclosed_file():
    line_value = """\
(ns pack.core)

(def not (fn [x] (if x false true))
; final closing paren is missing   ^
"""

    with pytest.raises(Unclosed):
        assert read_all_forms(line_value) == []


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
    return Interpreter(Map.empty())


def test_expand_and_evaluate__ns(initial_interpreter):
    text = """\
    (ns pack.core)
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'pack.core'
    assert results == [None]


def test_expand_and_evaluate__syntax(initial_interpreter):
    # Just check that the evaluator doesn't choke
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
    ;; bullshit definition of apply just for this test
    (def apply (fn [f args]
        (if (not (rest args))
            (f (first args))
            (if (not (rest (rest args)))
                (f (first args) (first (rest args)))
                (f (first args) (first (rest args)) (first (rest (rest args))))))))

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


def test_expand_and_evaluate__functionality(initial_interpreter):
    # Just check that the evaluator doesn't choke
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


class Any:
    def __init__(self, type=None):
        self.type = type

    def __eq__(self, o):
        if self.type is not None:
            return isinstance(o, self.type)
        return True


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
                (foldl func (func accum (first lst)) (rest lst)))))

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
    (not false)
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


def test_expand_and_evaluate__import_nested_fn(initial_interpreter):
    text = """\
    (ns pack.core)
    (import builtins)
    (def str (fn str [arg] ((. builtins str) arg)))

    (ns user)
    (str 'example)
    (def f (fn [x] (fn [y] (str y x))))
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    import builtins
    assert results == [
        None,
        Var(Sym(None, 'builtins'), builtins),
        Var(Sym('pack.core', 'str'), Any(Fn)),
        None,
        "example",
        Var(Sym('user', 'f'), Any(Fn)),
    ]
    assert results[2].value.env == ArrayMap.empty().assoc(
        Sym(None, 'builtins'), Any(Var)
    )
    assert results[5].value.env == ArrayMap.empty().assoc(
        Sym(None, 'str'), Any(Var)
    )


def test_expand_and_evaluate__require(initial_interpreter):
    text = """\
    (ns pack.core)
    (ns user)
    (require 'example)
    example/hello
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'user'
    assert results
    assert results[-2:] == [None, 92]


def test_expand_and_evaluate__raise(initial_interpreter):
    text = """\
    (ns pack.core)
    (ns user)
    (import builtins)
    (raise ((. builtins Exception) "hurray"))
    """
    forms = read_all_forms(text)

    with pytest.raises(Exception) as exc_info:
        results, interp = expand_and_evaluate_forms(forms, initial_interpreter)
    assert 'hurray' in str(exc_info.value)


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
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'pack.core'
    assert results == [None, 1, Vec.from_iter([1, 2])]


def test_expand_and_evaluate__let__error(initial_interpreter):

    with pytest.raises(SyntaxError):
        results, interp = expand_and_evaluate_forms(
            read_all_forms("""\
            (ns pack.core)
            (let* [x 1 y] [x y])
            """),
            initial_interpreter
        )

    with pytest.raises(SyntaxError):
        results, interp = expand_and_evaluate_forms(
            read_all_forms("""\
            (ns pack.core)
            (let* [{:a a} {:a 1}] [x y])
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
            (if (list? xs)
                (. xs hd))))
    (def rest
        (fn [xs]
            (if (list? xs)
                (. xs tl))))

    (ns user)

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
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'user'
    assert results
    assert results[-2:] == [None, 14]


@pytest.mark.skip
def test_expand_and_evaluate__refer(initial_interpreter):
    text = """\
    (ns pack.core)
    (ns example)
    (def xxx 5)
    (ns user)
    (refer 'example)
    xxx
    (var xxx)
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert results[-2] == [
        5,
        Var(Sym('example', 'xxx'), Any(Fn))
    ]
