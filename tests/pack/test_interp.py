import pytest

from pack.interp import FileString
from pack.interp import (
    try_read, read_sym, read_num, read_str, read_forms, read_all_forms,
    Reader
)
from pack.interp import Unmatched, SyntaxError, Unclosed
from pack.interp import Sym, Keyword, Vec, ArrayMap, Map, List, Cons, nil
from pack.interp import SemanticError, RecurError, EvalError
from pack.interp import Interpreter, Var, Fn, expand_and_evaluate_forms


class Any:
    "makes some assertions a little nicer"
    def __init__(self, type=None):
        self.type = type

    def __eq__(self, o):
        if self.type is not None:
            return isinstance(o, self.type)
        return True


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
    assert v.xs[0].height == 1
    assert v.xs[1].height == 1
    assert v.xs[0].xs[0].height == 0
    assert v.xs[1].xs[0].height == 0
    assert v[2] == 2
    assert v[33] == 33
    assert v[600] == 600
    assert v[1024] == 1024
    assert v[1028] == 1028
    assert v[-1] == 1029
    assert v[-2] == 1028


def test_vec__conj():
    assert Vec.empty().conj(1) == Vec.from_iter([1])

    assert Vec.from_iter(range(32)).conj(1) == Vec.from_iter(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            1]
    )

    assert Vec.from_iter(range(33)).conj(1) == Vec.from_iter(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
         32, 1]
    )

    assert Vec.from_iter(range(64)).conj(1) == Vec.from_iter(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         1]
    )
    for i in range(1022, 1028):
        assert Vec.from_iter(range(i)).conj(i) == \
            Vec.from_iter(range(i + 1))


def test_vec__add():
    assert Vec.empty() + Vec.empty() == Vec.empty()

    assert Vec.from_iter([1, 2]) + Vec.empty() == Vec.from_iter([1, 2])

    assert Vec.empty() + Vec.from_iter([1, 2]) == Vec.from_iter([1, 2])

    for i in range(0, 2000, 50):
        assert Vec.from_iter(range(i)) + Vec.from_iter(range(i, 2000)) \
            == Vec.from_iter(range(2000))


def test_vec__iter():
    v = Vec.from_iter(range(3000))
    assert Vec.from_iter(v) == v

    assert Vec.from_iter(reversed(Vec.from_iter(reversed(v)))) == v


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


def test_filestring():

    fs = FileString('a', file='fake.pack', lineno=1, col=0)
    sym, new_fs = read_sym(fs)
    assert sym, new_fs == (Sym(None, 'a'), '')
    assert new_fs.file == 'fake.pack'
    assert new_fs.lineno == 1
    assert new_fs.col == 1

    assert sym.n.file == 'fake.pack'
    from pack.interp import location_from
    assert location_from(sym) == ('fake.pack', 1, 0)

    new_fs = fs[:]
    assert fs.file == 'fake.pack'
    assert new_fs.lineno == 1
    assert new_fs.col == 0

    fs = FileString('(1\n 2\n 3)\n"more"', file='fake.pack', lineno=1, col=0)
    xs, remaining = try_read(fs)
    assert remaining == '\n"more"'
    assert remaining.file == 'fake.pack'
    assert remaining.lineno == 3
    assert remaining.col == 3

    fs = FileString('some/crazy', file='fake.pack', lineno=1, col=0)
    from pack.interp import split_ident
    some, crazy = split_ident(fs)
    assert isinstance(some, FileString)
    assert isinstance(crazy, FileString)


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
        assert try_read('(')

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
    assert try_read(';hi\n') == (Reader.NOTHING, '')
    assert try_read(';hi') == (Reader.NOTHING, '')


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


# -------------
#  Compiler
# -------------


def test_deduce_scope():
    from pack.interp import ana, create_deduce_scope_coalg

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
    from pack.interp import remove_quote

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
    from pack.interp import ana, take_pairs, Nil

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
