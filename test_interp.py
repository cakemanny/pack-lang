import pytest

from interp import try_read, read_ident, read_num, read_forms, read_all_forms
from interp import Unmatched, SyntaxError, Unclosed
from interp import Num, Sym, Vec, ArrayMap, Map, List, Cons, nil
from interp import Interpreter, Var, expand_and_evaluate_forms

NIL = tuple()


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


def test_read_ident():
    assert read_ident('a') == (Sym(None, 'a'), '')
    assert read_ident('z') == (Sym(None, 'z'), '')
    assert read_ident('hi ') == (Sym(None, 'hi'), ' ')
    assert read_ident('h9 ') == (Sym(None, 'h9'), ' ')
    assert read_ident('🫣🫒 😈') == (Sym(None, '🫣🫒'), ' 😈')
    assert read_ident('ns/n') == (Sym('ns', 'n'), '')
    assert read_ident('ns//') == (Sym('ns', '/'), '')

    assert read_ident('ns/') == (Sym(None, 'ns/'), '')

    assert read_ident('py/sys.stdin') == (Sym('py', 'sys.stdin'), '')


def test_read_num():
    assert read_num('1') == (Num('1'), '')
    assert read_num('1 ') == (Num('1'), ' ')
    assert read_num('123') == (Num('123'), '')
    assert read_num('123 ') == (Num('123'), ' ')
    assert read_num('123.3 ') == (Num('123.3'), ' ')

    with pytest.raises(SyntaxError):
        read_num('12.3.3 ')


def test_try_read():
    assert try_read('()') == (nil, '')
    assert try_read('( )') == (nil, '')
    assert try_read('(  )') == (nil, '')
    assert try_read('[]') == (Vec(), '')
    assert try_read('[  ]') == (Vec(), '')
    assert try_read('[1 2 3]') == (Vec([Num('1'), Num('2'), Num('3')]), '')
    assert try_read('{  }') == (ArrayMap.from_iter(NIL), '')
    assert try_read('{1 2 3 4 5 6}') == (
        ArrayMap.from_iter((
            (Num('1'), Num('2')),
            (Num('3'), Num('4')),
            (Num('5'), Num('6'))
        )),
        ''
    )

    with pytest.raises(Unmatched):
        try_read('[  )')

    # incomplete form
    with pytest.raises(Unclosed):
        assert try_read('(') == (None, '')

    assert try_read('(1)') == (Cons(Num('1'), nil), '')

    assert try_read('(\n1\n2\n3\n)') == (
        Cons(Num('1'), Cons(Num('2'), Cons(Num('3'), nil))), ''
    )

    with pytest.raises(SyntaxError) as exc_info:
        try_read('{ 1 }')
    assert 'even number of forms' in str(exc_info.value)

    assert try_read('-1')[0] == Num('-1')
    assert try_read('+1')[0] == Num('+1')

    assert try_read(';hi\n1') == (None, '1')
    assert try_read(';hi\n') == (None, '')
    assert try_read(';hi') == (None, '')


def test_vec():
    # Leaf vector
    assert len(Vec(range(31))) == 31
    assert Vec(range(31))[30] == 30
    assert Vec(range(31))[-1] == 30
    assert Vec(range(31))[-2] == 29

    assert len(Vec(range(32))) == 32

    # Two level Vector
    assert len(Vec(range(33))) == 33
    assert Vec(range(33))[32] == 32
    assert Vec(range(33))[2] == 2


def test_vec_3():
    # Three level vector
    v = Vec(range(1030))
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


def test_try_read__reader_macros():
    assert try_read("'(1 2)") == try_read("(quote (1 2))")


@pytest.mark.parametrize('line_values,expected_forms', [
    (['1'], (Num('1'),)),
    (['[1', '2]'], (Vec((Num('1'), Num('2'),)),)),
    (['(', ')'], (nil,)),
    (['(', ') []'], (nil, Vec(),)),
    (['()', '[]'], (nil,)),  # second line doesn't happen yet
    (['(', ') [', ']'], (nil, Vec(),)),
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

;; vim:ft=clojure:
"""

    with pytest.raises(Unclosed):
        assert read_all_forms(line_value) == []


# -------------
#  Interpreter
# -------------

@pytest.fixture
def initial_interpreter():
    return Interpreter(Map.empty())


def test_expand_and_evaluate__1(initial_interpreter):
    text = """\
    (ns pack.core)
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'pack.core'
    assert results == [None]


def test_expand_and_evaluate__2(initial_interpreter):
    text = """\
    (ns pack.core)
    (if 1 2)
    """
    forms = read_all_forms(text)

    results, interp = expand_and_evaluate_forms(forms, initial_interpreter)

    assert 'pack.core' in interp.namespaces
    assert interp.current_ns.name == 'pack.core'
    assert results == [None, 2]


def test_expand_and_evaluate__3(initial_interpreter):
    from interp import Fn
    text = """\
    (ns pack.core)
    ;(def not)

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


def test_expand_and_evaluate__4(initial_interpreter):
    from interp import Fn
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
        Vec([1, 2, 3, False]),
        None,
        True,
    ]


def test_expand_and_evaluate__5(initial_interpreter):
    from interp import Fn
    text = """\
    (ns pack.core)
    (import builtins)
    (def str (fn str [arg] ((. builtins str) arg)))

    (ns user)
    (str 'example)
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
    ]
    assert results[2].value.env == ArrayMap.empty().assoc(
        Sym(None, 'builtins'), Any(Var)
    )


@pytest.mark.skip
def test_expand_and_evaluate__6(initial_interpreter):
    from interp import Fn
    text = """\
    (ns pack.core)
    (import builtins)
    (def str (fn str [arg] ((. builtins str) arg)))

    ;(def require
    ;    (fn require [ns-sym]
    ;        (load-lib (str ns-sym))))

    (ns user)
    ;(require 'example)
    ;example/hello
    (str 'example)
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
        Vec([1, 2, 3, False]),
        None,
        True,
    ]
