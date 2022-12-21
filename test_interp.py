import pytest

from interp import try_read, read_ident, read_num, read_forms, read_all_forms
from interp import Unmatched, SyntaxError, Unclosed
from interp import Num, Sym, List, Vec, ArrayMap, Map

NIL = tuple()


def test_read_ident():
    assert read_ident('a') == (Sym(None, 'a'), '')
    assert read_ident('z') == (Sym(None, 'z'), '')
    assert read_ident('hi ') == (Sym(None, 'hi'), ' ')
    assert read_ident('h9 ') == (Sym(None, 'h9'), ' ')
    assert read_ident('🫣🫒 😈') == (Sym(None, '🫣🫒'), ' 😈')
    assert read_ident('ns/n') == (Sym('ns', 'n'), '')
    assert read_ident('ns//') == (Sym('ns', '/'), '')

    assert read_ident('ns/') == (Sym(None, 'ns/'), '')


def test_read_num():
    assert read_num('1') == (Num('1'), '')
    assert read_num('1 ') == (Num('1'), ' ')
    assert read_num('123') == (Num('123'), '')
    assert read_num('123 ') == (Num('123'), ' ')


def test_try_read():
    assert try_read('()') == (List(NIL), '')
    assert try_read('( )') == (List(NIL), '')
    assert try_read('(  )') == (List(NIL), '')
    assert try_read('[]') == (Vec(NIL), '')
    assert try_read('[  ]') == (Vec(NIL), '')
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

    assert try_read('(1)') == (List((Num('1'),)), '')

    assert try_read('(\n1\n2\n3\n)') == (
        List((Num('1'), Num('2'), Num('3'),)), ''
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

    m2 = ArrayMap.from_iter((('b', 2), ('a', 1)))
    assert m == m2
    assert m2 == m

    m3 = ArrayMap.from_iter((('a', 1), ('b', 3)))
    assert m != m3
    assert m3 != m


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


def test_hamt_2():

    m = Map.empty()
    for i in range(0, 100):
        m = m.assoc(i, i)

    assert len(m) == 100

    m = Map.empty()
    for i in range(0, 100):
        m = m.assoc(chr(i), i)

    assert len(m) == 100


def test_try_read__reader_macros():
    assert try_read("'(1 2)") == try_read("(quote (1 2))")


@pytest.mark.parametrize('line_values,expected_forms', [
    (['1'], (Num('1'),)),
    (['[1', '2]'], (Vec((Num('1'), Num('2'),)),)),
    (['(', ')'], (List(NIL),)),
    (['(', ') []'], (List(NIL), Vec(NIL),)),
    (['()', '[]'], (List(NIL),)),  # second line doesn't happen yet
    (['(', ') [', ']'], (List(NIL), Vec(NIL),)),
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
