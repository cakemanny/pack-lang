import pytest

from interp import try_read, read_ident, read_num, read_forms
from interp import Unmatched, SyntaxError, Unclosed
from interp import Num, Ident, List, Vec, Hash

NIL = tuple()


def test_read_ident():
    assert read_ident('a') == (Ident('a'), '')
    assert read_ident('z') == (Ident('z'), '')
    assert read_ident('hi ') == (Ident('hi'), ' ')


def test_read_num():
    assert read_num('1') == (Num('1'), '')
    assert read_num('1 ') == (Num('1'), ' ')
    assert read_num('123') == (Num('123'), '')
    assert read_num('123 ') == (Num('123'), ' ')


def test_try_read():

    assert try_read('()') == (List(NIL), '')
    assert try_read('( )') == (List(NIL), '')
    assert try_read('(  )') == (List(NIL), '')
    assert try_read('[  ]') == (Vec(NIL), '')
    assert try_read('{  }') == (Hash(NIL), '')
    assert try_read('{1 2 3 4 5 6}') == (
        Hash((
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
