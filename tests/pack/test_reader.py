import pytest

from pack.data import Sym, Vec, ArrayMap, Cons, nil
from pack.exceptions import Unmatched, SyntaxError, Unclosed
from pack.reader import FileString
from pack.reader import (
    try_read, read_sym, read_num, read_str, read_forms, read_all_forms,
    Reader, split_ident, location_from
)


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
        try_read('  )')

    with pytest.raises(SyntaxError):
        try_read('[  )')

    with pytest.raises(SyntaxError):
        try_read("(let [[x y] [5 6] x)")

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
