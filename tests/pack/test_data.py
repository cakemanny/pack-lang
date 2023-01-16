from pack.data import Vec, ArrayMap, Map, List, Cons, nil


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

    # calllable
    assert Vec.from_iter(range(10))(2) == 2

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


def test_subvec():

    sv = Vec.from_iter(range(10)).subvec(2)
    assert len(sv) == 8
    assert sv[4] == list(sv)[4] == 6
    assert sv[-4] == list(sv)[-4]
    assert list(sv) == list(range(2, 10))
    assert list(iter(sv)) == list(sv)

    sv2 = Vec.from_iter([1, 2, 3, 4, 5, 6]).subvec(2, 4)
    assert repr(sv2) == '[3 4]'

    v = Vec.from_iter([0, 1, 2, 3, 4, 5])
    assert v.subvec(0, 0) == Vec.empty()
    assert Vec.empty() == v.subvec(0, 0)

    assert v.subvec(0, 6) == v
    assert v == v.subvec(0, 6)

    assert v[2:4] == v.subvec(2, 4)
    assert v[2:4][1:] == v.subvec(3, 4)
    assert v[2:4][1:] == Vec.from_iter([3])
    assert v[::2] == Vec.from_iter([0, 2, 4])
    assert v[-2:-1] == Vec.from_iter([4])

    assert len(v[100:101]) == 0
    assert len(v[0:20]) == 6

    sv = v.subvec(1, 5)
    assert len(sv) == 4
    assert Vec.from_iter(reversed(sv)) == Vec.from_iter([4, 3, 2, 1])

    # Callable
    assert v.subvec(2)(2) == 4


def test_subvec__addition():
    # Addable
    v = Vec.from_iter([0, 1, 2, 3, 4, 5])

    assert Vec.empty() + v.subvec(1, 3) == Vec.from_iter([1, 2])
    assert v.subvec(1, 3) + Vec.empty() == Vec.from_iter([1, 2])

    assert v.subvec(2) + v.subvec(4) == Vec.from_iter([2, 3, 4, 5, 4, 5])
    assert v.subvec(2) + v == Vec.from_iter([2, 3, 4, 5] + list(v))
    assert v + v.subvec(2) == Vec.from_iter(list(v) + [2, 3, 4, 5])


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
