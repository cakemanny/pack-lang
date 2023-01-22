from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pack.recursion import cata_n, cata_f


E = TypeVar('E')


@dataclass(frozen=True, slots=True)
class List(Generic[E]):
    pass


@dataclass(frozen=True, slots=True)
class Cons(List):
    hd: Any
    tl: E


@dataclass(frozen=True, slots=True)
class Nil(List):
    pass


def fmap_list(f, fa):
    match fa:
        case Cons(hd, a):
            return Cons(hd, f(a))
        case Nil():
            return Nil()


def test_cata_n():

    def alg(lst):
        match lst:
            case Cons(i, tl):
                return Cons(2 * i, tl)
            case Nil():
                return lst

    xs = Cons(5, Cons(5, Cons(5, Cons(5, Nil()))))

    cata = cata_n(fmap_list)

    assert cata(alg, 2)(xs) == Cons(10, Cons(10, Cons(5, Cons(5, Nil()))))
    assert cata(alg, 100)(xs) == Cons(10, Cons(10, Cons(10, Cons(10, Nil()))))
    assert cata(alg, 0)(xs) == Cons(5, Cons(5, Cons(5, Cons(5, Nil()))))


# This doesn't really test cata_f, so to say - but it is a nice/simple example
# of using a higher order carrier
def test_cata_f():

    # fold left in terms of cata
    def alg(lst):
        match lst:
            case Cons(i, tl):
                return lambda m: tl(i + m)
            case Nil():
                return lambda m: m

    xs = Cons(3, Cons(4, Cons(6, Cons(7, Nil()))))

    assert cata_f(fmap_list)(alg)(xs)(0) == 20
