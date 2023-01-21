from pack.interp import Cons, Nil
from pack.recursion import cata_n


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
