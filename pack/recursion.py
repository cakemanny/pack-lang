"""
pack.recursion

Recursion scheme functions we are using.

Most of these have been derived from the slide from Tim Williams
talk https://github.com/willtim/recursion-schemes/

Found via the Matryoshka External Resources section:
https://github.com/precog/matryoshka#external-resources
"""
from dataclasses import dataclass
from typing import Generic, TypeVar


L = TypeVar('L')
R = TypeVar('R')


@dataclass(frozen=True)
class Left(Generic[L, R]):
    l: L


@dataclass(frozen=True)
class Right(Generic[L, R]):
    r: R


def fan_in(b_to_d, c_to_d):
    """
    (|||) ::: (b -> d) -> (c -> d) -> Either b c -> d
    (|||) = either
    """
    def aux(either_b_or_c):
        match either_b_or_c:
            case Left(b):
                return b_to_d(b)
            case Right(c):
                return c_to_d(c)
    return aux


def cata_f(fmap, unfix=lambda x: x):
    """
    generalised fold-right over any functor. takes the fmap for that functor

    cata alg = alg . fmap (cata alg) . unfix
    """
    def cata(alg):
        return lambda f: alg(fmap(cata(alg), unfix(f)))
    return cata


def ana_f(fmap, fix=lambda x: x):
    """
    generalised unfold.
    we can use it for top down traverals

    ana :: Functor f => (a -> f a) -> a -> Fix f
    ana coalg = Fix . fmap (ana coalg) . coalg
    """
    def ana(coalg):
        return lambda a: fix(fmap(ana(coalg), coalg(a)))
    return ana


def hylo_f(fmap):
    """
    unfold and then fold

    hylo :: Functor f => (f b -> b) -> (a -> f a) -> a -> b
    hylo g h = cata g . ana h
    <=>
    hylo f g = f . fmap (hylo f g) . g
    """
    # we implement the second, fused version, which is more space-efficient
    def hylo(alg, coalg):
        return lambda a: alg(fmap(hylo(alg, coalg), coalg(a)))
    return hylo


def apo_f(fmap, fix=lambda x: x):
    """
    shortcutting anamorphism

    apo :: Fixpoint f t => (a -> f (Either a t)) -> a -> t
    apo coa = inF . fmap (apo coa ||| id) . coa
    """
    identity = lambda x: x

    def apo(coa):
        return lambda a: fix(fmap(fan_in(apo(coa), identity), coa(a)))
    return apo


def compose(*fs):
    "compose functions of a single argument. compose(f, g)(x) == f(g(x))"
    def composed(x):
        for f in reversed(fs):
            x = f(x)
        return x
    return composed
