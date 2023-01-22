import dataclasses
from collections.abc import Sequence, Mapping, Set
from dataclasses import dataclass
from itertools import chain, islice
from typing import Any, Optional, Collection, Iterable, Iterator, Union

from pack.util import take_pairs


# ----------------
#  Data Structures
# ----------------


@dataclass(frozen=True, slots=True)
class Sym:
    ns: Optional[str]
    n: str

    def __repr__(self):
        if self.ns:
            return self.ns + '/' + self.n
        return self.n


@dataclass(frozen=True, slots=True)
class Keyword:
    ns: Optional[str]
    n: str

    def __repr__(self):
        if self.ns:
            return f':{self.ns}/{self.n}'
        return f':{self.n}'

    def __call__(self, a_map, default=None):
        try:
            return a_map.get(self, default)
        except AttributeError:
            # clearly not a map
            return None


# should make this a sequence ...
class List:
    @staticmethod
    def from_iter(it):
        if isinstance(it, List):
            return it

        try:
            rev_it = reversed(it)
        except TypeError:
            # not reversible
            rev_it = reversed(tuple(it))
        xs = nil
        for x in rev_it:
            xs = Cons(x, xs)
        return xs


class Nil(List):

    __slots__ = ()

    def __repr__(self):
        return '()'

    def __len__(self):
        return 0

    def __iter__(self):
        yield from ()

    def __reversed__(self):
        return self

    def __add__(self, o):
        if not isinstance(o, List):
            raise TypeError(
                'can only concatenate List (not "%s") to List'
                % (type(o).__name__)
            )
        return o

    if False:
        def __eq__(self, o):
            if isinstance(o, Nil):
                return True
            if o is None:
                return True
            return False

        def __hash__(self):
            return hash(None)


nil = Nil()
# all instances of Nil() are nil
Nil.__new__ = lambda cls: nil


@dataclass(frozen=True, slots=True)
class Cons(List):
    hd: Any
    tl: 'Optional[List]'

    def __post_init__(self):
        if self.tl is not None and self.tl is not nil:
            assert isinstance(self.tl, Cons), type(self.tl)

    def __iter__(self):
        cons = self
        while cons is not None and cons is not nil:
            yield cons.hd
            cons = cons.tl

    def __reversed__(self):
        # FIXME: This should return an iterator, right?
        result = nil
        for x in self:
            result = Cons(x, result)
        return result

    def __repr__(self):
        return '(' + ' '.join(map(repr, self)) + ')'

    def __bool__(self):
        return True

    def __len__(self):
        return sum(map(lambda _: 1, self), 0)

    def __add__(self, o):
        if o is nil or o is None:
            return self
        if not isinstance(o, List):
            raise TypeError(
                'can only concatenate List (not "%s") to List'
                % (type(o).__name__)
            )
        result = o
        for x in reversed(self):
            result = Cons(x, result)
        return result


# itertools recipes
# https://docs.python.org/3/library/itertools.html#itertools-recipes
def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch


@dataclass(frozen=True, slots=True)
class Vec(Sequence):
    """
    A Trie with at most 32 elements in each node
    """
    xs: tuple[Union[Any, 'Vec'], ...]  # '|' instead of 'Union', broke on py3.11
    height: int

    # A classic sign that this should be split into two different
    # subclasses Leaf and non-leaf Node
    def _is_leaf(self):
        return self.height == 0

    def __len__(self):
        # O(log32(n))
        if self._is_leaf():
            return len(self.xs)
        # Since vectors are contiguous, we just need to look at the
        # final subnode
        return (1 << (5 * self.height)) * (len(self.xs) - 1) + len(self.xs[-1])

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            return self._slice(idx)
        if idx < 0:
            return self[len(self) + idx]
        if idx >= len(self):
            raise IndexError('vector index out of range')

        if self._is_leaf():
            return self.xs[idx]

        subvec_idx = idx >> (5 * self.height)

        mask = (1 << (5 * self.height)) - 1

        return self.xs[subvec_idx][mask & idx]

    def _slice(self, s: slice):
        start, stop, stride = s.indices(len(self))
        if stride != 1:
            return Vec.from_iter(islice(self, start, stop, stride))
        if start >= stop:
            return Vec.empty()
        return SubVec(self, start, stop)

    def subvec(self, start, end=None):
        if start < 0 or start > len(self):
            raise IndexError
        return self._slice(slice(start, end))

    def __iter__(self):
        if self._is_leaf():
            return iter(self.xs)
        return chain.from_iterable(self.xs)

    def __reversed__(self):
        if self._is_leaf():
            return reversed(self.xs)
        return chain.from_iterable(map(reversed, self.xs))

    def conj(self, x):
        # Should we just use _add_iter with a a 1-tuple?
        if self._is_leaf():
            if len(self.xs) < 32:
                return Vec(self.xs + (x,), 0)
            return Vec((self, Vec((x,), 0)), 1)

        old_tail = self.xs[-1]
        new_tail = old_tail.conj(x)
        if new_tail.height == old_tail.height:
            return Vec(
                self.xs[:-1] + (new_tail,),
                self.height,
            )
        if len(self.xs) < 32:
            return Vec(
                self.xs + new_tail.xs[1:],
                self.height,
            )

        new_new_tail = x
        for i in range(self.height + 1):
            new_new_tail = Vec((new_new_tail,), i)

        return Vec((self, new_new_tail), self.height + 1)

    def _fill(self, ys: Iterator):
        """
        fill the space in the vector from ys, without overflowing or
        increasing height
        """
        if self._is_leaf():
            first = islice(ys, 32 - len(self.xs))
            return Vec(self.xs + tuple(first), 0), ys

        last_node, remaining = self.xs[-1]._fill(ys)
        more_vecs = (
            Vec.from_iter(batch)
            for batch in islice(
                batched(remaining, (1 << (5 * self.height))),
                (32 - len(self.xs)),
            )
        )
        return Vec(
            self.xs[:-1] + (last_node,) + tuple(more_vecs),
            self.height
        ), remaining

    def _add_iter(self, ys):
        "append the contents of an iterator to this vector"
        filled, remaining = self._fill(iter(ys))
        try:
            n = next(remaining)
            remaining = chain(iter([n]), remaining)
            is_more = True
        except StopIteration:
            is_more = False

        if not is_more:
            return filled
        return Vec(xs=(filled,), height=(self.height + 1))._add_iter(remaining)

    def __add__(self, other):
        if not isinstance(other, Vec):
            return NotImplemented
            raise TypeError(
                'can only concatenate Vec (not "%s") to Vec'
                % (type(other).__name__)
            )
        return self._add_iter(other)

    def __call__(self, idx: int):
        return self[idx]

    def __repr__(self):
        return '[' + ' '.join(map(repr, self)) + ']'

    @staticmethod
    def from_iter(xs: Iterable):
        # Every time the first batch in the iterator has less than
        # 32 items we nest the iterator into another iterator that
        # batches up up to 32 of those.
        def aux(it, level):
            it0 = (Vec(ys, level) for ys in batched(it, 32))

            first = next(it0)
            if len(first.xs) < 32:
                return first
            try:
                second = next(it0)
            except StopIteration:
                return first

            # undo having taken the first item
            it0 = chain(iter([first, second]), it0)

            return aux(it0, level + 1)

        # Due to the construction of aux, only an empty xs will cause
        # StopIteration to be raised.
        try:
            return aux(xs, 0)
        except StopIteration:
            return _EMPTY_VEC

    @staticmethod
    def empty():
        return _EMPTY_VEC


_EMPTY_VEC = Vec((), 0)


class SubVec(Sequence):

    def __init__(self, vec, start, end):
        if start < 0 or start >= end or end > len(vec):
            raise IndexError

        self.vec = vec
        self.start = start
        self.end = end

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            return self._slice(idx)
        if idx < 0:
            return self[len(self) + idx]
        if idx >= len(self):
            raise IndexError('vector index out of range')
        return self.vec[self.start + idx]

    def _slice(self, s: slice):
        start, stop, stride = s.indices(len(self))
        if stride != 1:
            return Vec.from_iter(islice(self, start, stop, stride))
        offset_slice = slice(start + self.start, stop + self.start, stride)
        return self.vec._slice(offset_slice)

    def subvec(self, start, end=None):
        if start < 0 or start > len(self):
            raise IndexError
        return self._slice(slice(start, end))

    def __len__(self):
        return self.end - self.start

    # It would be sensible to think about implementing __iter__

    def __repr__(self):
        return '[' + ' '.join(map(repr, self)) + ']'

    def __eq__(self, other):
        if not isinstance(other, (Vec, SubVec)):
            return NotImplemented
        for x, y in zip(self, other):
            if x != y:
                return False
        return True

    def __radd__(self, other):
        # Maybe there's a more efficient way to do this
        if not isinstance(other, (Vec, SubVec)):
            return NotImplemented
        return Vec.from_iter(chain(
            other, islice(self.vec, self.start, self.end)
        ))

    def __add__(self, other):
        # Maybe there's a more efficient way to do this
        if not isinstance(other, (Vec, SubVec)):
            return NotImplemented
        return Vec.from_iter(chain(
            islice(self.vec, self.start, self.end), other
        ))

    def __call__(self, idx: int):
        return self[idx]


@dataclass(frozen=True, slots=True)
class ArrayMap(Mapping):
    kvs: tuple

    def __len__(self):
        return len(self.kvs) // 2

    def __iter__(self):
        return islice(self.kvs, 0, None, 2)

    def __getitem__(self, key):
        kvs = self.kvs
        for i in range(0, len(kvs), 2):
            k = kvs[i]
            if key is k or key == k:
                return kvs[i + 1]
        raise KeyError(key)

    def __call__(self, key, default=None):
        return self.get(key, default)

    def __eq__(self, o):
        if not isinstance(o, Mapping):
            return False
        if len(o) != len(self):
            return False
        # inefficient nested loop join
        try:
            for k, v in self.items():
                if v != o[k]:
                    return False
            for k in o:
                if self[k] != o[k]:
                    return False
        except KeyError:
            return False
        return True

    # We implement items and values ourselves because the mixins from
    # Mapping would be O(n^2)
    def items(self) -> Set:
        kvs = self.kvs

        class ItemsView(Set):
            def __iter__(self):
                return take_pairs(kvs)

            def __len__(self):
                return len(kvs) // 2

            def __contains__(self, item):
                return any(item == (k, v) for (k, v) in self)
        return ItemsView()

    def values(self) -> Collection:
        kvs = self.kvs

        class ValuesView:
            def __len__(self):
                return len(kvs) // 2

            def __contains__(self, value):
                for v in iter(self):
                    if v is value or v == value:
                        return True
                return False

            def __iter__(self):
                return islice(kvs, 1, None, 2)
        return ValuesView()

    def __repr__(self):
        return '{' + '  '.join(
            f'{k!r} {v!r}' for (k, v) in self.items()
        ) + '}'

    def assoc(self, key, value):
        new_kvs = list(self.kvs)
        for i in range(0, len(new_kvs), 2):
            if new_kvs[i] == key:
                if new_kvs[i + 1] is value or new_kvs[i + 1] == value:
                    return self
                new_kvs[i + 1] = value
                break
        else:
            new_kvs.append(key)
            new_kvs.append(value)
        if len(new_kvs) > 16:
            return Map.from_iter(take_pairs(new_kvs))
        return ArrayMap(tuple(new_kvs))

    def dissoc(self, key):
        # TODO: implement an efficient version when there are multiple
        # keys to dissoc
        if key not in self.keys():
            return self
        return self.from_iter(
            (k, v) for (k, v) in self.items() if k != key
        )

    def __or__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        result = self
        for k, v in other.items():
            result = result.assoc(k, v)
        return result

    @staticmethod
    def empty():
        return _EMPTY_ARRAY_MAP

    @classmethod
    def from_iter(cls, it):
        def aux():
            seen = set()
            seen_add = seen.add
            for k, v in it:
                if k not in seen:
                    seen_add(k)
                    yield k
                    yield v
        return cls(tuple(aux()))


_EMPTY_ARRAY_MAP = ArrayMap(())


@dataclass(frozen=True, slots=True)
class Map(Mapping):
    """
    A HAMT Map. A Map is a 32 element tuple containing either a map entry or a
    another map with further levels of the tree. The index is the first 5
    bits of a 32-bit hash
    """
    xs: tuple

    kindset: int
    "A 32-bit bitset with 0 a map node, 1 for a map entry"

    height: int

    _len: int

    def __post_init__(self):
        assert self._len >= 0
        assert self.height >= 0

    def __len__(self):
        return self._len

    def _hash32(self, k):
        return hash(k) & ((1 << 32) - 1)

    def _idx_for_key(self, k):
        h = self._hash32(k)
        return (h >> (self.height * 5)) & 0b11111

    def _is_leaf(self, idx):
        return bool(self.kindset & (1 << idx))

    def __getitem__(self, k):
        idx = self._idx_for_key(k)
        if self._is_leaf(idx):
            entry = self.xs[idx]
            if entry is None or entry[0] != k:
                raise KeyError(k)
            return entry[1]

        next_map = self.xs[idx]
        if next_map is None:
            raise KeyError(k)
        return next_map[k]

    def __iter__(self):
        for i, x in enumerate(self.xs):
            if self._is_leaf(i):
                yield x[0]
            elif x is not None:
                for k in x:
                    yield k

    def _kindset_setting_subnode(self, idx):
        "copy of kindset with idx now as a subnode"
        # clear bit idx
        return (self.kindset & (((1 << 32) - 1) ^ (1 << idx)))

    def _kindset_setting_leaf(self, idx):
        "copy of kindset with idx now as a leaf slot (map entry)"
        return (self.kindset | (1 << idx))

    def _with_replacement(self, idx, new_value, *, leaf: bool):
        "return a new map with a single item in the xs tuple replaced"
        if leaf:
            (k, v) = new_value  # assertion

        new_xs = tuple(
            new_value if i == idx else x
            for (i, x) in enumerate(self.xs)
        )
        new_kindset = (
            self._kindset_setting_leaf(idx) if leaf
            else self._kindset_setting_subnode(idx)
        )

        len_of_replaced = (
            1 if self._is_leaf(idx)
            else len(node) if (node := self.xs[idx]) is not None
            else 0
        )
        len_of_replacement = (
            1 if leaf
            else len(new_value) if new_value is not None
            else 0
        )
        new_len = (self._len - len_of_replaced + len_of_replacement)

        return Map(new_xs, new_kindset, _len=new_len, height=self.height)

    def assoc(self, k, v):
        idx = self._idx_for_key(k)

        if self._is_leaf(idx):
            entry = self.xs[idx]
            assert entry is not None
            if entry == (k, v):
                return self
            if entry[0] != k:
                if self.height == 0:
                    assert self._hash32(k) == self._hash32(entry[0])
                    raise NotImplementedError('hash collision')
                # conflict, replace entry with self-node
                new_subnode = (
                    dataclasses.replace(
                        Map.empty(), height=(self.height - 1)
                    )
                    .assoc(entry[0], entry[1])
                    .assoc(k, v)
                )
                return self._with_replacement(idx, new_subnode, leaf=False)

            # This will break when we have to bucket things
            assert entry[0] == k
            # Replace a key
            return self._with_replacement(idx, (k, v), leaf=True)
        subnode = self.xs[idx]
        if subnode is None:
            # put (k, v) as a entry in this node
            return self._with_replacement(idx, (k, v), leaf=True)

        new_subnode = subnode.assoc(k, v)
        return self._with_replacement(idx, new_subnode, leaf=False)

    def dissoc(self, key):
        idx = self._idx_for_key(key)

        if self._is_leaf(idx):
            (k, v) = self.xs[idx]
            if k != key:
                if self.height == 0:
                    raise NotImplementedError('hash collision')
                # key is already not there
                return self
            # replace leaf with empty node
            return self._with_replacement(idx, None, leaf=False)
        subnode = self.xs[idx]
        if subnode is None:
            return self
        new_subnode = subnode.dissoc(key)
        assert len(new_subnode) != 0
        if len(new_subnode) == 1:
            new_entry = next(iter(new_subnode.items()))
            return self._with_replacement(idx, new_entry, leaf=True)
        return self._with_replacement(idx, new_subnode, leaf=False)

    def __or__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        result = self
        for k, v in other.items():
            result = result.assoc(k, v)
        return result

    def __repr__(self):
        return '{' + '  '.join(
            f'{k!r} {v!r}' for (k, v) in self.items()
        ) + '}'

    @classmethod
    def empty(cls):
        return _EMPTY_MAP

    @staticmethod
    def from_iter(it):
        m = Map.empty()
        for k, v in it:
            m = m.assoc(k, v)
        return m


_EMPTY_MAP = Map(
    tuple([None] * 32), kindset=0, _len=0, height=6
)
