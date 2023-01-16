from itertools import chain


def take_pairs(xs):
    """yield pairs from an even length iterable: ABCDEF -> AB CD EF"""
    # See https://docs.python.org/3/library/functions.html#zip Tips and tricks
    return zip(it := iter(xs), it, strict=True)


def untake_pairs(pairs):
    "does the reverse of take_pairs. pairs should be an iterable of tuples"
    return chain.from_iterable(pairs)
