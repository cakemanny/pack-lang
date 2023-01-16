# Some classes needed by both interp and compiler

from dataclasses import dataclass
from typing import Any

from pack.data import ArrayMap, Map, Sym


@dataclass
class Var:
    "This is the mutable thing"
    symbol: Sym
    value: Any
    metadata: ArrayMap | Map

    def __init__(self, symbol, value=None, metadata=ArrayMap.empty()):
        assert isinstance(symbol, Sym)
        self.symbol = symbol
        self.value = value
        self.metadata = metadata

    def __repr__(self):
        return f"#'{self.symbol}"
