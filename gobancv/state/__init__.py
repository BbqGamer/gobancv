from typing import NamedTuple

BLACK = 'k'
WHITE = 'w'


class Stone(NamedTuple):
    x: int
    y: int
    color: str
