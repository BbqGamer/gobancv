from state import Stone, BLACK, WHITE
from typing import NewType


def similarity(b_1: list[Stone], b_2: list[Stone]):
    """Compare jaccard similarity of two boards"""
    s_1 = set(b_1)
    s_2 = set(b_2)
    numerator = len(s_1 & s_2)
    denominator = len(s_1 | s_2)
    if denominator == 0:
        return 0
    return numerator / denominator 


def read_stones(content: str) -> list[Stone]:
    """Read stones from a string in the format:
            2;12
            4;7,8,17
            6;16

            3;6,7,12,15
            15;4,17
        Where first chunk are black stones and the second chunk are white stones.
        first number in line indicates y coordinate and the rest are x coordinates.
        min coordinate here is 1 and max 19 (not starting from 0)
    """
    stones = []
    chunks = content.split('\n\n')
    assert len(chunks) == 2

    for chunk, color in zip(chunks, [BLACK, WHITE]):
        for line in chunk.splitlines():
            y, xs = line.split(';')
            for x in xs.split(','):
                stones.append(Stone(int(x), int(y), color))

    return stones
