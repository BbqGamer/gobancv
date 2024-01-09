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


def flip_board(stones: list[Stone], board_size: int):
    """Flip board by 90 degrees"""
    new_stones = []
    for stone in stones:
        new_stones.append(
            Stone(y=board_size - stone.x + 1, x=stone.y, color=stone.color))
    return new_stones


def most_similiar_board(stones: list[Stone], target_stones: list[Stone], board_size: int, threshold):
    best_similarity = similarity(stones, target_stones)
    best_stones = stones
    for _ in range(3):
        stones = flip_board(stones, board_size)
        if similarity(stones, target_stones) > best_similarity:
            best_similarity = similarity(stones, target_stones)
            best_stones = stones
    return best_stones if best_similarity > threshold else None


def diff_stones(prev: list[Stone], current: list[Stone]):
    added = [s for s in current if s not in prev]
    removed = [s for s in prev if s not in current]
    return added, removed
