import matplotlib.pyplot as plt
from typing import NamedTuple

BLACK = 'k'
WHITE = 'w'


class Stone(NamedTuple):
    x: int
    y: int
    color: str


def draw_board(stones: list[Stone], board_size: int = 19):
    fig = plt.figure(figsize=(8, 8))
    fig.set_facecolor('#BA8C63')
    ax = fig.add_subplot(111)

    for i in range(board_size):
        ax.plot((i, i), (0, board_size-1), 'k')
        ax.plot((0, board_size-1), (i, i), 'k')
    ax.set_position((0, 0, 1, 1))

    ax.set_axis_off()

    ax.set_xlim(-1, board_size)
    ax.set_ylim(-1, board_size)

    for (x, y, color) in stones:
        assert color in [BLACK, WHITE]
        assert 0 <= x < board_size
        assert 0 <= y < board_size

        ax.plot(x, y, 'o', markersize=30, markeredgecolor=(
            0, 0, 0), markerfacecolor=color, markeredgewidth=2)
    plt.show()


if __name__ == "__main__":
    stones = [
        (10, 10, BLACK),
        (11, 11, WHITE)
    ]

    draw_board(stones)
