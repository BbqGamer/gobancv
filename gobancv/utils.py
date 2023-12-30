import matplotlib.pyplot as plt
from typing import NamedTuple
import numpy as np

BLACK = 'k'
WHITE = 'w'


class Stone(NamedTuple):
    x: int
    y: int
    color: str


def draw_board(stones: list[Stone], board_size: int = 19, show=True):
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
        assert 1 <= x <= board_size
        assert 1 <= y <= board_size

        ax.plot(x-1, y-1, 'o', markersize=30, markeredgecolor=(
            0, 0, 0), markerfacecolor=color, markeredgewidth=2)

    ax.invert_yaxis()
    if show:
        plt.show()
    else:
        return fig

def board_to_numpy(stones: list[Stone], board_size: int = 19):
    fig = draw_board(stones, board_size, show=False)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


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


EXAMPLE_BOARD = """2;12
4;7,8,17
6;16

3;6,7,12,15
15;4,17"""


if __name__ == "__main__":
    stones = read_stones(EXAMPLE_BOARD)
    draw_board(stones)
