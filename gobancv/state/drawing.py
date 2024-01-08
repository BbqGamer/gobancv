import matplotlib.pyplot as plt
import numpy as np
from state import BLACK, WHITE, Stone
from state.board import read_stones


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

        if board_size == 19:
            marker_size = 30
        elif board_size == 13:
            marker_size = 40
        else:
            marker_size = 50

        ax.plot(x-1, y-1, 'o', markersize=marker_size, markeredgecolor=(
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


EXAMPLE_BOARD = """2;12
4;7,8,17
6;16

3;6,7,12,15
15;4,17"""


if __name__ == "__main__":
    stones = read_stones(EXAMPLE_BOARD)
    draw_board(stones)
