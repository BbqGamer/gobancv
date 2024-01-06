from state import Stone
from typing import Optional
from warp import get_warped
import cv2 as cv
import numpy as np
from grid import (
    get_lines,
    filter_lines,
    get_mean_dist,
    draw_intersections
)
from lines import draw_lines_polar
from points import get_intersections
from stones import find_circles, draw_circles, closest_intersection
from sklearn.cluster import KMeans


def detect_go_game(img, debug=0) -> Optional[tuple[list[Stone], int]]:
    warped = get_warped(img, debug)
    if warped is None:
        return None  # no board found

    lines = get_lines(warped)
    if lines is None:
        return None
    h, v = filter_lines(lines, warped.shape[1])

    board_size = len(h)
    BOARD_SIZES = [9, 13, 19]
    if len(h) != len(v) or board_size not in BOARD_SIZES:
        return None  # invalid go board

    intersections = get_intersections(h, v)

    # get params for neighborhood
    h_mean = get_mean_dist(h)
    v_mean = get_mean_dist(v)
    radius = min(h_mean, v_mean) // 2

    minRadius = int(radius * 3 / 4)
    maxRadius = int(radius * 4 / 3)
    circles = find_circles(warped, minRadius, maxRadius)

    if debug:
        debug_img = warped.copy()
        draw_lines_polar(debug_img, h)
        draw_lines_polar(debug_img, v)
        draw_intersections(debug_img, intersections)
        draw_circles(debug_img, circles)
        cv.imshow('DEBUG main', debug_img)

    # find closest intersection
    if circles is None:
        return [], board_size
    stones = []
    intersections = sorted(intersections)
    for circle in circles[0]:
        closest = closest_intersection(circle, intersections, board_size)
        if closest is not None:
            stones.append((circle, closest))

    # cluster stone colors
    colors = []
    for (circle, _) in stones:
        x, y, _ = circle
        x = int(x)
        y = int(y)
        roi = warped[y - radius:y + radius, x - radius:x + radius]
        mean = cv.mean(roi)
        colors.append(mean[:3])

    kmeans = KMeans(n_clusters=2, random_state=0).fit(colors)
    board = []
    c1 = np.sum(kmeans.cluster_centers_[0][0])
    c2 = np.sum(kmeans.cluster_centers_[1][0])
    black = 0 if c1 < c2 else 1
    for (_, (cy, cx)), label in zip(stones, kmeans.labels_):
        color = 'k' if label == black else 'w'
        board.append(Stone(cy, cx, color))
    return board, board_size
