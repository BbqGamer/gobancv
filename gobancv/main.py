from state import Stone
from typing import Optional
import cv2 as cv
import numpy as np
from grid import (
    draw_intersections
)
from lines import draw_lines, line_filter, process_lines, extrapolate_lines_to_board_size
from points import get_intersections
from stones import draw_circles, closest_intersection
from sklearn.cluster import KMeans


def detect_go_game(warped, debug) -> Optional[tuple[list[Stone], int]]:
    gray_warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

    OR = line_filter(gray_warped)
    lines = cv.HoughLines(OR, 1, np.pi / 180, 220)
    if lines is None:
        return None

    res = process_lines(lines, OR.shape)
    if res == None:
        return None
    horizontal, vertical = res

    LEGAL_BOARD_SIZES = [9, 13, 19]
    approx_board_size = max(horizontal.shape[0], vertical.shape[0])
    board_size = min(
        LEGAL_BOARD_SIZES, key=lambda x: abs(x - approx_board_size))

    horizontal = extrapolate_lines_to_board_size(
        horizontal, board_size, OR.shape[1])
    vertical = extrapolate_lines_to_board_size(
        vertical, board_size, OR.shape[0])
    intersections = get_intersections(horizontal, vertical, board_size)

    # get params for neighborhood

    expectedRad = gray_warped.shape[0] // (2 * board_size)
    minRad = int(3 * expectedRad / 4)
    maxRad = int(6 * expectedRad / 4)

    blured = cv.medianBlur(gray_warped, 11)
    circles = cv.HoughCircles(
        blured,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=expectedRad,
        param1=100,  # upper threshold for Canny edge detector
        param2=20,
        minRadius=minRad,
        maxRadius=maxRad
    )

    if debug == 'lines':
        debug_img = warped.copy()
        draw_lines(debug_img, horizontal)
        draw_lines(debug_img, vertical)
        draw_intersections(debug_img, intersections)
        draw_circles(debug_img, circles)
        cv.imshow('DEBUG main', debug_img)

    # find closest intersection
    if circles is None:
        return [], board_size
    stones = []
    intersections = sorted(intersections)
    print("SHAPE", len(intersections))
    intersection_positions = [(i // board_size + 1, i % board_size + 1)
                              for i in range(len(intersections))]
    visited = set()
    for circle in circles[0]:
        closest = closest_intersection(circle, intersections, board_size)
        if closest is not None:
            stones.append((circle, closest))
            visited.add(closest)
    # add intersections that didn't have a circle
    no_circles = []
    for intersection, pos in zip(intersections, intersection_positions):
        if pos not in visited:
            no_circles.append((intersection, pos))

    # cluster stone colors
    circle_colors = []
    for (circle, _) in stones:
        x, y, _ = circle
        x = int(x)
        y = int(y)
        mask = np.zeros(warped.shape[:2], dtype=np.uint8)
        cv.circle(mask, (x, y), minRad, 255, thickness=cv.FILLED)
        mean = cv.mean(warped, mask=mask)
        circle_colors.append(mean[:3])

    if len(circle_colors) == 0:
        return [], board_size

    if len(circle_colors) == 1:
        # there has been only one move made
        xpos = int(stones[0][1][0])
        ypos = int(stones[0][1][1])
        return [Stone(ypos, xpos, 'k')], board_size

    kmeans = KMeans(n_clusters=2, random_state=0,
                    n_init='auto').fit(circle_colors)
    board = []
    c1 = np.sum(kmeans.cluster_centers_[0][0])
    c2 = np.sum(kmeans.cluster_centers_[1][0])
    black = 0 if c1 < c2 else 1
    for (_, (cy, cx)), label in zip(stones, kmeans.labels_):
        color = 'k' if label == black else 'w'
        board.append(Stone(cy, cx, color))

    for intersection in no_circles:
        (x, y), (cy, cx) = intersection
        mask = np.zeros(warped.shape[:2], dtype=np.uint8)
        cv.circle(mask, (int(x), int(y)), minRad, 255, thickness=cv.FILLED)
        color = cv.mean(warped, mask=mask)[:3]
        distances_to_centers = [np.linalg.norm(color - c)
                                for c in kmeans.cluster_centers_]
        closest = np.argmin(distances_to_centers)
        if distances_to_centers[closest] < 30:
            color = 'k' if closest == black else 'w'
            board.append(Stone(cy, cx, color))

    return board, board_size
