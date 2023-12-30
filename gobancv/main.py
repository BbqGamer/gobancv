from utils import Stone
from typing import Optional
from warp import get_warped
import cv2 as cv
import numpy as np
from grid import (
    get_intersections,
    get_lines,
    filter_lines,
    draw_lines,
    get_mean_dist,
    draw_intersections
)
from stones import find_circles, draw_circles, closest_intersection
from utils import Stone
from sklearn.cluster import KMeans

def detect_go_game(img, debug=False) -> Optional[list[Stone]]:
    warped = get_warped(img)
    if warped is None:
        return None # no board found

    lines = get_lines(warped)
    h, v = filter_lines(lines, warped.shape[1])

    board_size = len(h)
    BOARD_SIZES = [9, 13, 19]
    if len(h) != len(v) or board_size not in BOARD_SIZES:
        return None # invalid go board

    intersections = get_intersections(h, v)

    # get params for neighborhood
    h_mean = get_mean_dist(h)
    v_mean = get_mean_dist(v)
    radius = min(h_mean, v_mean) // 2
    
    minRadius = int(radius * 3 / 4)
    maxRadius = int(radius * 4 / 3)
    circles = find_circles(warped, minRadius, maxRadius, debug)

    if debug:
        debug_img = warped.copy()
        draw_lines(debug_img, h)
        draw_lines(debug_img, v)
        draw_intersections(debug_img, intersections)
        draw_circles(debug_img, circles)
        cv.imshow('DEBUG main', debug_img)

    # find closest intersection
    stones = []
    intersections = sorted(intersections) 
    for circle in circles[0]:
        cy, cx = closest_intersection(circle, intersections, board_size)
        stones.append(Stone(cy, cx, 'k'))

    return stones

