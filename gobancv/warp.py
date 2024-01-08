import cv2 as cv
import numpy as np
from lines import draw_lines, cluster_by_directions, line_filter, draw_segments
from points import cluster_intersections, sort_points_clockwise, get_intersections
import itertools


def find_board(img, debug=0):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    OR = line_filter(gray)

    dilated = cv.dilate(OR, np.ones((3, 3), np.uint8))
    lines = cv.HoughLinesP(dilated, 1, np.pi/180, 100,
                           minLineLength=100, maxLineGap=30)
    tmp = np.zeros_like(gray)
    draw_segments(tmp, lines)

    contours, _ = cv.findContours(
        tmp, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    biggest = max(contours, key=lambda x: cv.contourArea(x))

    noisy_board = np.zeros_like(gray)
    cv.drawContours(noisy_board, [biggest], -1, 255, 2)  # type: ignore

    border = np.zeros_like(gray)
    lines = cv.HoughLines(noisy_board, 1, np.pi/180, 200)
    if lines is None:
        return None
    a, b = cluster_by_directions(lines)

    intersections = get_intersections(a, b)
    corner_candidates = list(cluster_intersections(intersections))
    if debug >= 2:
        clustered = np.zeros_like(gray)
        draw_lines(border, lines)
        draw_lines(clustered, a)
        draw_lines(clustered, b, color=50)
        print(f"Found {len(intersections)} intersections")
        print(f"Found {len(corner_candidates)} corners")
        cv.imshow('DEBUG find_board', np.concatenate(
            [tmp, noisy_board, border, clustered], axis=1))

    if len(corner_candidates) != 4:
        if debug:
            print("Could not find exactly 4 corners")
        return None
    else:
        if debug:
            with_corners = img.copy()
            for p in corner_candidates:
                coord = tuple(map(int, p))
                cv.drawMarker(with_corners,
                              coord, 255, cv.MARKER_CROSS, 20, 2)  # type: ignore
            cv.imshow('With corners', with_corners)
        sorted_points = sort_points_clockwise(corner_candidates)

        # check if points make up a square
        ERROR = 100
        lengths = [np.linalg.norm(p1 - p2)
                   for p1, p2 in itertools.pairwise(sorted_points)]
        if not all(abs(l - lengths[0]) < ERROR for l in lengths):
            if debug:
                print("Points do not make up a square")
            return None
        return np.array(sorted_points)


def get_warped(img, debug=0):
    board = find_board(img, debug)
    if board is None:
        return None

    w = min(img.shape[0], img.shape[1])

    src_points = board.astype(np.float32)
    dest_points = np.array([[0, 0], [w, 0], [w, w], [0, w]], dtype=np.float32)
    M = cv.getPerspectiveTransform(src_points, dest_points)

    warped = cv.warpPerspective(img, M, (w, w))
    if debug:
        cv.imshow('warped', warped)
    return warped
