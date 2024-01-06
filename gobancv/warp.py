import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from grid import get_intersections


def draw_lines_polar(img, lines, color=255):
    h = img.shape[0]
    w = img.shape[1]
    if lines is None:
        return img
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + w * (-b))
        y1 = int(y0 + h * (a))
        x2 = int(x0 - w * (-b))
        y2 = int(y0 - h * (a))
        cv.line(img, (x1, y1), (x2, y2), color, 2)


def draw_lines(img, lines):
    if lines is None:
        return img
    imgc = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(imgc, (x1, y1), (x2, y2), 255, 2)
    return imgc


def line_filter(gray):
    def line_filter_aux(gray, kernel):
        down = cv.filter2D(gray, cv.CV_8U, kernel)
        up = cv.filter2D(gray, cv.CV_8U, np.flip(kernel))
        thup = cv.threshold(up, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        thdown = cv.threshold(
            down, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        dup = cv.dilate(thup, np.ones((3, 3), np.uint8))
        ddown = cv.dilate(thdown, np.ones((3, 3), np.uint8))
        AND = cv.bitwise_and(dup, ddown)
        return AND

    sobel_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    a = line_filter_aux(gray, sobel_kernel.T)
    b = line_filter_aux(gray, sobel_kernel)
    OR = cv.bitwise_or(a, b)
    return OR


def cluster_by_directions(lines):
    """Cluster lines by direction using KMeans and return two lists"""
    # We take abs of sin bacause we want e.g. 1 degree and 179 degree to be close
    thetas = np.abs(np.sin(lines[:, 0, 1])).reshape(-1, 1)
    if len(thetas) < 2:
        return [], []
    clustering = KMeans(n_clusters=2).fit(thetas)
    a = np.array([l for l, l_ in zip(lines, clustering.labels_) if l_ == 0])
    b = np.array([l for l, l_ in zip(lines, clustering.labels_) if l_ == 1])
    return a, b


def inside_image(p, shape):
    return 0 <= p[0] < shape[1] and 0 <= p[1] < shape[0]


def cluster_intersections(intersections):
    """Cluster intersections using DBSCAN and return the cluster centers"""
    if len(intersections) < 2:
        return []
    clustering = DBSCAN(eps=50, min_samples=1).fit(intersections)
    for label in set(clustering.labels_):
        if label == -1:
            continue
        yield np.mean([p for p, l in zip(intersections, clustering.labels_) if l == label], axis=0)


def sort_points_clockwise(points):
    def get_angle(point):
        x, y = point[0] - reference_point[0], point[1] - reference_point[1]
        return np.arctan2(y, x)

    reference_point = np.mean(points, axis=0)
    sorted_points = sorted(points, key=get_angle)

    return sorted_points


def find_board(img, debug=0):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    OR = line_filter(gray)

    dilated = cv.dilate(OR, np.ones((3, 3), np.uint8))
    lines = cv.HoughLinesP(dilated, 1, np.pi/180, 100,
                           minLineLength=100, maxLineGap=30)
    tmp = np.zeros_like(gray)
    tmp = draw_lines(tmp, lines)

    contours, _ = cv.findContours(
        tmp, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=lambda x: cv.contourArea(x))

    noisy_board = np.zeros_like(gray)
    cv.drawContours(noisy_board, [biggest], -1, 255, 2)  # type: ignore

    border = np.zeros_like(gray)
    lines = cv.HoughLines(noisy_board, 1, np.pi/180, 200)

    clustered = np.zeros_like(gray)
    a, b = cluster_by_directions(lines)

    intersections = get_intersections(a, b)
    corner_candidates = list(cluster_intersections(intersections))
    if debug >= 2:
        draw_lines_polar(border, lines)
        draw_lines_polar(clustered, a)
        draw_lines_polar(clustered, b, color=50)
        print(f"Found {len(intersections)} intersections")
        print(f"Found {len(corner_candidates)} corners")
        cv.imshow('DEBUG find_board', np.concatenate(
            [tmp, noisy_board, border, clustered], axis=1))

    if len(corner_candidates) != 4:
        if debug:
            print("Could not find exactly 4 corners")
        return None
    else:
        with_corners = img.copy()
        for p in corner_candidates:
            coord = tuple(map(int, p))
            cv.drawMarker(with_corners, coord, 255, cv.MARKER_CROSS, 20, 2)
        if debug:
            cv.imshow('With corners', with_corners)
        return np.array(sort_points_clockwise(corner_candidates))


def get_warped(img, debug=0):
    board = find_board(img, debug)
    if board is None:
        return None

    w = min(img.shape[0], img.shape[1])

    src_points = board.astype(np.float32)
    dest_points = np.array([[0, 0], [w, 0], [w, w], [0, w]], dtype=np.float32)
    M = cv.getPerspectiveTransform(src_points, dest_points)

    warped = cv.warpPerspective(img, M, (w, w))
    cv.imshow('warped', warped)
    return warped
