import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

CV_8U = cv.CV_8U  # type: ignore


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
        cv.line(img, (x1, y1), (x2, y2), color, 2)  # type: ignore


def draw_lines(img, lines, color=255):
    if lines is None:
        return img
    imgc = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(imgc, (x1, y1), (x2, y2), color, 2)  # type: ignore
    return imgc


def line_filter(gray):
    def line_filter_aux(gray, kernel):
        down = cv.filter2D(gray, CV_8U, kernel)
        up = cv.filter2D(gray, CV_8U, np.flip(kernel))
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
