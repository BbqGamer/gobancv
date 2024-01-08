import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import cv2 as cv
from sklearn.cluster import KMeans, DBSCAN

CV_8U = cv.CV_8U  # type: ignore


def draw_lines(img, lines, color=255):
    """Takes pointer to image and list of lines to draw
       Lines are in (rho, theta) format
       Optionally takes color
    """
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
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))
        cv.line(img, (x1, y1), (x2, y2), color, 2)  # type: ignore


def draw_segments(img, lines, color=255):
    """Takes pointer to image and list of segments to draw
       Lines are in (x1, y1, x2, y2) format
       Optionally takes color
    """
    if lines is None:
        return img
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img, (x1, y1), (x2, y2), color, 2)  # type: ignore


def line_filter(gray):
    """Detect edges in an image using Sobel filters thresholding 
       and morphological operations
    """
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
    """Cluster lines by angle using KMeans and return two lists"""
    # We take abs of sin bacause we want e.g. 1 degree and 179 degree to be close
    thetas = np.abs(np.sin(lines[:, 0, 1])).reshape(-1, 1)
    if len(thetas) < 2:
        return [], []
    clustering = KMeans(n_clusters=2).fit(thetas)
    a = np.array([l for l, l_ in zip(lines, clustering.labels_) if l_ == 0])
    b = np.array([l for l, l_ in zip(lines, clustering.labels_) if l_ == 1])
    return a, b


def filter_horizontal(lines):
    """Filter only lines (rho, theta) with horizontal angle"""
    eps = np.pi/2 * 1/100
    mask = np.logical_or(lines[:, 0, 1] < eps, lines[:, 0, 1] > np.pi - eps)
    return lines[mask]


def filter_vertical(lines):
    """Filter only lines (rho, theta) with vertical angle"""
    eps = np.pi/2 * 1/100
    up = np.logical_and(lines[:, 0, 1] > np.pi/2 - eps,
                        lines[:, 0, 1] < np.pi/2 + eps)
    return lines[up]


def cluster_lines(lines, eps=10):
    """Finds clusters of lines using DBSCAN returns mean of each cluster"""
    rhos = lines[:, 0, 0]
    thetas = lines[:, 0, 1]
    clustering = DBSCAN(eps=eps, min_samples=1).fit(rhos.reshape(-1, 1))
    centers = []
    for cluster in np.unique(clustering.labels_):
        mean_r = np.mean(rhos[clustering.labels_ == cluster])
        mean_t = np.mean(thetas[clustering.labels_ == cluster])
        centers.append([mean_r, mean_t])
    return np.array(centers).reshape(-1, 1, 2)


def closest_line(line, lines, eps=20):
    """Finds line that is closes to line in lines in terms of both rho and theta
       If the line is close enough to some other line in lines, returns line from lines
       otherwise return line given as argument
    """

    distances_to_lines = np.linalg.norm(lines - line, axis=2)
    closest = np.argmin(distances_to_lines)
    if distances_to_lines[closest] > eps:
        return line
    return lines[closest]


def extrapolate_lines(lines, max_rho, min_rho=0):
    """Extrapolates lines to given rho range using median rho difference"""
    s_lines = np.sort(lines, axis=0)
    window_size = 2
    rhos = s_lines[:, :, 0]
    window = sliding_window_view(rhos, (window_size, 1))
    median_rho_diff = np.median(np.diff(window, axis=2), axis=0).item()
    median_theta = np.median(s_lines[:, :, 1], axis=0).item()

    first = s_lines[0, :, :]

    # go left
    current_rho = first[0][0].copy()

    extrapolated = []

    ERROR = 5
    while current_rho > min_rho - ERROR:
        new_line = np.array([[current_rho, median_theta]])
        # find closest line in s_lines
        closest = closest_line(new_line, s_lines)
        extrapolated.append(closest)
        current_rho -= median_rho_diff

    # go right
    current_rho = first[0][0].copy()
    while current_rho < max_rho + ERROR:
        current_rho += median_rho_diff
        new_line = np.array([[current_rho, median_theta]])
        closest = closest_line(new_line, s_lines)
        extrapolated.append(closest)

    return np.array(extrapolated)
