import cv2 as cv
import numpy as np
from itertools import pairwise

def get_lines(warped):
    gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    thresholded = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 10
    )
    eroded = cv.erode(thresholded, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv.Canny(eroded, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 150)
    return lines

def draw_lines(img, lines):
    if lines is None:
        return img
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


def filter_lines(lines, width):
    # get horizontal and vertical lines
    horizontal = []
    vertical = []
    for line in lines:
        rho, theta = line[0]
        # ignore lines that are too close to the edge
        if rho < 10 or rho > width - 10:
            continue
        if abs(theta) < np.pi / 8 or abs(theta) > 7 * np.pi / 8:
            vertical.append(line)
        elif abs(theta - np.pi / 2) < np.pi / 8 or \
             abs(theta - np.pi / 2) > 7 * np.pi / 8:
            horizontal.append(line)
    return dedup(horizontal), dedup(vertical)

def dedup(lines):
    # remove lines that are too close to each other
    lines = sorted(lines, key=lambda x: x[0][0])
    deduped = []
    for i in range(len(lines)):
        if i == 0:
            deduped.append(lines[i])
        else:
            rho, _ = lines[i][0]
            prev_rho, _ = lines[i - 1][0]
            if abs(rho - prev_rho) > 10:
                deduped.append(lines[i])
            else :
                deduped[-1] = (lines[i] + deduped[-1]) / 2
    return deduped


def draw_intersections(img, intersections):
    for p in intersections:
        cv.circle(img, (int(p[0]), int(p[1])), 5, (0, 255, 0), -1)


BOARD_SIZES = [9, 13, 19]
def get_intersections(h, v):
    if len(h) not in BOARD_SIZES or len(v) not in BOARD_SIZES:
        return None

    intersections = []
    for hline in h:
        for vline in v:
            h_rho, h_theta = hline[0]
            v_rho, v_theta = vline[0]
            A = np.array([
                [np.cos(h_theta), np.sin(h_theta)],
                [np.cos(v_theta), np.sin(v_theta)]
            ])
            b = np.array([[h_rho], [v_rho]])
            x0, y0 = np.linalg.solve(A, b)
            intersections.append((x0, y0))
    return intersections




def get_mean_dist(lines):
    def mean_diff(pairs):
        return np.mean([abs(a - b) for a, b in pairs])
    mean = mean_diff(pairwise([hline[0][0] for hline in lines]))
    return int(mean)


