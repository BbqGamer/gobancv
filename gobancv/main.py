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
    get_mean_dist
)
from stones import get_histograms
from utils import Stone, board_to_numpy
from sklearn.cluster import KMeans

def detect_go_game(img) -> Optional[list[Stone]]:
    warped = get_warped(img)
    if warped is not None:
        lines = get_lines(warped)
        h, v = filter_lines(lines, warped.shape[1])

        draw_lines(warped, h)
        draw_lines(warped, v)
        intersections = get_intersections(h, v)
        if intersections is not None:
            lines = get_lines(warped)
            # get params for neighborhood
            h_mean = get_mean_dist(h)
            v_mean = get_mean_dist(v)
            radius = min(h_mean, v_mean) // 2
            colors = get_histograms(warped, intersections, radius)
            kmeans = KMeans(n_clusters=3, random_state=0)
            kmeans.fit(colors)
            labels = kmeans.labels_
            light = np.argmax(kmeans.cluster_centers_, axis=0)
            dark = np.argmin(kmeans.cluster_centers_, axis=0)
            print(light, dark)

            stones = []
            for i in range(len(intersections)):
                b = i // 19 + 1
                a = i % 19 + 1
                if labels[i] == dark[0]:
                    stones.append(Stone(a, b, 'k'))
                elif labels[i] == light[0]:
                    stones.append(Stone(a, b, 'w'))
            
            board = board_to_numpy(stones, 19)

            board = cv.resize(
                board,
                warped.shape[:2],
                interpolation=cv.INTER_AREA
            )
            return board
    return None

