from sklearn.cluster import DBSCAN
import numpy as np


def get_intersections(h, v):
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
            intersections.append((x0[0], y0[0]))
    return intersections


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
