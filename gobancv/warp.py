import cv2 as cv
import numpy as np

def find_board(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 50, 150)
    contours, _ = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv.contourArea, reverse=True):
        # check if contour is large enough (board should be 0.3+ of the image)
        if cv.contourArea(contour) < img.shape[0] * img.shape[1] * 0.2:
            return None

        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            return approx
    return None


def get_warped(img):
    board = find_board(img)
    if board is None:
        return None

    w = min(img.shape[0], img.shape[1])
    
    src_points = board.astype(np.float32)
    dest_points = np.array([[0, 0], [0, w], [w, w], [w, 0]], dtype=np.float32)
    M = cv.getPerspectiveTransform(src_points, dest_points)
    
    warped = cv.warpPerspective(img, M, (w, w))
    return warped

