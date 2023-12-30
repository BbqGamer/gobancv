import cv2 as cv
import numpy as np

def get_histograms(img, intersections, radius):
    colors = []
    for (x, y) in intersections:
        x = int(x[0])
        y = int(y[0])
    
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv.rectangle(mask, (x - radius, y - radius), (x + radius, y + radius), 255, thickness=cv.FILLED)
        roi = cv.bitwise_and(img, img, mask=mask)
        mean = cv.mean(roi)
        colors.append(mean[:3])
    return colors


def find_circles(img, minRadius, maxRadius):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray, 5)
    circles = cv.HoughCircles(
        blur,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=200, # upper threshold for Canny edge detector
        param2=15,
        minRadius=minRadius,
        maxRadius=maxRadius
    )
    return circles


def draw_circles(img, circles):
    if circles is None:
        return img
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

