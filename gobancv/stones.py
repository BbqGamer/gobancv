import cv2 as cv
import numpy as np

def get_histograms(img, intersections, radius):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    colors = []
    for (x, y) in intersections:
        x = int(x[0])
        y = int(y[0])
        cv.circle(img, (x, y), radius, (0, 255, 0), 2) 
    
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv.circle(mask, (x, y), radius, 255, thickness=cv.FILLED)
        mean = cv.mean(gray, mask=mask)[0]
        if mean < 80:
            colors.append((0, 0, 0))
        elif mean > 165:
            colors.append((255, 255, 255))
        else:
            colors.append((0, 0, 255))
    return colors

