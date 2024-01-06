import cv2 as cv
import numpy as np

CV_8U = cv.CV_8U  # type: ignore


def find_roi(img, debug):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(gray, CV_8U, 1, 0).astype('float32')
    sobely = cv.Sobel(gray, CV_8U, 0, 1).astype('float32')
    mag_sob, _ = cv.cartToPolar(sobelx, sobely)
    normalized = cv.normalize(
        mag_sob, None, 0, 255, cv.NORM_MINMAX, CV_8U)  # type: ignore
    th = cv.threshold(normalized, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    dilated = cv.dilate(th, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv.findContours(
        dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=cv.contourArea)
    hull = cv.convexHull(biggest)
    bbox = cv.boundingRect(hull)
    mask = np.zeros_like(gray)
    cv.rectangle(mask, bbox, 255, -1)  # type: ignore
    if debug:
        row1 = np.concatenate([sobelx, sobely, normalized], axis=1)
        row2 = np.concatenate([th, dilated, mask], axis=1)
        cv.imshow('debug', np.concatenate([row1, row2], axis=0))
        cv.waitKey(0)
        cv.destroyAllWindows()
    return mask
