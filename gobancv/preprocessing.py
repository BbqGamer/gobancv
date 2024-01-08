import cv2 as cv


def imshow(img):
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


class Pipeline:
    def __init__(self):
        self.steps = []

    def transform(self, img):
        for f in self.steps:
             img = f(img)
        return img

    def imshow(self, img):
        imshow(self.transform(img))
    
    def append(self, f):
        self.steps.append(f)

    def __repr__(self):
        return " -> ".join([f.__name__ for f in self.steps])


def grayscale():
    def to_gray(img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return to_gray


def gaussian_blur(kernel_size):
    def gaussian(img):
        return cv.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return gaussian


def canny(low_threshold, high_threshold):
    def canny_detection(img):
        return cv.Canny(img, low_threshold, high_threshold)
    return canny_detection


def pipeline():
    p = Pipeline()
    p.append(grayscale())
    p.append(gaussian_blur(5))
    p.append(canny(50, 150))
    return p


def tophat(img):
    b, g, r = cv.split(img)

    # Apply a top-hat transform to each channel
    kernel_size = 9  # You may need to adjust this based on your image characteristics
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))

    tophat_b = cv.morphologyEx(b, cv.MORPH_TOPHAT, kernel)
    tophat_g = cv.morphologyEx(g, cv.MORPH_TOPHAT, kernel)
    tophat_r = cv.morphologyEx(r, cv.MORPH_TOPHAT, kernel)

    # Add the top-hat transform results to the original image
    normalized_b = cv.add(b, tophat_b)
    normalized_g = cv.add(g, tophat_g)
    normalized_r = cv.add(r, tophat_r)

    # Merge the channels back into a color image
    normalized_image = cv.merge((normalized_b, normalized_g, normalized_r))
    return normalized_image

