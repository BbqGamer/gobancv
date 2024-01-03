import csv
import os
import cv2
import argparse


def get_four_points(image_path):
    img = cv2.imread(image_path)
    cv2.imshow("Image", img)

    print(f"Select four points clockwise starting from the top-left corner.")
    
    points = []

    def select_point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"({x}, {y})")
            points.append((x, y))
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("Image", img)

    cv2.setMouseCallback("Image", select_point)
    
    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):  # Press Esc to exit
            cv2.destroyAllWindows()
            return None
        # press enter to confirm
        if key == 13:
            cv2.destroyAllWindows()
            return points

def save_labels(labels, output_file):
    with open(output_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"])
        for image_path, points in labels:
            writer.writerow([image_path] + [p for point in points for p in point])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_folder", required=True,
        help="Path to the folder containing images")

    parser.add_argument(
        "--output_file", required=False, default="border_labels.csv",
        help="Path to the output file")

    args = parser.parse_args()

    image_files = [
        f for f in os.listdir(args.images_folder)
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ]

    labels = []
    for image_file in image_files:
        image_path = os.path.join(args.images_folder, image_file)
        points = get_four_points(image_path)
        labels.append((image_path, points))

    save_labels(labels, args.output_file)

if __name__ == "__main__":
    main()

