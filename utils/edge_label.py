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
            if len(points) != 4:
                print("Please select four points.")
            else:
                cv2.destroyAllWindows()
                return points

def save_labels(image_path, points, output_file):
    if points is None:
        return

    image_name = os.path.basename(image_path)
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            [image_name] + [str(coord) for point in points for coord in point]
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_folder", required=True,
        help="Path to the folder containing images")

    parser.add_argument(
        "--output_file", required=False, default="border_labels.csv",
        help="Path to the output file")

    args = parser.parse_args()

    if not os.path.exists(args.output_file):
        with open(args.output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                ["Image_Name", "X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"]
            )

    image_files = [
        f for f in os.listdir(args.images_folder)
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ]

    for image_file in image_files:
        image_path = os.path.join(args.images_folder, image_file)
        points = get_four_points(image_path)

        if points is not None:
            save_labels(image_path, points, args.output_file)

if __name__ == "__main__":
    main()

