import cv2 as cv
import argparse


parser = argparse.ArgumentParser(
    description='Detect goban state from camera, press q to exit'
)
parser.add_argument(
    '--camera', type=int, default=0, required=False,
    help='camera id (to be passed to cv.VideoCapture)'
)

parser.add_argument(
    '--filename', type=str, required=False,
    help="Input file with go game (not to be used with --camera)"
)

parser.add_argument(
    '--delay', type=int, required=False,
    help="Control delay between frames (most useful when working with files)"
)


args = parser.parse_args()

if args.filename and args.camera:
    parser.error('Input cannot be both footage from camera and file')

source = args.camera
delay = 1
if args.filename:
    source = args.filename
    delay = 25  # miliseconds
if args.delay:
    delay = args.delay

cap = cv.VideoCapture(source)

if not cap.isOpened():
    print("Cannot open camera")
    exit(1)
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv.imshow('frame', frame)
    if cv.waitKey(delay) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
