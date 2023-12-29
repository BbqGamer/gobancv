import cv2 as cv
import argparse
from warp import get_warped
import numpy as np


parser = argparse.ArgumentParser(
    description='Detect goban state from camera, press q to exit'
)
parser.add_argument(
    '--camera', type=int, default=0, required=False,
    help='camera id (to be passed to cv.VideoCapture)'
)

parser.add_argument(
    '--input-file', type=str, required=False,
    help="Input file with go game (not to be used with --camera)"
)

parser.add_argument(
    '--delay', type=int, required=False,
    help="Control delay between frames (most useful when working with files)"
)

parser.add_argument(
    '--output-file', type=str, required=False,
    help="Choose where to save the processed video"
)


args = parser.parse_args()

if args.input_file and args.camera:
    parser.error('Input cannot be both footage from camera and file')

source = args.camera
delay = 1
if args.input_file:
    source = args.input_file
    delay = 25  # miliseconds
if args.delay:
    delay = args.delay

cap = cv.VideoCapture(source)
fps = int(cap.get(cv.CAP_PROP_FPS))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

out = None
if args.output_file:
    fourcc = cv.VideoWriter.fourcc(*'DIVX')
    out = cv.VideoWriter(args.output_file, fourcc, fps, (width, height))

if not cap.isOpened():
    print("Cannot open camera")
    exit(1)
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    warped = get_warped(frame)
    if warped is None:
        h = min(frame.shape[:2])
        warped = np.zeros((h, h, 3), dtype=np.uint8)

    new_frame = np.concatenate((frame, warped), axis=1)

    if out:
        out.write(new_frame)
    cv.imshow('frame', new_frame)

    if cv.waitKey(delay) == ord('q'):
        break

cap.release()
if out:
    out.release()
cv.destroyAllWindows()
