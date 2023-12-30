import cv2 as cv
import argparse
import numpy as np
from main import detect_go_game


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

parser.add_argument(
    '--debug', action='store_true', required=False,
    help="Show intermediate steps"
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

iter = 0
board = np.zeros((height, height, 3), dtype=np.uint8)
warped = None

while True:
    ret, frame = cap.read()
    iter += 1

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    if iter % 10 == 0:
        new_board = detect_go_game(frame)
        if new_board is not None:
            board = new_board
    
    new_frame = np.concatenate((frame, board), axis=1)
    cv.imshow('frame', new_frame)

    if cv.waitKey(delay) == ord('q'):
        break

cap.release()
if out:
    out.release()
cv.destroyAllWindows()
