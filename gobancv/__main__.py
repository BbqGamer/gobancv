import cv2 as cv
import argparse
import numpy as np
from main import detect_go_game
from utils import board_to_numpy


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
    '--debug', action='store_true', default=False,
    help="Show intermediate steps"
)

parser.add_argument(
    '--frequency', type=int, default=10,
    help="Update board every n frames"
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

while True:
    ret, frame = cap.read()
    iter += 1

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    if iter % args.frequency == 0: # update board every n frames
        res = detect_go_game(frame, debug=args.debug)
        if res is not None:
            stones, board_size = res
            board = cv.resize(
                board_to_numpy(stones, board_size),
                (height, height),
                interpolation=cv.INTER_AREA
            )
    
    new_frame = np.concatenate((frame, board), axis=1)
    cv.imshow('frame', new_frame)

    if cv.waitKey(delay) == ord('q'):
        break

cap.release()
if out:
    out.release()
cv.destroyAllWindows()
