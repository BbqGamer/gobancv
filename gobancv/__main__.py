import cv2 as cv
import argparse
from warp import get_warped
from grid import get_intersections, get_lines, filter_lines, draw_lines, get_mean_dist
from stones import get_histograms
from utils import Stone, board_to_numpy
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
board = None
warped = None
while True:
    ret, frame = cap.read()
    iter += 1

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    if iter % 10 == 0:
        warped = get_warped(frame)
        if warped is not None:
            lines = get_lines(warped)
            h, v = filter_lines(lines, warped.shape[1])

            draw_lines(warped, h)
            draw_lines(warped, v)
            intersections = get_intersections(h, v)
            if intersections is not None:
                lines = get_lines(warped)
                # get params for neighborhood
                h_mean = get_mean_dist(h)
                v_mean = get_mean_dist(v)
                radius = min(h_mean, v_mean) // 2
                colors = get_histograms(warped, intersections, radius)

                stones = []
                for i, (x, y) in enumerate(sorted(intersections)):
                    # cv.circle(warped, (int(x), int(y)), radius, colors[i], -1)
                    b = i // 19 + 1
                    a = i % 19 + 1
                    if colors[i] == (0, 0, 0):
                        stones.append(Stone(a, b, 'k'))
                    elif colors[i] == (255, 255, 255):
                        stones.append(Stone(a, b, 'w'))
                
                board = board_to_numpy(stones, 19)
                board = cv.resize(board, warped.shape[:2], interpolation=cv.INTER_AREA)



    if warped is None:
        h = min(frame.shape[:2])
        warped = np.zeros((h, h, 3), dtype=np.uint8)

    if board is None:
        board = np.zeros_like(warped)
    
    if args.debug:
        new_frame = np.concatenate((frame, warped, board), axis=1)
    else:
        new_frame = frame

    if out:
        out.write(new_frame)
    cv.imshow('frame', new_frame)

    if cv.waitKey(delay) == ord('q'):
        break

cap.release()
if out:
    out.release()
cv.destroyAllWindows()
