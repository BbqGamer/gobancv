import cv2 as cv
import argparse
import numpy as np
from main import detect_go_game
from state.drawing import board_to_numpy
from state.board import similarity, most_similiar_board
from warp import get_warped


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
    '--debug', type=int, default=0,
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
prev_stones = None
prev_warped = None
prev_board_size = None
board_diff_threshold = 0.8

while True:
    ret, frame = cap.read()
    iter += 1

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if iter % args.frequency == 0:  # update board every n frames
        warped = get_warped(frame, args.debug)
        if warped is None:
            continue
        if prev_warped is not None:
            warpedblur = cv.medianBlur(warped, 9)
            prev_warpedblur = cv.medianBlur(prev_warped, 9)
            diff = cv.absdiff(warpedblur, prev_warpedblur)
            diff = np.where(diff > 10, diff, 0)
            diffsum = np.sum(diff)
            cv.imshow('diff', diff)

            if diffsum > warped.shape[0] * warped.shape[1] * 3 * 10:
                prev_warped = warped.copy()
                print("Skipping frame because of too much difference")
                continue

        prev_warped = warped.copy()
        res = detect_go_game(warped, args.debug)
        if res is not None:
            new_stones, board_size = res
            if prev_board_size is not None and prev_board_size != board_size:
                continue
            prev_board_size = board_size
            if prev_stones is not None:
                new_stones = most_similiar_board(
                    new_stones, prev_stones, board_size, threshold=board_diff_threshold)
            if new_stones is None:
                board_diff_threshold -= 0.1
                continue
            else:
                board_diff_threshold = 0.8
                prev_stones = new_stones
            board = cv.resize(
                board_to_numpy(prev_stones, board_size),
                (height, height),
                interpolation=cv.INTER_AREA
            )
            new_board = cv.cvtColor(board, cv.COLOR_RGB2BGR)

    prev_frame = frame.copy()
    new_frame = np.concatenate((frame, board), axis=1)
    cv.imshow('frame', new_frame)
    if out:
        out.write(frame)

    if cv.waitKey(delay) == ord('q'):
        break

cap.release()
if out:
    out.release()
cv.destroyAllWindows()
