#!/usr/bin/env python3
"""Generate a printable ChArUco board image + a YAML config the other tools read.

Run inside recon-sfm (has opencv-contrib). Example:
    recon sfm python scripts/calib/make_charuco_board.py \
        --out calib/boards/charuco_5x7 --squares-x 5 --squares-y 7 \
        --square-len-mm 40 --marker-len-mm 30 --dict DICT_5X5_1000

Print at 100% scale, then MEASURE an actual square edge and put the real value in
the generated YAML's `square_len_m` before calibrating — printers rescale.
"""
import argparse, os, yaml, cv2, numpy as np

DICTS = {n: getattr(cv2.aruco, n) for n in dir(cv2.aruco) if n.startswith("DICT_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output path prefix (no extension)")
    ap.add_argument("--squares-x", type=int, default=5)
    ap.add_argument("--squares-y", type=int, default=7)
    ap.add_argument("--square-len-mm", type=float, default=40.0)
    ap.add_argument("--marker-len-mm", type=float, default=30.0)
    ap.add_argument("--dict", default="DICT_5X5_1000", choices=sorted(DICTS))
    ap.add_argument("--px-per-mm", type=float, default=10.0, help="render resolution")
    args = ap.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(DICTS[args.dict])
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_len_mm / 1000.0, args.marker_len_mm / 1000.0, dictionary)

    w = int(args.squares_x * args.square_len_mm * args.px_per_mm)
    h = int(args.squares_y * args.square_len_mm * args.px_per_mm)
    img = board.generateImage((w, h), marginSize=int(args.square_len_mm * args.px_per_mm / 2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    png = args.out + ".png"
    cv2.imwrite(png, img)

    cfg = {
        "type": "charuco",
        "dict": args.dict,
        "squares_x": args.squares_x,
        "squares_y": args.squares_y,
        "square_len_m": args.square_len_mm / 1000.0,   # <-- update to measured value
        "marker_len_m": args.marker_len_mm / 1000.0,
    }
    with open(args.out + ".yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {png} and {args.out}.yaml")


if __name__ == "__main__":
    main()
