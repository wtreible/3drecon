#!/usr/bin/env python3
"""Calibrate camera intrinsics from ChArUco images; emit a reusable JSON and a
COLMAP camera line.

    recon sfm python scripts/calib/charuco_calibrate.py \
        --board calib/boards/charuco_5x7.yaml \
        --images calib/intrinsics/ubiquiti_g4/*.jpg \
        --name ubiquiti_g4 --out calib/intrinsics

Outputs calib/intrinsics/<name>.json (fx,fy,cx,cy,dist,image_size,rms) and
calib/intrinsics/<name>.colmap.txt (a COLMAP OPENCV camera line you can paste into
cameras.txt or feed to run_colmap.py --camera-json).
"""
import argparse, glob, json, os, sys, yaml, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _progress import track


def load_board(path):
    cfg = yaml.safe_load(open(path))
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["dict"]))
    board = cv2.aruco.CharucoBoard(
        (cfg["squares_x"], cfg["squares_y"]),
        cfg["square_len_m"], cfg["marker_len_m"], d)
    return board


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    ap.add_argument("--images", required=True, nargs="+", help="globs or files")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="calib/intrinsics")
    args = ap.parse_args()

    board = load_board(args.board)
    detector = cv2.aruco.CharucoDetector(board)

    files = []
    for g in args.images:
        files += sorted(glob.glob(g)) if any(c in g for c in "*?[") else [g]
    if not files:
        raise SystemExit("no images matched")

    all_obj, all_img, size = [], [], None
    used = 0
    for f in track(files, "detect charuco", total=len(files)):
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]  # (w, h)
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < 8:
            print(f"  skip (only {0 if ch_ids is None else len(ch_ids)} corners): {f}")
            continue
        obj, imgp = board.matchImagePoints(ch_corners, ch_ids)
        if obj is None or len(obj) < 8:
            continue
        all_obj.append(obj); all_img.append(imgp); used += 1

    if used < 5:
        raise SystemExit(f"need >=5 good views, got {used}")

    rms, K, dist, _, _ = cv2.calibrateCamera(all_obj, all_img, size, None, None)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d = dist.ravel()
    k1, k2, p1, p2 = (list(d) + [0, 0, 0, 0])[:4]

    os.makedirs(args.out, exist_ok=True)
    rec = {
        "name": args.name, "model": "OPENCV", "width": int(size[0]), "height": int(size[1]),
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "dist": d.tolist(), "rms_reproj_px": float(rms), "n_views": used,
    }
    json.dump(rec, open(os.path.join(args.out, f"{args.name}.json"), "w"), indent=2)
    # COLMAP OPENCV model: fx fy cx cy k1 k2 p1 p2
    line = f"# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[fx fy cx cy k1 k2 p1 p2]\n" \
           f"1 OPENCV {int(size[0])} {int(size[1])} " \
           f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} {k1:.8f} {k2:.8f} {p1:.8f} {p2:.8f}\n"
    open(os.path.join(args.out, f"{args.name}.colmap.txt"), "w").write(line)
    print(f"RMS reproj = {rms:.3f}px over {used} views -> {args.out}/{args.name}.json")


if __name__ == "__main__":
    main()
