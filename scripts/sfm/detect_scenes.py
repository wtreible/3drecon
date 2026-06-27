#!/usr/bin/env python3
"""Detect scene boundaries (fades-to-black and hard cuts) in a video and optionally split
it into per-scene clips. Run in recon-sfm (cv2 + ffmpeg, no extra deps).

Useful when one capture contains several segments separated by fades/cuts — e.g. a
walk-through with fades between rooms. Each detected scene becomes its own clip, which can
be ingested as a separate sequence.

    recon sfm python scripts/sfm/detect_scenes.py --video projects/x/raw_video/hall.mp4
    recon sfm python scripts/sfm/detect_scenes.py --video ... --split --out projects/x/raw_video
"""
import argparse, json, os, subprocess, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _progress import counter


def detect(video, sample_fps, black_thresh, content_thresh, min_scene):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(src_fps / sample_fps)))

    times, bright, hists = [], [], []
    pbar = counter(total or None, "analyze")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            times.append(i / src_fps)
            bright.append(float(g.mean()))
            h = cv2.calcHist([g], [0], None, [64], [0, 256])
            hists.append(cv2.normalize(h, h).flatten())
        i += 1
        pbar.update(1)
    pbar.close(); cap.release()
    dur = i / src_fps if i else 0.0
    n = len(times)
    if n < 2:
        return dur, [[0.0, round(dur, 2)]]

    black = [b < black_thresh for b in bright]
    cut = [False] * n
    for k in range(1, n):
        d = 1.0 - float(cv2.compareHist(hists[k - 1], hists[k], cv2.HISTCMP_CORREL))
        if d > content_thresh and not black[k] and not black[k - 1]:
            cut[k] = True

    # content runs separated by black (fade) runs or hard cuts
    segs, s = [], None
    for k in range(n):
        if black[k]:
            if s is not None:
                segs.append((s, k - 1)); s = None
        else:
            if s is None:
                s = k
            elif cut[k]:
                segs.append((s, k - 1)); s = k
    if s is not None:
        segs.append((s, n - 1))

    scenes = []
    for a, b in segs:
        t0 = times[a]
        t1 = min(times[b] + 1.0 / sample_fps, dur)
        if t1 - t0 >= min_scene:
            scenes.append([round(t0, 2), round(t1, 2)])
    return dur, (scenes or [[0.0, round(dur, 2)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None, help="dir for split clips (default: alongside the video)")
    ap.add_argument("--split", action="store_true", help="write a clip per scene")
    ap.add_argument("--sample-fps", type=float, default=5.0, help="analysis rate")
    ap.add_argument("--black-thresh", type=float, default=28.0, help="mean luma (0-255) below = fade/black")
    ap.add_argument("--content-thresh", type=float, default=0.45, help="histogram distance for a hard cut (0-1)")
    ap.add_argument("--min-scene", type=float, default=2.0, help="drop scenes shorter than this (s)")
    ap.add_argument("--json", action="store_true", help="emit JSON (scenes / clips)")
    args = ap.parse_args()

    dur, scenes = detect(args.video, args.sample_fps, args.black_thresh, args.content_thresh, args.min_scene)
    if not args.json:
        print(f"{len(scenes)} scene(s) in {os.path.basename(args.video)} ({dur:.1f}s):")
        for j, (a, b) in enumerate(scenes, 1):
            print(f"  scene {j}: {a:6.1f}–{b:6.1f}s  ({b - a:.1f}s)")

    clips = []
    if args.split:
        out = args.out or os.path.dirname(args.video) or "."
        os.makedirs(out, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.video))[0]
        for j, (a, b) in enumerate(scenes, 1):
            dst = os.path.join(out, f"{stem}_scene{j:02d}.mp4")
            subprocess.run(["ffmpeg", "-y", "-ss", str(a), "-i", args.video, "-t", str(round(b - a, 2)),
                            "-c:v", "libx264", "-crf", "18", "-an", dst],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            clips.append(dst)
        if not args.json:
            print(f"split -> {len(clips)} clips in {out}")

    if args.json:
        print(json.dumps({"video": args.video, "duration": round(dur, 2),
                          "scenes": scenes, "clips": clips}))


if __name__ == "__main__":
    main()
