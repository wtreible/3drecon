#!/usr/bin/env python3
"""Extract sharp, well-spaced frames from handheld video for SfM/splatting.

    recon sfm python scripts/sfm/video_to_frames.py \
        --video projects/mill/raw_video/walk1.mp4 \
        --out projects/mill/frames --target-fps 2 --max-frames 600

Strategy: sample candidate frames at --target-fps, score each by variance-of-Laplacian
(sharpness), then within each sampling window keep the sharpest and drop frames below a
relative blur threshold. Good defaults for phone video of a static scene.
"""
import argparse, glob, os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _progress import track, counter


def sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-fps", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=800)
    ap.add_argument("--start", type=float, default=None, help="extract only from this time (s)")
    ap.add_argument("--end", type=float, default=None, help="extract only up to this time (s)")
    ap.add_argument("--blur-rel", type=float, default=0.4,
                    help="drop frames below this fraction of the median sharpness")
    ap.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality (1-100)")
    ap.add_argument("--percent", action="store_true",
                    help="emit progress as bare 0-100 integers (to feed a TUI gauge)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / args.target_fps)))
    os.makedirs(args.out, exist_ok=True)

    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    # optional [start,end) time window (scene-domain split): seek to start, stop at end
    start_f = max(0, int(round(args.start * src_fps))) if args.start else 0
    end_f = int(round(args.end * src_fps)) if args.end else (nframes or 10**9)
    # clear stale frames from a prior run so re-extraction never accumulates leftovers
    # (this process runs as root in-container, so it can remove earlier root-owned frames)
    for old in glob.glob(os.path.join(args.out, "frame_*.jpg")):
        try:
            os.remove(old)
        except OSError:
            pass
    if start_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    window = max(1, end_f - start_f)
    total_frames = window or None
    pbar = None if args.percent else counter(total_frames, "scan video")
    cands, i, last = [], start_f, -1
    while i < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        if (i - start_f) % step == 0:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cands.append((i, sharpness(g), frame))
        i += 1
        if pbar:
            pbar.update(1)
        elif args.percent and total_frames:
            pct = min(99, (i - start_f) * 100 // window)   # reserve 100 for completion
            if pct != last:
                print(pct, flush=True); last = pct
    if pbar:
        pbar.close()
    cap.release()
    if not cands:
        raise SystemExit("no frames sampled")

    med = float(np.median([s for _, s, _ in cands]))
    kept = [(idx, f) for idx, s, f in cands if s >= args.blur_rel * med]
    # cap count: keep the sharpest if over budget
    if len(kept) > args.max_frames:
        order = sorted(cands, key=lambda x: -x[1])[:args.max_frames]
        keep_idx = {idx for idx, _, _ in order}
        kept = [(idx, f) for idx, _, f in cands if idx in keep_idx]
    kept.sort(key=lambda x: x[0])

    writer = kept if args.percent else track(kept, "write frames", total=len(kept))
    for n, (idx, frame) in enumerate(writer):
        cv2.imwrite(os.path.join(args.out, f"frame_{n:05d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
    if args.percent:
        print(100, flush=True)
    print(f"{len(cands)} sampled (every {step} @ src {src_fps:.1f}fps) -> "
          f"{len(kept)} kept (median sharpness {med:.0f}) in {args.out}")


if __name__ == "__main__":
    main()
