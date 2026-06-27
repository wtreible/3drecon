#!/usr/bin/env python3
"""Headless preview of a 3DGS .ply with gsplat — renders an overview from a synthesized
camera so you can QC a splat (floaters, framing, extent) without a GUI viewer.

    recon dl python scripts/splat/render_ply.py \
        --in projects/demo_lighthouse/exports/splat_clean.ply \
        --out projects/demo_lighthouse/exports/preview.jpg --azimuth 45 --elevation 35

Color uses the SH DC term only (view-independent) — enough to judge geometry/floaters.
"""
import argparse, os, sys, math, numpy as np, torch, gsplat
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clean_ply import read_ply

SH_C0 = 0.28209479177387814


def look_at(eye, target, up=(0, 0, 1)):
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    f = target - eye; f /= np.linalg.norm(f)            # forward (+z, OpenCV)
    r = np.cross(f, up); r /= np.linalg.norm(r)          # right (+x)
    d = np.cross(f, r)                                    # down (+y)
    R = np.stack([r, d, f], 0)                           # world->cam
    t = -R @ eye
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["overview", "interior"], default="overview",
                    help="overview = orbit from outside; interior = stand at center, look outward")
    ap.add_argument("--azimuth", type=float, default=45.0)
    ap.add_argument("--elevation", type=float, default=35.0)
    ap.add_argument("--distance-scale", type=float, default=1.6, help="× scene extent (overview)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=0.0, help="0 = auto (60 overview / 85 interior)")
    args = ap.parse_args()

    d = read_ply(args.inp)
    dev = "cuda"
    means = torch.tensor(np.stack([d['x'], d['y'], d['z']], 1), dtype=torch.float32, device=dev)
    quats = torch.tensor(np.stack([d['rot_0'], d['rot_1'], d['rot_2'], d['rot_3']], 1),
                         dtype=torch.float32, device=dev)
    scales = torch.tensor(np.exp(np.stack([d['scale_0'], d['scale_1'], d['scale_2']], 1)),
                          dtype=torch.float32, device=dev)
    opac = torch.sigmoid(torch.tensor(d['opacity'], dtype=torch.float32, device=dev))
    fdc = np.stack([d['f_dc_0'], d['f_dc_1'], d['f_dc_2']], 1)
    rgb = torch.tensor(np.clip(SH_C0 * fdc + 0.5, 0, 1), dtype=torch.float32, device=dev)

    # synthesize an overview camera looking at the robust centroid
    p = means.cpu().numpy()
    ctr = np.median(p, 0)
    extent = float(np.linalg.norm(np.percentile(p, 97, 0) - np.percentile(p, 3, 0)))
    az, el = math.radians(args.azimuth), math.radians(args.elevation)
    dir_ = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    if args.mode == "interior":
        eye = ctr                      # stand at the room center, look outward
        target = ctr + dir_
        fov = args.fov or 85.0
    else:
        eye = ctr + args.distance_scale * extent * dir_   # orbit from outside, look in
        target = ctr
        fov = args.fov or 60.0
    viewmat = torch.tensor(look_at(eye, target)[None], dtype=torch.float32, device=dev)
    f = 0.5 * args.width / math.tan(math.radians(fov) / 2)
    K = torch.tensor([[[f, 0, args.width / 2], [0, f, args.height / 2], [0, 0, 1]]],
                     dtype=torch.float32, device=dev)

    out, _, _ = gsplat.rasterization(means, quats, scales, opac, rgb, viewmat, K,
                                     args.width, args.height, render_mode="RGB")
    img = (out[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    import cv2
    cv2.imwrite(args.out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"{len(d):,} gaussians | extent ~{extent:.1f} | wrote {args.out}")


if __name__ == "__main__":
    main()
