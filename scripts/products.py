#!/usr/bin/env python3
"""Inspect a project's inputs and print the catalog of reconstruction OUTPUTS you can make.
Run inside recon-sfm (has pycolmap/exiftool). Usually via `recon products <project>`.

The idea: given whatever you have (images, a COLMAP model, calibration, ArUco tags, LiDAR,
GPS), this shows which output products are READY, already DONE, or what input they still NEED.
"""
import argparse, glob, os, subprocess, sys

C = {"g": "\033[32m", "y": "\033[33m", "d": "\033[90m", "b": "\033[1m", "c": "\033[36m", "x": "\033[0m"}


def find_images(proj):
    for sub in ("frames", "images"):
        d = os.path.join(proj, sub)
        n = len(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")) +
                glob.glob(os.path.join(d, "*.jpeg")))
        if n:
            return d, n
    # fallback: subdirectory holding the most images (imported datasets, e.g. KITTI image_02/data)
    from collections import Counter
    c = Counter()
    for f in glob.glob(os.path.join(proj, "**", "*"), recursive=True):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            c[os.path.dirname(f)] += 1
    if c:
        d, n = c.most_common(1)[0]
        return d, n
    return None, 0


def sparse_models(proj):
    base = os.path.join(proj, "colmap", "sparse")
    out = []
    if os.path.isdir(base):
        try:
            import pycolmap
            for d in sorted(os.listdir(base)):
                p = os.path.join(base, d)
                if os.path.isdir(p):
                    try:
                        out.append((p, pycolmap.Reconstruction(p).num_reg_images()))
                    except Exception:
                        pass
        except Exception:
            for d in sorted(os.listdir(base)):
                p = os.path.join(base, d)
                if os.path.isdir(p):
                    out.append((p, -1))
    out.sort(key=lambda x: -x[1])
    return out


def has_gps(img_dir, n=3):
    if not img_dir:
        return False
    files = sorted(glob.glob(os.path.join(img_dir, "*")))[:n]
    if not files:
        return False
    try:
        out = subprocess.run(["exiftool", "-n", "-GPSLatitude", *files],
                             capture_output=True, text=True, timeout=30).stdout
        return "GPS Latitude" in out
    except Exception:
        return False


def find_lidar(proj):
    pats = ["lidar/*.ply", "lidar/*.bin", "lidar/*.pcd", "**/velodyne_points/data/*.bin"]
    hits = []
    for p in pats:
        hits += glob.glob(os.path.join(proj, p), recursive=True)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    args = ap.parse_args()
    P = args.project.rstrip("/")
    name = os.path.basename(P)

    import json
    mpath = os.path.join(P, "recon.json")
    mm = json.load(open(mpath)) if os.path.exists(mpath) else {}
    seqs = mm.get("sequences", [])
    if seqs:
        runs = mm.get("runs", [])
        print(f"\n{C['b']}Project:{C['x']} {P}  {C['c']}(multi-sequence — colmap/<seq>, frames/<seq>, exports/<seq>){C['x']}")
        print(f"{C['b']}Sequences:{C['x']}  (each tracked separately)")
        for s in seqs:
            nm = s["name"] if isinstance(s, dict) else s
            sr = [r for r in runs if r.get("seq") == nm]
            last = f", last: {sr[-1]['output']} ({sr[-1]['status']})" if sr else ", no runs yet"
            print(f"  {C['g']}{nm}{C['x']}  {C['d']}— {len(sr)} run(s){last}{C['x']}")
        print(f"\n  {C['d']}pick a sequence + outputs:  recon menu {P}{C['x']}\n")
        return

    img_dir, nimg = find_images(P)
    models = sparse_models(P)
    best = models[0] if models else None
    calib = glob.glob(os.path.join(P, "**/*.json"), recursive=True)
    calib = [c for c in calib if "intrinsic" in c.lower() or "calib" in c.lower()] or \
            glob.glob(os.path.join(os.path.dirname(os.path.dirname(P)), "calib", "intrinsics", "*.json"))
    tagmap = glob.glob(os.path.join(P, "**/tag_map.yaml"), recursive=True)
    lidar = find_lidar(P)
    gps = has_gps(img_dir)
    dense = os.path.exists(os.path.join(P, "colmap", "dense", "fused.ply"))
    rawvid = bool(glob.glob(os.path.join(P, "raw_video", "*")))
    frames_done = bool(glob.glob(os.path.join(P, "frames", "*.jpg")) + glob.glob(os.path.join(P, "frames", "*.png")))
    has = lambda pat: bool(glob.glob(os.path.join(P, pat)))

    GRN, DOT = C['g'] + "✓" + C['x'], C['d'] + "·" + C['x']

    def block(label, items):
        """Print paths one per line; label + ✓ on the first, indented after. items: [(path, note)]."""
        if not items:
            print(f"  {DOT} {label:<8} {C['d']}(none){C['x']}"); return
        for i, (pth, note) in enumerate(items):
            sym = GRN if i == 0 else " "
            lab = label if i == 0 else ""
            tail = f"  {C['d']}({note}){C['x']}" if note else ""
            print(f"  {sym} {lab:<8} {pth}{tail}")

    ldirs = sorted(set(os.path.dirname(x) for x in lidar))
    print(f"\n{C['b']}Project:{C['x']} {P}")
    print(f"{C['b']}Inputs detected:{C['x']}")
    block("frames:", [(img_dir, f"{nimg} images")] if img_dir else [])
    block("sparse:", [(m, f"{ni} imgs") for m, ni in models])
    block("lidar:",  [(d, f"{sum(1 for x in lidar if os.path.dirname(x) == d)} files") for d in ldirs])
    block("calib:",  [(c, "") for c in calib[:3]])
    block("aruco:",  [(t, "") for t in tagmap])
    print(f"  {GRN if gps else DOT} {'gps:':<8} {'yes (geotagged)' if gps else C['d']+'no'+C['x']}")
    mpath = os.path.join(P, "recon.json")
    if os.path.exists(mpath):
        import json
        mm = json.load(open(mpath))
        runs = mm.get("runs", [])
        last = f", last: {runs[-1]['output']} ({runs[-1]['status']})" if runs else ""
        print(f"  {C['c']}manifest{C['x']}: capture={mm.get('capture', {}).get('type', '?')}, "
              f"{len(runs)} run(s){last}")

    # product catalog: (key, title, produces, ready?, done?, note)
    cat = [
        ("frames",  "Extract frames",        "frames from the data video",           rawvid,          frames_done,          "needs a video"),
        ("sparse",  "COLMAP SfM",            "sparse cloud + camera poses",          nimg,            bool(models),         "needs images"),
        ("dense",   "COLMAP dense MVS",      "dense point cloud + mesh",             nimg or bool(models), dense,           "needs images"),
        ("splat",   "Gaussian splat",        "splat .ply (+cleaned)",                bool(models) or nimg, has("exports/splat*.ply"), "needs images"),
        ("nerf",    "NeRF (nerfacto)",       "trained NeRF + poisson mesh",          bool(models) or nimg, has("exports/poisson*.ply"), "needs images"),
        ("ortho",   "Photogrammetry (ODM)",  "orthomosaic + DEM + mesh",             gps,             has("odm/odm_orthophoto/*"), "needs GPS images"),
        ("vggt",    "VGGT feed-forward",     "poses + depth + points (no COLMAP)",   nimg,            has("exports/vggt*"), "needs images"),
        ("localize","ArUco camera poses",    "camera extrinsics in tag frame",       bool(tagmap and calib), has("localization/camera_poses.json"), "needs tag map + calib"),
        ("register","LiDAR registration",    "lidar aligned to reconstruction",      bool(lidar and (models or dense)), False, "needs lidar + a cloud"),
        ("calibrate","Camera intrinsics",    "calibration .json from ChArUco",       nimg,            bool(calib),          "needs ChArUco images"),
    ]
    print(f"\n{C['b']}Outputs you can make:{C['x']}   ({C['g']}done{C['x']} / {C['c']}ready{C['x']} / {C['d']}needs input{C['x']})")
    for key, title, produces, ready, done, note in cat:
        if done:
            tag = f"{C['g']}● done {C['x']}"
        elif ready:
            tag = f"{C['c']}○ ready{C['x']}"
        else:
            tag = f"{C['d']}· {note:<18}{C['x']}"
        cmd = f"recon make {key} {P}" if ready or done else ""
        print(f"  {tag}  {C['b']}{key:<9}{C['x']} {title:<22} {C['d']}→ {produces}{C['x']}")
        if cmd:
            print(f"             {C['d']}{cmd}{C['x']}")
    print()


if __name__ == "__main__":
    main()
