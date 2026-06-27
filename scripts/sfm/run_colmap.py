#!/usr/bin/env python3
"""Drive a COLMAP/GLOMAP reconstruction with sensible defaults for our two workloads.

Run inside recon-sfm. Two modes:

  # full SfM (optionally with fixed intrinsics from a calibration json)
  recon sfm python scripts/sfm/run_colmap.py auto \
      --project projects/mill --images projects/mill/frames \
      --matcher sequential --mapper glomap \
      --camera-json calib/intrinsics/iphone15.json

  # fixed/known poses -> triangulate structure only (e.g. ArUco-localized cameras)
  recon sfm python scripts/sfm/run_colmap.py triangulate \
      --project projects/mill --images projects/mill/localization \
      --input projects/mill/localization     # dir with cameras.txt + images.txt

CUDA SIFT is used by default (we built COLMAP for sm_120). Outputs land in
<project>/colmap/{database.db,sparse}.
"""
import argparse, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _progress import stage


def run(*a):
    print("+", " ".join(a)); sys.stdout.flush()
    subprocess.run(a, check=True)


def report_models(sparse_dir):
    """List COLMAP sub-models (incremental mapping fragments by area) and flag the largest."""
    try:
        import pycolmap
    except Exception:
        print(f"done -> {sparse_dir} (inspect sub-models with colmap model_analyzer)"); return
    rows = []
    for d in sorted(os.listdir(sparse_dir)):
        p = os.path.join(sparse_dir, d)
        if not os.path.isdir(p):
            continue
        try:
            rec = pycolmap.Reconstruction(p)
            rows.append((d, rec.num_reg_images(), len(rec.points3D)))
        except Exception:
            pass
    if not rows:
        print("WARNING: no models produced — SfM failed"); return
    rows.sort(key=lambda r: -r[1])
    print(f"\n{len(rows)} model(s) produced (mapping fragments by connected area):")
    for d, ni, npts in rows:
        print(f"  sparse/{d}: {ni:>4} images, {npts:>8,} points" + ("   <- largest, use this" if d == rows[0][0] else ""))


def cam_args(camera_json):
    """Return COLMAP feature_extractor args that fix intrinsics from a calib json."""
    c = json.load(open(camera_json))
    params = [c["fx"], c["fy"], c["cx"], c["cy"], *(c.get("dist", [0, 0, 0, 0]) + [0, 0, 0, 0])[:4]]
    return ["--ImageReader.camera_model", "OPENCV",
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_params", ",".join(f"{x:.8f}" for x in params)]


def auto(a):
    col = a.colmap_dir or os.path.join(a.project, "colmap")
    os.makedirs(os.path.join(col, "sparse"), exist_ok=True)
    db = os.path.join(col, "database.db")

    stage(1, 3, "Feature extraction (CUDA SIFT)")
    fe = ["colmap", "feature_extractor", "--database_path", db,
          "--image_path", a.images, "--FeatureExtraction.use_gpu", "1"]
    if a.camera_json:
        fe += cam_args(a.camera_json)
    run(*fe)

    stage(2, 3, f"Feature matching ({a.matcher})")
    matcher = {"exhaustive": "exhaustive_matcher",
               "sequential": "sequential_matcher",
               "vocab": "vocab_tree_matcher"}[a.matcher]
    run("colmap", matcher, "--database_path", db, "--FeatureMatching.use_gpu", "1")

    stage(3, 3, f"Mapping ({a.mapper})")
    if a.mapper == "glomap":
        run("glomap", "mapper", "--database_path", db,
            "--image_path", a.images, "--output_path", os.path.join(col, "sparse"))
    else:
        mp = ["colmap", "mapper", "--database_path", db,
              "--image_path", a.images, "--output_path", os.path.join(col, "sparse")]
        if a.camera_json:  # don't refine known intrinsics
            mp += ["--Mapper.ba_refine_focal_length", "0",
                   "--Mapper.ba_refine_principal_point", "0",
                   "--Mapper.ba_refine_extra_params", "0"]
        run(*mp)
    report_models(os.path.join(col, "sparse"))


def triangulate(a):
    col = os.path.join(a.project, "colmap")
    out = os.path.join(col, "sparse_fixed")
    os.makedirs(out, exist_ok=True)
    db = os.path.join(col, "database.db")
    stage(1, 3, "Feature extraction (CUDA SIFT)")
    run("colmap", "feature_extractor", "--database_path", db,
        "--image_path", a.images, "--FeatureExtraction.use_gpu", "1")
    stage(2, 3, "Feature matching (exhaustive)")
    run("colmap", "exhaustive_matcher", "--database_path", db, "--FeatureMatching.use_gpu", "1")
    # point_triangulator keeps the supplied poses fixed and only builds structure
    stage(3, 3, "Triangulating structure (fixed poses)")
    run("colmap", "point_triangulator", "--database_path", db,
        "--image_path", a.images, "--input_path", a.input, "--output_path", out)
    print(f"done -> {out} (structure triangulated against fixed poses in {a.input})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("auto"); p.set_defaults(fn=auto)
    p.add_argument("--project", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--colmap-dir", default=None, help="output dir for db+sparse (default <project>/colmap)")
    p.add_argument("--matcher", default="exhaustive", choices=["exhaustive", "sequential", "vocab"])
    p.add_argument("--mapper", default="colmap", choices=["colmap", "glomap"])
    p.add_argument("--camera-json", default=None)

    p = sub.add_parser("triangulate"); p.set_defaults(fn=triangulate)
    p.add_argument("--project", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--input", required=True, help="dir with cameras.txt + images.txt (known poses)")

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
