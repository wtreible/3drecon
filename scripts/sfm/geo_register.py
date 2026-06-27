#!/usr/bin/env python3
"""Georeference a COLMAP model from image GPS EXIF (drone flights).

    recon sfm python scripts/sfm/geo_register.py \
        --images projects/drone_site1/images \
        --input projects/drone_site1/colmap/sparse/0 \
        --output projects/drone_site1/colmap/sparse_geo

Reads GPS lat/lon/alt with exiftool, writes a COLMAP geo file (NAME lat lon alt) and
runs `colmap model_aligner` with an ENU alignment so the model lands in metric,
gravity-aligned coordinates. Use --ecef for a global frame instead.
"""
import argparse, csv, io, os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _progress import stage


def read_gps(images_dir):
    out = subprocess.run(
        ["exiftool", "-n", "-csv", "-GPSLatitude", "-GPSLongitude", "-GPSAltitude", images_dir],
        check=True, capture_output=True, text=True).stdout
    rows = list(csv.DictReader(io.StringIO(out)))
    geo = {}
    for r in rows:
        try:
            lat, lon = float(r["GPSLatitude"]), float(r["GPSLongitude"])
        except (KeyError, ValueError):
            continue
        alt = float(r.get("GPSAltitude") or 0.0)
        geo[os.path.basename(r["SourceFile"])] = (lat, lon, alt)
    return geo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--input", required=True, help="sparse model dir (sparse/0)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--ecef", action="store_true", help="align to ECEF instead of local ENU")
    ap.add_argument("--max-error", type=float, default=3.0)
    args = ap.parse_args()

    stage(1, 2, "Reading GPS EXIF (exiftool)")
    geo = read_gps(args.images)
    if len(geo) < 3:
        raise SystemExit(f"need GPS on >=3 images, found {len(geo)}")
    os.makedirs(args.output, exist_ok=True)
    geo_file = os.path.join(args.output, "geo.txt")
    with open(geo_file, "w") as f:
        for name, (lat, lon, alt) in sorted(geo.items()):
            f.write(f"{name} {lat:.10f} {lon:.10f} {alt:.4f}\n")
    print(f"wrote {len(geo)} GPS priors -> {geo_file}")

    stage(2, 2, "Aligning model to GPS (colmap model_aligner)")
    subprocess.run([
        "colmap", "model_aligner",
        "--input_path", args.input, "--output_path", args.output,
        "--ref_images_path", geo_file, "--ref_is_gps", "1",
        "--alignment_type", "ecef" if args.ecef else "enu",
        "--robust_alignment", "1", "--robust_alignment_max_error", str(args.max_error),
    ], check=True)
    print(f"georeferenced model -> {args.output}")


if __name__ == "__main__":
    main()
