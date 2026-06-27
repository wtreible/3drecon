#!/usr/bin/env python3
"""Dense MVS reconstruction from a COLMAP sparse model: dense point cloud + textured mesh.
Run inside recon-sfm (needs CUDA for patch-match stereo).

    recon sfm python scripts/sfm/colmap_dense.py \
        --project projects/demo_castle --sparse projects/demo_castle/colmap/sparse/0 \
        --images projects/demo_castle/frames

Outputs <project>/colmap/dense/{fused.ply, meshed-poisson.ply}.
Patch-match stereo is GPU-heavy and scales with image count/resolution — expect minutes.
"""
import argparse, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _progress import stage


def run(*a):
    print("+", " ".join(a)); sys.stdout.flush()
    subprocess.run(a, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--sparse", required=True, help="sparse model dir (e.g. colmap/sparse/0)")
    ap.add_argument("--images", required=True)
    ap.add_argument("--max-image-size", type=int, default=2000, help="downscale cap for stereo")
    ap.add_argument("--out", default=None, help="dense output dir (default <project>/colmap/dense)")
    ap.add_argument("--no-mesh", action="store_true")
    args = ap.parse_args()

    dense = args.out or os.path.join(args.project, "colmap", "dense")
    os.makedirs(dense, exist_ok=True)

    stage(1, 4, "Undistorting images into dense workspace")
    run("colmap", "image_undistorter", "--image_path", args.images,
        "--input_path", args.sparse, "--output_path", dense,
        "--output_type", "COLMAP", "--max_image_size", str(args.max_image_size))

    stage(2, 4, "Patch-match stereo (CUDA — the slow step)")
    run("colmap", "patch_match_stereo", "--workspace_path", dense,
        "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true")

    stage(3, 4, "Fusing depth maps -> dense point cloud")
    fused = os.path.join(dense, "fused.ply")
    run("colmap", "stereo_fusion", "--workspace_path", dense,
        "--workspace_format", "COLMAP", "--input_type", "geometric", "--output_path", fused)

    if not args.no_mesh:
        stage(4, 4, "Poisson meshing -> textured surface")
        run("colmap", "poisson_mesher", "--input_path", fused,
            "--output_path", os.path.join(dense, "meshed-poisson.ply"))
    print(f"Done -> {dense}/fused.ply" + ("" if args.no_mesh else f" + meshed-poisson.ply"))


if __name__ == "__main__":
    main()
