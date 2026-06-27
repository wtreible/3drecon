#!/usr/bin/env python3
"""VGGT feed-forward reconstruction: images -> camera poses + dense point cloud, no COLMAP.
Run inside recon-dl. Good for a fast first look or hard low-overlap scenes.

    recon dl python scripts/dl/vggt_infer.py \
        --images projects/demo_castle/frames \
        --out projects/demo_castle/exports/vggt --max-images 16

Outputs <out>_points.ply (confidence-filtered) and <out>_cameras.json.
VGGT memory scales with image count — keep --max-images modest (it samples evenly).
Weights (facebook/VGGT-1B) download once to /work/data/hf_cache.
"""
import argparse, glob, json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _progress import stage
import open3d as o3d
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True, help="output path prefix")
    ap.add_argument("--max-images", type=int, default=16)
    ap.add_argument("--conf-percentile", type=float, default=50.0,
                    help="drop points below this confidence percentile")
    args = ap.parse_args()

    files = sorted(sum((glob.glob(os.path.join(args.images, e))
                        for e in ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG")), []))
    if not files:
        raise SystemExit(f"no images in {args.images}")
    if len(files) > args.max_images:                       # sample evenly
        idx = np.linspace(0, len(files) - 1, args.max_images).round().astype(int)
        files = [files[i] for i in idx]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    dev = "cuda"
    stage(1, 3, f"Loading VGGT-1B + {len(files)} images")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
    images = load_and_preprocess_images(files).to(dev)     # [S,3,H,W]

    stage(2, 3, "Feed-forward inference")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pred = model(images)
    print("    prediction keys:", list(pred.keys()))

    def first(t):
        t = t[0] if t.dim() >= 4 and t.shape[0] == 1 else t
        return t.float().cpu().numpy()
    wp = first(pred["world_points"])                       # [S,H,W,3]
    conf = first(pred["world_points_conf"])                # [S,H,W]
    cols = first(pred["images"]).transpose(0, 2, 3, 1)     # [S,H,W,3] in [0,1]

    stage(3, 3, "Filtering by confidence + writing outputs")
    thr = np.percentile(conf, args.conf_percentile)
    m = conf >= thr
    pts = wp[m].reshape(-1, 3)
    rgb = np.clip(cols[m].reshape(-1, 3), 0, 1)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(args.out + "_points.ply", pc)

    # camera poses, if decodable
    try:
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
        json.dump({"extrinsics": first(extri).tolist(), "intrinsics": first(intri).tolist()},
                  open(args.out + "_cameras.json", "w"))
        cams = " + _cameras.json"
    except Exception as e:
        cams = f" (poses skipped: {e})"
    print(f"{len(pts):,} points (conf>=p{args.conf_percentile:g}) -> {args.out}_points.ply{cams}")


if __name__ == "__main__":
    main()
