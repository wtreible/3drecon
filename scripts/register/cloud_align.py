#!/usr/bin/env python3
"""Register a source point cloud (e.g. a LiDAR scan) onto a target (e.g. the SfM/splat
cloud) to bring it into the common metric frame.

    recon sfm python scripts/register/cloud_align.py \
        --source projects/mill/lidar/scan01.ply \
        --target projects/mill/colmap/dense/fused.ply \
        --voxel 0.05 --out projects/mill/lidar/scan01_aligned

Pipeline: voxel downsample -> FPFH features -> global registration (TEASER++ if the
binding is present, else Open3D RANSAC) -> point-to-plane ICP refinement. Writes the
4x4 transform (source->target) and the transformed source cloud.
"""
import argparse, json, os, sys, numpy as np, open3d as o3d
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _progress import stage


def prep(pcd, voxel):
    d = pcd.voxel_down_sample(voxel)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return d, fpfh


def teaser_global(src_d, tgt_d, src_f, tgt_f, voxel):
    """Try TEASER++; return 4x4 or None if unavailable/failed."""
    try:
        import teaserpp_python  # noqa
    except Exception:
        return None
    try:
        # correspondences by nearest feature
        sf = np.asarray(src_f.data).T; tf = np.asarray(tgt_f.data).T
        tree = o3d.geometry.KDTreeFlann(tgt_f)
        sp = np.asarray(src_d.points); tp = np.asarray(tgt_d.points)
        S, T = [], []
        for i in range(len(sp)):
            _, idx, _ = tree.search_knn_vector_xd(src_f.data[:, i], 1)
            S.append(sp[i]); T.append(tp[idx[0]])
        S = np.asarray(S).T; T = np.asarray(T).T
        params = teaserpp_python.RobustRegistrationSolver.Params()
        params.noise_bound = voxel
        params.estimate_scaling = False
        solver = teaserpp_python.RobustRegistrationSolver(params)
        solver.solve(S, T)
        sol = solver.getSolution()
        Tm = np.eye(4); Tm[:3, :3] = sol.rotation; Tm[:3, 3] = sol.translation
        return Tm
    except Exception as e:
        print(f"  TEASER++ failed ({e}); falling back to RANSAC")
        return None


def ransac_global(src_d, tgt_d, src_f, tgt_f, voxel):
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, tgt_d, src_f, tgt_f, True, voxel * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return res.transformation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--voxel", type=float, default=0.05, help="downsample size in meters")
    ap.add_argument("--out", required=True, help="output path prefix")
    args = ap.parse_args()

    stage(1, 4, "Loading point clouds")
    src = o3d.io.read_point_cloud(args.source)
    tgt = o3d.io.read_point_cloud(args.target)
    print(f"source {len(src.points)} pts, target {len(tgt.points)} pts")

    stage(2, 4, "Downsampling + FPFH features")
    src_d, src_f = prep(src, args.voxel)
    tgt_d, tgt_f = prep(tgt, args.voxel)

    stage(3, 4, "Global registration")
    T0 = teaser_global(src_d, tgt_d, src_f, tgt_f, args.voxel)
    method = "teaser++"
    if T0 is None:
        T0 = ransac_global(src_d, tgt_d, src_f, tgt_f, args.voxel); method = "ransac"
    print(f"global ({method}) done; refining with ICP")

    stage(4, 4, "ICP refinement")
    icp = o3d.pipelines.registration.registration_icp(
        src, tgt, args.voxel * 2, T0,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(f"ICP fitness={icp.fitness:.3f} rmse={icp.inlier_rmse:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savetxt(args.out + "_transform.txt", icp.transformation)
    json.dump({"method": method, "fitness": icp.fitness, "rmse": icp.inlier_rmse,
               "transform": icp.transformation.tolist()},
              open(args.out + ".json", "w"), indent=2)
    src.transform(icp.transformation)
    o3d.io.write_point_cloud(args.out + ".ply", src)
    print(f"wrote {args.out}.ply and {args.out}_transform.txt")


if __name__ == "__main__":
    main()
