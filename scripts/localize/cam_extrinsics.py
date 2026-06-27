#!/usr/bin/env python3
"""Localize cameras into the tag-map frame from one image per camera, and export
COLMAP pose priors (images.txt + cameras.txt).

    recon sfm python scripts/localize/cam_extrinsics.py \
        --tag-map projects/mill/localization/tag_map.yaml \
        --intrinsics calib/intrinsics/ubiquiti_g4.json \
        --views cam01=projects/mill/localization/cam01.jpg \
                cam02=projects/mill/localization/cam02.jpg \
        --out projects/mill/localization

For each view: gather every mapped tag's 4 world corners <-> image corners, run a
single solvePnP -> world->camera pose (COLMAP's convention). Writes poses.json plus
COLMAP cameras.txt/images.txt you can feed to run_colmap.py for fixed-pose triangulation.
"""
import argparse, json, os, sys, yaml, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _progress import track

# reuse the quaternion helper
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "amap", str(pathlib.Path(__file__).with_name("aruco_map.py")))
amap = importlib.util.module_from_spec(spec); spec.loader.exec_module(amap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-map", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--views", required=True, nargs="+", help="name=path ...")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tm = yaml.safe_load(open(args.tag_map))
    world_tag = {int(k): np.array(v) for k, v in tm["tags"].items()}
    L = tm["marker_len_m"]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, tm["dict"]))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    objp = amap.marker_objp(L)

    cam = json.load(open(args.intrinsics))
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]])
    dist = np.array(cam.get("dist", []), float)

    os.makedirs(args.out, exist_ok=True)
    poses, cam_lines, img_lines = {}, [], []
    cam_lines.append("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS")
    cam_lines.append(f"1 OPENCV {cam['width']} {cam['height']} "
                     f"{cam['fx']:.6f} {cam['fy']:.6f} {cam['cx']:.6f} {cam['cy']:.6f} "
                     + " ".join(f"{x:.8f}" for x in (cam.get('dist', [0, 0, 0, 0]) + [0, 0, 0, 0])[:4]))
    img_lines.append("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME")

    for idx, spec_ in enumerate(track(args.views, "localize cams", total=len(args.views)), 1):
        name, path = spec_.split("=", 1)
        img = cv2.imread(path)
        if img is None:
            print(f"  !! cannot read {path}"); continue
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if ids is None:
            print(f"  !! no tags in {name}"); continue
        OBJ, IMG = [], []
        for c, i in zip(corners, ids.ravel()):
            i = int(i)
            if i not in world_tag:
                continue
            wc = (world_tag[i] @ np.hstack([objp, np.ones((4, 1))]).T).T[:, :3]  # world corners
            OBJ.append(wc); IMG.append(c.reshape(-1, 2))
        if len(OBJ) < 1:
            print(f"  !! {name}: detected tags not in map"); continue
        OBJ = np.vstack(OBJ).astype(np.float32); IMG = np.vstack(IMG).astype(np.float32)
        ok, rvec, tvec, _ = cv2.solvePnPRansac(OBJ, IMG, K, dist, reprojectionError=3.0)
        if not ok:
            print(f"  !! {name}: solvePnP failed"); continue
        R = cv2.Rodrigues(rvec)[0]; t = tvec.ravel()
        C = (-R.T @ t)  # camera center in world
        q = amap.rmat_to_quat(R)
        poses[name] = {"R_world2cam": R.tolist(), "t": t.tolist(), "center": C.tolist(),
                       "n_tags": len(OBJ) // 4}
        img_lines.append(f"{idx} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} "
                         f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 {name}")
        img_lines.append("")  # COLMAP wants a (points2D) line; empty = no 2D points yet
        print(f"  {name}: localized from {len(OBJ)//4} tags, center={C.round(3)}")

    json.dump(poses, open(os.path.join(args.out, "camera_poses.json"), "w"), indent=2)
    open(os.path.join(args.out, "cameras.txt"), "w").write("\n".join(cam_lines) + "\n")
    open(os.path.join(args.out, "images.txt"), "w").write("\n".join(img_lines) + "\n")
    print(f"wrote camera_poses.json + COLMAP cameras.txt/images.txt to {args.out}")


if __name__ == "__main__":
    main()
