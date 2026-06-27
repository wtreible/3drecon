#!/usr/bin/env python3
"""Build a tag map: estimate the 3D pose of every ArUco tag in a common frame from
images where tags are co-visible. Anchors the frame at --root-id (identity).

    recon sfm python scripts/localize/aruco_map.py \
        --images projects/mill/localization/survey/*.jpg \
        --intrinsics calib/intrinsics/survey_cam.json \
        --dict DICT_4X4_50 --marker-len-m 0.30 --root-id 0 \
        --out projects/mill/localization/tag_map.yaml

Method: detect markers, solvePnP each -> pose in camera frame; for co-visible pairs
form relative tag->tag transforms; BFS from the root tag averaging duplicate edges
(Markley quaternion averaging) to get each tag's pose in the root frame.
"""
import argparse, glob, json, os, sys, yaml, cv2, numpy as np
from collections import defaultdict, deque
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _progress import track


def rmat_to_quat(R):
    q = np.empty(4)  # w,x,y,z
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q[:] = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[:] = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[:] = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        else:
            s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[:] = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return q / np.linalg.norm(q)


def quat_to_rmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)]])


def avg_transforms(Ts):
    """Average a list of 4x4 transforms: mean translation, Markley quaternion mean."""
    M = np.zeros((4, 4))
    tsum = np.zeros(3)
    for T in Ts:
        q = rmat_to_quat(T[:3, :3])
        M += np.outer(q, q)
        tsum += T[:3, 3]
    w, V = np.linalg.eigh(M)
    q = V[:, -1]
    out = np.eye(4)
    out[:3, :3] = quat_to_rmat(q / np.linalg.norm(q))
    out[:3, 3] = tsum / len(Ts)
    return out


def marker_objp(L):
    h = L / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, nargs="+")
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--dict", required=True)
    ap.add_argument("--marker-len-m", type=float, required=True)
    ap.add_argument("--root-id", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cam = json.load(open(args.intrinsics))
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]])
    dist = np.array(cam.get("dist", []), float)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    objp = marker_objp(args.marker_len_m)

    files = []
    for g in args.images:
        files += sorted(glob.glob(g)) if any(c in g for c in "*?[") else [g]

    edges = defaultdict(list)  # (i,j) -> [T_i_j]
    seen = set()
    for f in track(files, "detect tags", total=len(files)):
        img = cv2.imread(f)
        if img is None:
            continue
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if ids is None:
            continue
        poses = {}  # tag_id -> T_cam_tag (tag->cam)
        for c, i in zip(corners, ids.ravel()):
            ok, rvec, tvec = cv2.solvePnP(objp, c.reshape(-1, 2), K, dist,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rvec)[0]; T[:3, 3] = tvec.ravel()
            poses[int(i)] = T; seen.add(int(i))
        tags = list(poses)
        for a in tags:
            for b in tags:
                if a != b:
                    edges[(a, b)].append(np.linalg.inv(poses[a]) @ poses[b])  # T_a_b

    if args.root_id not in seen:
        raise SystemExit(f"root tag {args.root_id} never detected; seen={sorted(seen)}")

    # BFS from root, composing averaged edge transforms.
    world = {args.root_id: np.eye(4)}
    adj = defaultdict(set)
    for (a, b) in edges:
        adj[a].add(b)
    q = deque([args.root_id])
    while q:
        a = q.popleft()
        for b in adj[a]:
            if b in world:
                continue
            world[b] = world[a] @ avg_transforms(edges[(a, b)])  # T_world_b
            q.append(b)

    missing = seen - set(world)
    out = {"frame": f"tag{args.root_id}", "dict": args.dict,
           "marker_len_m": args.marker_len_m,
           "tags": {int(k): world[k].tolist() for k in sorted(world)}}
    yaml.safe_dump(out, open(args.out, "w"), sort_keys=False)
    print(f"mapped {len(world)} tags into frame of tag {args.root_id}; "
          f"unreachable (no co-visibility path): {sorted(missing) or 'none'} -> {args.out}")


if __name__ == "__main__":
    main()
