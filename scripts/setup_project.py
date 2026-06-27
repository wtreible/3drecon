#!/usr/bin/env python3
"""recon setup <src> — turn a folder of random files into a pipeline-ready project.

Scans <src> for videos / images / lidar, asks the role of each (data vs calibration vs
skip) and the capture style, organizes them under projects/<name>/, and writes a
recon.json manifest. Multiple data videos become separate **sequences** under seq/<stem>/,
each a self-contained mini-project (own frames/colmap/exports/recon.json) tracked
independently. Runs on the host (stdlib only).
"""
import argparse, glob, os, shlex, shutil, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifest
import tui

VIDEO = (".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v", ".ogv")
IMAGE = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
LIDAR = (".ply", ".bin", ".pcd", ".las", ".laz", ".pts")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBDIRS = ("raw_video", "frames", "images", "colmap", "lidar", "calib", "exports")
# rough JPEG bytes-per-pixel by quality (content-dependent — for size estimates only)
QBPP = {95: 0.30, 90: 0.22, 85: 0.16, 80: 0.12, 75: 0.09, 70: 0.07}
MAXF = 3000  # per-video safety ceiling — high so the sampling RATE drives the count
             # (a low cap would make every rate collapse to the same number)


def probe_videos(recon, rels):
    """Return [(duration_s, w, h)] for placed videos (one container call via cv2)."""
    if not rels:
        return []
    code = ("import cv2,sys\n"
            "for p in sys.argv[1:]:\n"
            "  c=cv2.VideoCapture(p); fps=c.get(cv2.CAP_PROP_FPS) or 30\n"
            "  print('PROBE', round((c.get(cv2.CAP_PROP_FRAME_COUNT) or 0)/fps,2), int(c.get(3)), int(c.get(4)))")
    out = subprocess.run([recon, "sfm", "python", "-c", code, *rels],
                         capture_output=True, text=True).stdout
    res = []
    for ln in out.splitlines():
        if ln.startswith("PROBE"):
            _, d, w, h = ln.split()
            res.append((float(d), int(w), int(h)))
    return res


def estimate(rate, quality, probes):
    """Rough (#frames, MB) for extracting at `rate` fps and JPEG `quality`."""
    frames = sum(min(MAXF, max(1, round(d * rate))) for d, _, _ in probes)
    w, h = max(((w, h) for _, w, h in probes), default=(1920, 1080))
    mb = frames * w * h * QBPP.get(quality, 0.15) / 1e6
    return frames, mb


def ask(prompt, default=None, choices=None):
    """Unified prompt → whiptail dialog (or text fallback): yes/no, menu, or inputbox."""
    if choices in (["y", "n"], ["n", "y"]):
        return "y" if tui.yesno("recon setup", prompt.strip(), default == "y") else "n"
    if choices:
        return tui.menu("recon setup", prompt.strip(), choices, default)
    return tui.inputbox("recon setup", prompt.strip(), default or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="folder of files to ingest")
    ap.add_argument("--move", action="store_true", help="move files instead of copying")
    a = ap.parse_args()
    src = os.path.abspath(a.src)
    if not os.path.isdir(src):
        sys.exit(f"not a directory: {src}")
    place = shutil.move if a.move else shutil.copy2
    rel = lambda p: os.path.relpath(p, REPO)

    files = [f for f in glob.glob(os.path.join(src, "**", "*"), recursive=True) if os.path.isfile(f)]
    vids = sorted(f for f in files if f.lower().endswith(VIDEO))
    imgs = sorted(f for f in files if f.lower().endswith(IMAGE))
    lids = sorted(f for f in files if f.lower().endswith(LIDAR))
    print(f"\nScanned {src}\n  {len(vids)} videos · {len(imgs)} images · {len(lids)} lidar files\n")

    name = ask("Project name", os.path.basename(src.rstrip("/")) or "project")
    proj = os.path.join(REPO, "projects", name)
    captype = ask("Capture style", "handheld_interior",
                  ["drone_aerial", "drone_aerial_orbit", "object_orbit",
                   "handheld_interior", "driving", "other"])

    # --- gather roles first, so we can choose flat vs multi-sequence layout ---
    data_videos, calib_items = [], []
    for v in vids:
        role = ask(f"Video '{os.path.basename(v)}' — role", "data", ["data", "calibration", "skip"])
        if role == "data":
            data_videos.append(v)
        elif role == "calibration":
            cam = ask("  camera name", "cam"); kind = ask("  board", "charuco", ["charuco", "checkerboard"])
            calib_items.append((v, cam, f"{kind}_video"))
    img_role = None
    if imgs:
        img_role = ask(f"{len(imgs)} images — role", "data", ["data", "calibration", "skip"])
        if img_role == "calibration":
            icam = ask("  camera name", "cam"); ikind = ask("  board", "charuco", ["charuco", "checkerboard"])
            calib_items.append((imgs, icam, f"{ikind}_images"))
    do_lidar = bool(lids) and ask(f"{len(lids)} lidar files — import?", "y", ["y", "n"]) == "y"

    matcher = "sequential" if captype in ("handheld_interior", "driving") else "exhaustive"
    defaults = {"sparse": {"matcher": matcher, "mapper": "colmap"}}

    def mkdirs(d):
        for s in SUBDIRS:
            os.makedirs(os.path.join(d, s), exist_ok=True)

    def place_shared(into):
        cal = []
        for item, cam, kind in calib_items:
            cd = os.path.join(into, "calib", f"{cam}_calib"); os.makedirs(cd, exist_ok=True)
            for f in (item if isinstance(item, list) else [item]):
                place(f, os.path.join(cd, os.path.basename(f)))
            cal.append({"path": rel(cd), "kind": kind, "camera": cam})
        lid = []
        if do_lidar:
            for f in lids:
                place(f, os.path.join(into, "lidar", os.path.basename(f)))
            lid = [rel(os.path.join(into, "lidar", os.path.basename(f))) for f in lids]
        return cal, lid

    base = {"capture": {"type": captype, "notes": ""},
            "camera": {"model": None, "intrinsics": None}, "defaults": defaults}
    recon = os.path.join(REPO, "bin", "recon")
    hint = "recon" if shutil.which("recon") else recon
    projrel = f"projects/{name}"
    stem = lambda p: os.path.splitext(os.path.basename(p))[0].replace(" ", "_")

    def scene_split(video_rel, min_scene):
        """Detect fade/cut scenes and return one sequence dict per scene, all sharing the
        original video plus a [start,end] time range (frame-domain split — no re-encode).
        A video with <=1 scene returns a single plain sequence."""
        import json as _j
        o = subprocess.run([recon, "sfm", "python", "scripts/sfm/detect_scenes.py",
                            "--video", video_rel, "--json", "--min-scene", str(min_scene)],
                           capture_output=True, text=True).stdout
        scenes = []
        for ln in o.splitlines():
            if ln.strip().startswith("{"):
                scenes = _j.loads(ln.strip()).get("scenes", [])
                break
        if len(scenes) <= 1:
            return [{"name": stem(video_rel), "video": video_rel}]
        print(f"  {os.path.basename(video_rel)} -> {len(scenes)} scenes (frames/<scene>/)")
        return [{"name": f"{stem(video_rel)}_scene{i:02d}", "video": video_rel,
                 "scene": [round(t0, 2), round(t1, 2)]} for i, (t0, t1) in enumerate(scenes, 1)]

    # A prior run's frames/colmap outputs are written as root by the container, so a host-side
    # `rm -rf projects/<name>` silently fails on them and leaves stale files that mask a re-setup.
    # Clean an existing project via the container (root) so re-running setup truly starts fresh.
    if os.path.isdir(proj):
        if ask(f"Project '{name}' already exists — remove it and start fresh?", "y", ["y", "n"]) == "y":
            subprocess.run([recon, "sfm", "bash", "-lc", f"rm -rf {shlex.quote('/work/' + projrel)}"],
                           capture_output=True)
            print(f"  removed existing projects/{name}")

    mkdirs(proj)
    cal, lid = place_shared(proj)            # calibration + lidar shared at project level
    placed = []
    for v in data_videos:                    # videos live flat in raw_video/
        vdst = os.path.join(proj, "raw_video", os.path.basename(v)); place(v, vdst)
        placed.append(rel(vdst))
    # optionally split videos with fade/scene cuts into per-scene sequences (frame-domain:
    # one video, each scene -> its own frames/<scene>/ via a [start,end] time range)
    if placed and ask("Detect scene/fade cuts and split into per-scene sequences?", "n", ["y", "n"]) == "y":
        msmap = {"keep all scenes (>=2s)": 2, "drop brief shots (>=8s)": 8,
                 "rooms only (>=15s)": 15}
        min_scene = msmap[ask("Minimum scene length (shorter scenes are dropped)",
                              "drop brief shots (>=8s)", list(msmap))]
        seq_list = [s for vr in placed for s in scene_split(vr, min_scene)]
    else:
        seq_list = [{"name": stem(vr), "video": vr} for vr in placed]

    targets, menu_target = [], projrel
    multiseq = len(seq_list) > 1
    if multiseq:
        # type-first layout: each sequence -> colmap/<seq>, frames/<seq>, exports/<seq>
        manifest.save(proj, {**base, "name": name, "created": manifest.now(), "sequences": seq_list,
                             "inputs": {"calibration": cal, "lidar": lid, "data_images": None}, "runs": []})
        targets = [(s["name"], s["name"], s["video"], s.get("scene")) for s in seq_list]
        print(f"\n✓ Project ready: projects/{name}  ({len(seq_list)} sequences — colmap/<seq>, frames/<seq>, exports/<seq>)")
    else:
        video_rels = [s["video"] for s in seq_list]
        m = {**base, "name": name, "created": manifest.now(),
             "inputs": {"data_videos": video_rels, "data_images": None, "calibration": cal, "lidar": lid}, "runs": []}
        for vr in video_rels:
            targets.append(("", stem(vr), vr, None))
        if img_role == "data":
            for f in imgs:
                place(f, os.path.join(proj, "images", os.path.basename(f)))
            m["inputs"]["data_images"] = rel(os.path.join(proj, "images"))
        manifest.save(proj, m)
        print(f"\n✓ Project ready: projects/{name}   (manifest: projects/{name}/recon.json)")

    # --- frame extraction: pick rate + quality (with size estimates), then run with a gauge ---
    if targets and ask("Extract frames now?", "y", ["y", "n"]) == "y":
        # probe each distinct video once; per-target duration is its scene window if split
        uniq = list(dict.fromkeys(vr for _, _, vr, _ in targets))
        pd = dict(zip(uniq, probe_videos(recon, uniq)))
        probes = []
        for _, _, vr, scene in targets:
            d, w, h = pd.get(vr, (60.0, 1920, 1080))
            probes.append((scene[1] - scene[0] if scene else d, w, h))
        probes = probes or [(60.0, 1920, 1080)]
        rmap = {}
        for r in (1, 2, 3, 5):
            fr, mb = estimate(r, 90, probes)
            rmap[f"{r} fps  (~{fr} frames, ~{mb:.0f} MB)"] = r
        rate = rmap[tui.menu("Frame sampling", "Sampling rate", list(rmap), list(rmap)[1])]
        qmap = {}
        for nm, q in (("high", 95), ("medium", 85), ("low", 75)):
            _, mb = estimate(rate, q, probes)
            qmap[f"{nm} — q{q}  (~{mb:.0f} MB)"] = q
        quality = qmap[tui.menu("Frame quality", f"JPEG quality at {rate} fps", list(qmap), list(qmap)[0])]
        # pass the same ceiling used in the estimate so actual count ≈ estimate (rate-driven)
        extra = ["--target-fps", str(rate), "--jpeg-quality", str(quality), "--max-frames", str(MAXF)]
        n = len(targets)
        if tui.have_whiptail():
            # ONE gauge held open across all sequences (no close/reopen between videos);
            # the gauge's text is updated per video via the "XXX\n<pct>\n<msg>\nXXX" protocol.
            g = subprocess.Popen(["whiptail", "--title", "recon setup", "--gauge",
                                  "Preparing…", "8", "66", "0"], stdin=subprocess.PIPE, text=True)

            def gset(pct, msg=None):
                try:
                    g.stdin.write(f"XXX\n{pct}\n{msg}\nXXX\n" if msg else f"{pct}\n"); g.stdin.flush()
                    return True
                except (BrokenPipeError, ValueError):
                    return False
            for i, (seq, label, _, _) in enumerate(targets):
                seq_args = (["--seq", seq] if seq else [])
                if not gset(0, f"Extracting frames — {label}  ({i + 1}/{n})"):
                    break
                p = subprocess.Popen([recon, "make", "frames", projrel, *seq_args, *extra, "--percent"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                for line in p.stdout:
                    s = line.strip()
                    if s.isdigit() and 0 <= int(s) <= 100 and not gset(s):
                        break
                p.wait()
            try:
                g.stdin.close()
            except Exception:
                pass
            g.wait()
        else:
            for seq, label, _, _ in targets:
                seq_args = (["--seq", seq] if seq else [])
                print(f"\nExtracting frames — {label}  (rate {rate} fps, q{quality})")
                subprocess.run([recon, "make", "frames", projrel, *seq_args, *extra])

    if ask("Open the outputs menu now?", "n", ["y", "n"]) == "y":
        subprocess.run([recon, "menu", menu_target, "--configure"])
    else:
        print(f"\nNext:  {hint} menu {menu_target} --configure")


if __name__ == "__main__":
    main()
