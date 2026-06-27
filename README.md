# 3D Reconstruction Environment

A Docker-based, multi-use setup for photogrammetry / SfM / gaussian splatting, built for an
**RTX 5090 (Blackwell, `sm_120`)** on **WSL2 / Ubuntu 24.04**. Everything is compiled for
`sm_120` (CUDA 12.9, PyTorch cu128) because stock COLMAP/gsplat/nerfstudio builds don't run on
Blackwell out of the box.

## Images (`docker/`)

| Image | What's in it |
|---|---|
| `recon-sfm` | COLMAP + GLOMAP (CUDA, `sm_120`), OpenCV-contrib (ArUco/ChArUco), Open3D, TEASER++, pycolmap, ffmpeg, exiftool. **Torch-free.** |
| `recon-dl` (FROM `recon-sfm`) | PyTorch 2.x cu128, gsplat, nerfstudio (splatfacto), hloc + learned matchers, VGGT. |
| `odm` | `opendronemap/odm` for orthomosaic / DEM / mesh / dense cloud. |

The whole repo is bind-mounted at `/work` inside every container, so `projects/`, `calib/`,
`scripts/`, and `data/` are all visible. Model weights cache to `data/`.

## One-time setup

```bash
# 1) host: Docker CE + NVIDIA Container Toolkit (needs sudo; run it yourself)
sudo bash scripts/host_setup.sh
newgrp docker            # or log out/in so docker works without sudo

# 2) verify the GPU is visible to Docker
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi

# 3) build the images (recon-sfm first, then recon-dl, then pull ODM)
bin/recon build

# 4) smoke-test everything
bin/recon doctor
```

> Do **not** install an NVIDIA Linux driver inside WSL — CUDA comes through the Windows driver.

## The `recon` dispatcher

```
recon build [sfm|dl]      build images (default: all + pull ODM)
recon doctor              run smoke tests
recon setup <src>         ingest a folder of files -> pipeline-ready project (+ manifest)
recon scenes <video>      detect fade/scene cuts (--split to write per-scene clips)
recon products <proj>     show the OUTPUT catalog for a project (what you can make)
recon menu <proj> [-c]    interactive multi-select picker (+ per-task option prompts with -c)
recon make <out> <proj>   make an output (auto-wires paths): frames|sparse|dense|splat|nerf|vggt|ortho|...
recon sfm   <cmd...>      run in recon-sfm   (colmap, glomap, python scripts/...)
recon dl    <cmd...>      run in recon-dl    (ns-train, python scripts/..., vggt)
recon odm   <args...>     run OpenDroneMap
recon colmap <args...>    shortcut for: recon sfm colmap
recon shell [sfm|dl]      interactive shell
```

## Ingesting data (`recon setup`)

`recon setup <folder>` turns a folder of random files into a pipeline-ready project: it scans
for videos/images/lidar, asks the role of each (data vs calibration vs skip) and the capture
style, organizes everything under `projects/<name>/`, and writes a **`recon.json` manifest**.

The manifest records capture metadata (type, camera, which video is calibration vs data),
recommended SfM defaults for the capture style, and a **run log** — every `recon make` appends
what it produced, with which options, when, and pass/fail. `recon products` surfaces a summary
(`capture=…, N run(s), last: …`). This makes runs reproducible and lets the menu/products be
input-aware.

**Scene/fade cuts → sequences.** A video with fades or hard cuts between segments (e.g. a
walk-through that fades between rooms) reconstructs badly as one piece — the cuts break SfM. Setup
offers to **detect and split** it. The split is **frame-domain**: the one video is kept as-is and
each detected scene becomes a sequence carrying a `[start,end]` time range, so frames extract
straight into `frames/<scene>/` with no re-encoding or clip files. A min-scene-length picker (keep
≥2s / drop brief ≥8s / rooms ≥15s) discards transition shots. Detection is fade-to-black
(brightness dips) + hard cuts (histogram jumps), via cv2+ffmpeg. Standalone: `recon scenes <video>
[--min-scene N]` lists scenes; `recon scenes <video> --split` writes per-scene clips for ad-hoc use.

```
recon setup ~/Downloads/mill_walk      # interactive: classify files, set capture type
recon make frames projects/mill_walk   # extract frames from the ingested video
recon menu projects/mill_walk --configure
```

### Source-video archive (`demo_videos/`, git-ignored)

The original videos for each project live in `demo_videos/<project>/` so a project can be rebuilt
from scratch with different settings (frame rate, JPEG quality, capture defaults, scene-splitting).
`recon setup` **copies** (never moves), so the archive stays intact. To re-run a project clean:

```
recon setup demo_videos/<name>         # re-ingest (offers to remove the old project first)
```
Setup detects an existing project and offers to wipe it before recreating, so you don't have to
`rm` by hand. Or re-extract just the frames in place:
`recon make frames projects/<name> [--seq <s>] --target-fps 4 --jpeg-quality 80`.

## Outputs (the products menu)

Reconstruction isn't one pipeline — given a project's inputs, several outputs are possible.
`recon products <proj>` inspects what you have (images, COLMAP model, calibration, ArUco tags,
LiDAR, GPS) and lists which outputs are **done**, **ready**, or still **need input**, each with a
ready-to-run command. Then `recon make <output> <proj>` builds it (auto-wiring the standard paths
— frames dir, largest sparse model).

| output | needs | produces |
|---|---|---|
| `frames` | a data video | sharp, evenly-spaced frames extracted from the video |
| `sparse` | images | COLMAP SfM: sparse cloud + camera poses |
| `dense` | images | COLMAP MVS: dense point cloud + Poisson mesh (auto-runs `sparse` first if needed) |
| `splat` | images/sparse | gaussian splat `.ply` (scale-reg, auto-cleaned) |
| `nerf` | images/sparse | NeRF (nerfacto) + Poisson mesh |
| `ortho` | GPS-tagged images | OpenDroneMap orthomosaic + DEM + mesh |
| `vggt` | images | feed-forward poses + depth + point cloud (no COLMAP) |
| `localize` | ArUco tags + calib | camera extrinsics in the tag frame |
| `register` | LiDAR + a cloud | LiDAR aligned into the reconstruction |
| `calibrate` | ChArUco images | reusable camera-intrinsics `.json` |

```
recon products projects/demo_castle      # see the menu
recon make dense  projects/demo_castle    # COLMAP dense cloud + mesh
recon make nerf   projects/demo_castle    # NeRF instead of a splat
recon make vggt   projects/demo_castle    # fast feed-forward, no COLMAP
```

Splat exports are auto-cleaned (`clean_ply.py`); preview any `.ply` headlessly with
`scripts/splat/render_ply.py` (`--mode interior` for indoor scenes).

Every `recon make` tees its full output to `projects/<P>/logs/<runid>_<tool>.log` and records the
`runid` + log path in the manifest's run entry (alongside status/options/seq).

## Project layout

A single-video / single-image-set project is **flat**:

```
projects/<name>/
  raw_video/  frames/  images/  colmap/  exports/  nerfstudio/  localization/  lidar/  logs/
  recon.json
```

A project with **multiple data videos** (or a scene-split one) is **type-first**: each video is a
**sequence**, and every output type holds a per-sequence subdir. The single `recon.json` lists the
sequences; each run is tagged with its `seq`. Target one with `--seq <name>`:

```
projects/<name>/
  raw_video/<video>.mp4
  frames/<seq>/   colmap/<seq>/{sparse,dense}   exports/<seq>/   nerfstudio/<seq>/
  recon.json      # sequences:[{name, video, scene:[t0,t1]?}], runs:[{seq, runid, log, ...}]

recon make sparse projects/<name> --seq <seq>     # routes every stage into <type>/<seq>
recon menu  projects/<name>                        # pick a sequence, then outputs
```

Shared across the project (not per-sequence): `calib/`, `lidar/`, `logs/`.

```
calib/intrinsics/<camera>.json    # reusable per-camera calibration (calibrate once)
calib/boards/                     # ChArUco board definitions + printables
data/                             # shared datasets + model-weight caches (git-ignored)
```

---

## Workflows

### A. Camera intrinsics (ChArUco)
```bash
recon sfm python scripts/calib/make_charuco_board.py --out calib/boards/charuco_5x7 \
    --squares-x 5 --squares-y 7 --square-len-mm 40 --marker-len-mm 30 --dict DICT_5X5_1000
# print at 100%, MEASURE a square, update square_len_m in the .yaml, shoot ~20 views, then:
recon sfm python scripts/calib/charuco_calibrate.py --board calib/boards/charuco_5x7.yaml \
    --images calib/intrinsics/iphone15/*.jpg --name iphone15 --out calib/intrinsics
```
Produces `calib/intrinsics/iphone15.json` (+ a COLMAP camera line) reused by every other tool.

### B. Drone flight → photogrammetry **and** splat
```bash
# SfM (fixed intrinsics + global mapper for large aerial sets)
recon sfm python scripts/sfm/run_colmap.py auto --project projects/drone_site1 \
    --images projects/drone_site1/images --matcher exhaustive --mapper glomap \
    --camera-json calib/intrinsics/drone_cam.json
# georeference from GPS EXIF
recon sfm python scripts/sfm/geo_register.py --images projects/drone_site1/images \
    --input projects/drone_site1/colmap/sparse/0 --output projects/drone_site1/colmap/sparse_geo
# classic products (ortho/DEM/mesh)
recon odm --project-path /datasets drone_site1
# gaussian splat from the same imagery
recon dl bash scripts/splat/process_and_train.sh --type images \
    --data projects/drone_site1/images --project projects/drone_site1 \
    --colmap projects/drone_site1/colmap/sparse_geo
```

### C. Mill phone video → splat
```bash
recon sfm python scripts/sfm/video_to_frames.py --video projects/mill/raw_video/walk1.mp4 \
    --out projects/mill/frames --target-fps 2 --max-frames 600
recon sfm python scripts/sfm/run_colmap.py auto --project projects/mill \
    --images projects/mill/frames --matcher sequential --mapper colmap \
    --camera-json calib/intrinsics/iphone15.json
recon dl bash scripts/splat/process_and_train.sh --type images \
    --data projects/mill/frames --project projects/mill --colmap projects/mill/colmap/sparse/0
```
For repetitive industrial texture where SIFT struggles, use **hloc** (LightGlue/ALIKED) from
`recon-dl`, and consider **VGGT** as an initializer for low-overlap segments.

### D. Localize fixed Ubiquiti cameras via ArUco
```bash
# 1) build a tag map (poses of all tags in one frame) from a handheld survey of the tags
recon sfm python scripts/localize/aruco_map.py --images projects/mill/localization/survey/*.jpg \
    --intrinsics calib/intrinsics/survey_cam.json --dict DICT_4X4_50 --marker-len-m 0.30 \
    --root-id 0 --out projects/mill/localization/tag_map.yaml
# 2) localize each fixed camera from one snapshot that sees mapped tags
recon sfm python scripts/localize/cam_extrinsics.py --tag-map projects/mill/localization/tag_map.yaml \
    --intrinsics calib/intrinsics/ubiquiti_g4.json \
    --views cam01=projects/mill/localization/cam01.jpg cam02=projects/mill/localization/cam02.jpg \
    --out projects/mill/localization
# -> camera_poses.json + COLMAP cameras.txt/images.txt (poses are world->cam, COLMAP-ready)
# optionally triangulate structure against those fixed poses:
recon sfm python scripts/sfm/run_colmap.py triangulate --project projects/mill \
    --images projects/mill/localization --input projects/mill/localization
```

### E. LiDAR → common frame (exploratory; no targets yet)
```bash
recon sfm python scripts/register/cloud_align.py --source projects/mill/lidar/scan01.ply \
    --target projects/mill/colmap/dense/fused.ply --voxel 0.05 \
    --out projects/mill/lidar/scan01_aligned
```
This anchors LiDAR to the fiducial-defined metric frame by cloud-to-cloud registration
(TEASER++/RANSAC → ICP). For proper targetless LiDAR↔camera calibration later, add a
`recon-lidar` image around Koide's `direct_visual_lidar_calibration` (ROS2); a LiDAR-visible
planar/board target will materially improve accuracy.

## Requirements & portability

- **Linux + NVIDIA only.** The stack is CUDA end-to-end (CUDA COLMAP, torch cu128, gsplat). It runs
  on native Linux or WSL2 with an NVIDIA GPU and the NVIDIA Container Toolkit. **macOS is not
  supported** — there is no CUDA on macOS and Docker can't pass an NVIDIA GPU through; the
  containers won't run there (a Mac can only drive a remote Linux GPU box).
- **GPU arch.** The images default to **`sm_120` (RTX 5090)**. On another NVIDIA GPU, rebuild with
  the right arch, e.g. `CUDA_ARCH=89 TORCH_CUDA_ARCH_LIST=8.9 recon build` (4090) — the Dockerfile
  ARGs / env already accept this (`CUDA_TAG`, `CUDA_ARCH`, `TORCH_CUDA_ARCH_LIST`).
- **Reproducibility.** Pinned: CUDA base tag, COLMAP 4.0.4, GLOMAP 1.2.0, gsplat 1.4.0. Floating
  (latest at build time): torch/torchvision, nerfstudio, hloc, VGGT, and apt/pip deps — so a build
  today is not guaranteed identical to one months from now. Build on a comparable machine soon for
  best results.

## Notes / sharp edges
- **Containers run as the invoking host user** (`--user $(id -u):$(id -g)` for `sfm`/`dl`), so
  outputs (`frames/`, `colmap/`, `exports/`) are owned by you, not root — host-side `rm`/edits just
  work. `HOME` points at `/work/data/home`. Override with `RECON_ROOT=1 recon …` for root-only
  maintenance (apt, chowning). ODM still runs as root.
- COLMAP-with-CUDA is the most fragile build; if it fights the toolchain, **Brush** (Rust/wgpu)
  is a CUDA-arch-free splat fallback, and CPU COLMAP still works (slower).
- gsplat/VGGT wheels may lag the exact torch+cu128 combo — the Dockerfiles build from source with
  `TORCH_CUDA_ARCH_LIST=12.0` to cover that.
- VGGT has a per-forward image-count ceiling: use it as an initializer/segment tool, not a
  whole-flight SfM replacement.
