#!/usr/bin/env bash
# process_and_train.sh — images/video -> nerfstudio (splat OR nerf), train, export, clean.
# Run inside recon-dl. Shared by the `splat` and `nerf` products.
#
#   recon dl bash scripts/splat/process_and_train.sh --project projects/mill \
#       --data projects/mill/frames --type images --method splatfacto
#   ... --method nerfacto      # NeRF instead of a splat (exports a mesh)
#   ... --colmap projects/mill/colmap/sparse/1   # reuse an existing COLMAP model
#
# splatfacto* -> exports/splat.ply (+ splat_clean.ply);  nerfacto -> exports/mesh.ply
set -euo pipefail
DATA="" PROJECT="" TYPE="images" COLMAP="" METHOD="splatfacto" ITERS="30000" CLEAN=1 OUTBASE="" EXPDIR=""
while [[ $# -gt 0 ]]; do case "$1" in
  --data) DATA="$2"; shift 2;;
  --project) PROJECT="$2"; shift 2;;
  --type) TYPE="$2"; shift 2;;            # images | video
  --colmap) COLMAP="$2"; shift 2;;        # existing model dir (skip COLMAP)
  --method) METHOD="$2"; shift 2;;        # splatfacto | splatfacto-big | nerfacto
  --iterations) ITERS="$2"; shift 2;;
  --out-base) OUTBASE="$2"; shift 2;;     # nerfstudio base (default <project>/nerfstudio)
  --exports) EXPDIR="$2"; shift 2;;       # exports dir (default <project>/exports)
  --no-clean) CLEAN=0; shift;;
  *) echo "unknown arg $1" >&2; exit 2;;
esac; done
[[ -n "$DATA" && -n "$PROJECT" ]] || { echo "need --data and --project" >&2; exit 2; }
# ns-process-data resolves --colmap-model-path RELATIVE TO --output-dir, so a relative
# path gets mis-joined and "does not exist". Make it absolute (cwd is /work in-container).
[[ -n "$COLMAP" && "$COLMAP" != /* ]] && COLMAP="$(realpath "$COLMAP")"

PROC="${OUTBASE:-$PROJECT/nerfstudio}/processed"   # method-agnostic, reusable across splat/nerf
RUNS="${OUTBASE:-$PROJECT/nerfstudio}/runs"
EXPORTS="${EXPDIR:-$PROJECT/exports}"
mkdir -p "$PROC" "$EXPORTS"

echo "[1/4] ns-process-data ($TYPE)"
if [[ -n "$COLMAP" ]]; then
  ns-process-data "$TYPE" --data "$DATA" --output-dir "$PROC" --skip-colmap --colmap-model-path "$COLMAP"
elif [[ ! -f "$PROC/transforms.json" ]]; then
  ns-process-data "$TYPE" --data "$DATA" --output-dir "$PROC"
else
  echo "    reusing existing $PROC/transforms.json"
fi

echo "[2/4] ns-train $METHOD ($ITERS iters)"
EXTRA=()
# Scale regularization prevents the needle/spike gaussians that wreck third-party viewers.
[[ "$METHOD" == splatfacto* ]] && EXTRA+=(--pipeline.model.use-scale-regularization True)
ns-train "$METHOD" --data "$PROC" --output-dir "$RUNS" \
    --max-num-iterations "$ITERS" --viewer.quit-on-train-completion True "${EXTRA[@]}"

CFG=$(ls -t "$RUNS"/*/"$METHOD"/*/config.yml | head -1)
echo "[3/4] export  (config: $CFG)"
if [[ "$METHOD" == splatfacto* ]]; then
  ns-export gaussian-splat --load-config "$CFG" --output-dir "$EXPORTS"
  if [[ "$CLEAN" == 1 ]]; then
    echo "[4/4] clean floaters -> exports/splat_clean.ply"
    python "$(dirname "$0")/clean_ply.py" --in "$EXPORTS/splat.ply" --out "$EXPORTS/splat_clean.ply"
  fi
  echo "Done -> $EXPORTS/splat_clean.ply  (view in superspl.at)"
else
  ns-export poisson --load-config "$CFG" --output-dir "$EXPORTS" \
    --num-points 1000000 --remove-outliers True --normal-method open3d
  echo "[4/4] (nerf: mesh export)"
  echo "Done -> $EXPORTS/poisson_mesh.ply  (open in MeshLab/CloudCompare)"
fi
