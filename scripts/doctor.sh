#!/usr/bin/env bash
# doctor.sh — verification smoke tests for the reconstruction environment.
# Usage:  recon doctor    (or)    bash scripts/doctor.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE=(docker compose -f "$REPO_ROOT/docker/docker-compose.yml")
pass=0; fail=0
check() { # check "name" command...
  local name="$1"; shift
  if "$@" >/tmp/recon_doctor.out 2>&1; then echo "  PASS  $name"; pass=$((pass+1));
  else echo "  FAIL  $name"; sed 's/^/        /' /tmp/recon_doctor.out | tail -8; fail=$((fail+1)); fi
}

echo "[1] Host: GPU visible to Docker"
check "nvidia-smi in container" \
  docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi -L

echo "[2] recon-sfm: COLMAP / CV stack"
check "colmap CLI"      "${COMPOSE[@]}" run --rm sfm colmap -h
check "glomap CLI"      "${COMPOSE[@]}" run --rm sfm glomap -h
check "python cv/geom"  "${COMPOSE[@]}" run --rm sfm python -c "import cv2,cv2.aruco,open3d,pycolmap,numpy;print('ok')"

echo "[3] recon-dl: torch / gsplat / nerfstudio / vggt"
check "torch sees 5090" "${COMPOSE[@]}" run --rm dl python -c \
  "import torch;assert torch.cuda.is_available();print(torch.cuda.get_device_name(0));assert torch.cuda.get_device_capability(0)[0]>=12,'sm<120'"
# Real rasterization: proves gsplat's sm_120 kernels EXECUTE (not just import) and warms
# the persistent JIT cache. First run compiles (~8 min); subsequent runs are instant.
check "gsplat sm120 kernel" "${COMPOSE[@]}" run --rm dl python -c \
"import torch as t, gsplat as g; d='cuda'; N=64; \
m=t.randn(N,3,device=d); q=t.randn(N,4,device=d); s=t.rand(N,3,device=d)*.1; \
o=t.rand(N,device=d); c=t.rand(N,3,device=d); v=t.eye(4,device=d)[None]; \
K=t.tensor([[[150.,0,75],[0,150,75],[0,0,1]]],device=d); \
out,_,_=g.rasterization(m,q,s,o,c,v,K,150,150); t.cuda.synchronize(); \
print('gsplat sm120 kernel ok', tuple(out.shape))"
check "nerfstudio CLI"  "${COMPOSE[@]}" run --rm dl ns-train --help
check "hloc import"     "${COMPOSE[@]}" run --rm dl python -c "import hloc;print('hloc ok')"
check "vggt import"     "${COMPOSE[@]}" run --rm dl python -c "import vggt;print('vggt ok')"

echo
echo "Summary: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
