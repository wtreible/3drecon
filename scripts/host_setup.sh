#!/usr/bin/env bash
# host_setup.sh — one-time host setup for Docker + GPU on WSL2 (Ubuntu 24.04).
# Run this with sudo:   sudo bash scripts/host_setup.sh
# (or paste line-by-line). Do NOT install any NVIDIA *Linux driver* in WSL — CUDA reaches
# containers through the Windows driver; we only add the container runtime here.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then echo "Run as root: sudo bash $0" >&2; exit 1; fi
TARGET_USER="${SUDO_USER:-$USER}"

echo "==> Removing any conflicting/old docker packages (ignore errors)"
apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

echo "==> Installing Docker CE from Docker's official apt repo"
apt-get update
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
. /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "==> Installing NVIDIA Container Toolkit"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit

echo "==> Wiring the NVIDIA runtime into Docker"
nvidia-ctk runtime configure --runtime=docker

echo "==> Enabling + starting docker (systemd is on in this WSL distro)"
systemctl enable docker
systemctl restart docker

echo "==> Adding '$TARGET_USER' to the docker group (re-login or 'newgrp docker' to apply)"
usermod -aG docker "$TARGET_USER"

echo
echo "Done. Verify GPU passthrough (as $TARGET_USER, after 'newgrp docker'):"
echo "    docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi"
