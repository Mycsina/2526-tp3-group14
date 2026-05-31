#!/usr/bin/env bash
# Sync code to banana.ua.pt home directory
# Usage: ./sync-to-banana.sh
#
# Requires sshpass: sudo pacman -S sshpass  (Arch) / sudo apt install sshpass  (Debian)
#
# One-time setup on banana.ua.pt:
#   mkdir -p ~/cle/2526-tp3-group14

set -euo pipefail

USER="cle14"
SERVER="banana.ua.pt"
PASSWORD="sob0122"
REMOTE_DIR="cle/2526-tp3-group14"

echo "==> Syncing to ${USER}@${SERVER}:${REMOTE_DIR}..."

SSHPASS="${PASSWORD}" sshpass -e rsync -avz --delete \
    --include='src/' --include='src/**' \
    --include='Makefile' --include='*.cu' --include='*.h' --include='*.cuh' \
    --include='*.sh' \
    --exclude='*' \
    ./ "${USER}@${SERVER}:${REMOTE_DIR}/"

echo "==> Done. SSH into ${SERVER} and run:"
echo "    cd ~/${REMOTE_DIR} && make"