#!/usr/bin/env bash
# Sync code to banana.ua.pt home directory
# Usage: ./sync-to-banana.sh
#
# One-time setup on banana.ua.pt:
#   mkdir -p ~/cle/2526-tp3-group14

set -euo pipefail

SERVER="cle14@banana.ua.pt"
REMOTE_DIR="cle/2526-tp3-group14"

echo "==> Syncing to ${SERVER}:${REMOTE_DIR}..."

rsync -avz --delete \
    --include='src/' --include='src/**' \
    --include='Makefile' --include='*.cu' --include='*.h' --include='*.cuh' \
    --include='*.sh' \
    --exclude='*' \
    ./ "${SERVER}:${REMOTE_DIR}/"

echo "==> Done. SSH into ${SERVER} and run:"
echo "    cd ~/${REMOTE_DIR} && make"
