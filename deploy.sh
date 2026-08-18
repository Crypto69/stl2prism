#!/bin/sh
# Rebuild and restart the container, stamping the image with the git commit
# so the web UI (top right) and /api/version show what is running.
# Usage, on the NAS:  ./deploy.sh
set -e
cd "$(dirname "$0")"
GIT_SHA="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  GIT_SHA="${GIT_SHA}-dirty"
fi
BUILD_TIME="$(date -u +%Y-%m-%dT%H:%MZ)"
export GIT_SHA BUILD_TIME
echo "building stl2prism ${GIT_SHA} (${BUILD_TIME})"
docker compose build
docker compose up -d
echo "running: $(curl -s http://localhost:8321/api/version || echo '(not up yet)')"
