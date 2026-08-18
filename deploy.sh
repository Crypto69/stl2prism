#!/bin/sh
# Rebuild and restart the container, stamping the image with the git commit
# so the web UI (top right) and /api/version show what is running.
# Usage, on the NAS:  ./deploy.sh
set -e
cd "$(dirname "$0")"

# The NAS keeps git off the default PATH for non-login shells; look in the
# usual places, and if all else fails read the commit straight from .git.
GIT=""
for c in git /usr/bin/git /usr/local/bin/git /opt/bin/git /usr/local/git/bin/git \
         <app-path>/git/bin/git; do
  if command -v "$c" >/dev/null 2>&1; then GIT="$c"; break; fi
done

if [ -n "$GIT" ]; then
  GIT_SHA="$("$GIT" rev-parse --short HEAD)"
  if [ -n "$("$GIT" status --porcelain --untracked-files=no)" ]; then
    GIT_SHA="${GIT_SHA}-dirty"
  fi
else
  ref="$(sed -n 's/^ref: //p' .git/HEAD)"
  if [ -n "$ref" ] && [ -f ".git/$ref" ]; then
    full="$(cat ".git/$ref")"
  elif [ -n "$ref" ] && [ -f .git/packed-refs ]; then
    full="$(grep " $ref\$" .git/packed-refs | cut -d' ' -f1)"
  else
    full="$(cat .git/HEAD)"          # detached HEAD: the file holds the sha
  fi
  GIT_SHA="$(printf '%s' "$full" | cut -c1-7)"
  [ -n "$GIT_SHA" ] || GIT_SHA=unknown
  echo "note: git not on PATH; commit read from .git (dirty check skipped)"
fi
BUILD_TIME="$(date -u +%Y-%m-%dT%H:%MZ)"
export GIT_SHA BUILD_TIME
echo "building stl2prism ${GIT_SHA} (${BUILD_TIME})"
docker compose build
docker compose up -d
sleep 3
echo "running: $(curl -s http://localhost:8321/api/version || echo '(not up yet)')"
