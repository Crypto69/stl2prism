#!/bin/sh
# Pull latest, rebuild and restart the container, stamping the image with the
# git commit so the web UI (top right) and /api/version show what is running.
# Usage, on the NAS:  ./deploy.sh            (pulls main first)
#                     ./deploy.sh --no-pull  (build what is checked out)
set -e
cd "$(dirname "$0")"

# The NAS keeps git off the default PATH for non-login shells; look in the
# usual places, and if all else fails read the commit straight from .git.
GIT=""
for c in git /usr/bin/git /usr/local/bin/git /opt/bin/git /usr/local/git/bin/git \
         /Volume1/@apps/git/bin/git; do
  if command -v "$c" >/dev/null 2>&1; then GIT="$c"; break; fi
done

# On this NAS `git` is often only a shell alias, invisible to scripts. Fall back
# to git inside a throwaway container so pulling still works.
if [ -z "$GIT" ] && command -v docker >/dev/null 2>&1; then
  echo "note: no git binary; using docker alpine/git"
  GIT="docker run --rm -v $(pwd):/repo -w /repo -e HOME=/tmp alpine/git -c safe.directory=/repo"
fi

if [ -n "$GIT" ]; then
  if [ "$1" != "--no-pull" ]; then
    echo "pulling latest..."
    $GIT pull --ff-only
    chmod -R a+rX . 2>/dev/null || true   # TOS share ACLs strip modes on pull
  fi
  GIT_SHA="$($GIT rev-parse --short HEAD)"
  if [ -n "$($GIT status --porcelain --untracked-files=no)" ]; then
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
  echo "note: git not found; NOT pulled, commit read from .git (dirty check skipped)"
fi
BUILD_TIME="$(date -u +%Y-%m-%dT%H:%MZ)"
export GIT_SHA BUILD_TIME
echo "building stl2prism ${GIT_SHA} (${BUILD_TIME})"
docker compose build
docker compose up -d
sleep 3
echo "running: $(curl -s http://localhost:8321/api/version || echo '(not up yet)')"
