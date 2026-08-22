# Deploying stl2prism on a NAS — step by step

Runs the web app as one Docker container on a home NAS, reachable from any
browser on your LAN (and optionally over Tailscale). Written and tested on a
**TerraMaster F4-425 Plus (Intel N150, 16 GB, TOS 6)**; the Docker steps are
the same on Synology, QNAP or any Linux box — only the "TOS" notes are
TerraMaster-specific.

Placeholders used throughout:

| Placeholder | Meaning | Example |
|---|---|---|
| `<your-nas>` | the NAS hostname or LAN IP | `nas.local`, `192.0.2.10` |
| `<share-path>` | the shared folder you deploy into | `/Volume1/apps` |
| `<port>` | the LAN port the app listens on | `8321` |

## Step 1 — Check the hardware

- **x86_64 CPU** — required. The scan-repair library (pymeshlab) only ships
  Linux wheels for x86_64; on an ARM NAS the image as written will not build.
  The F4-425 Plus (Intel N150) is x86_64.
- **RAM** — 8 GB minimum, 16 GB comfortable. A 2-million-triangle scan peaks
  at several GB during Poisson repair.

## Step 2 — Prepare the NAS (one-time)

On TOS 6:

1. **App Center → install Docker** ("DockerEngine"). Confirm over SSH:
   `docker compose version`.
2. **App Center → install Git** (optional — the deploy script falls back to a
   throw-away `alpine/git` container if there is no `git` binary).
3. **Control Panel → Shared Folders → create a plain share** for apps
   (no encryption needed; the app stores only uploaded meshes and generated
   STEP files, purged after 24 h). Its path is your `<share-path>`.
4. **Enable SSH** (Control Panel → Terminal & SNMP) and log in as an admin
   user.

On another NAS: install Docker (with Compose v2), create a folder, enable SSH.

## Step 3 — Clone the repository

```bash
cd <share-path>
git clone https://github.com/Crypto69/stl2prism.git
cd stl2prism
chmod -R a+rX .       # TOS shares strip file modes on checkout; harmless elsewhere
mkdir -p data         # job storage, bind-mounted to /data inside the container
```

## Step 4 — Build the image

```bash
docker compose build
```

Builds natively on the NAS (2–3 minutes on the N150; it downloads the ~2 GB
CAD stack the first time). Do **not** copy an image built on an Apple-silicon
Mac unless you built it with `--platform linux/amd64`.

If the build fails with a download timeout in `pip` or `apt-get`, just run it
again — it is a mirror hiccup, not a code problem (pip is set to 300 s / 10
retries; pymeshlab alone is a 106 MB wheel).

## Step 5 — Start it

```bash
docker compose up -d
docker compose ps                          # state: running
curl -s http://127.0.0.1:<port>/ | head    # serves the UI's index.html
```

Open `http://<your-nas>:<port>` in a browser, drop an STL, press Convert.
The report card with green PASS gates is the acceptance test. The badge
top-right shows `version · commit · build time` (also at `/api/version`).

## Step 6 — Update to a new version

```bash
cd <share-path>/stl2prism
./deploy.sh            # pulls main, fixes share file modes, builds, restarts
./deploy.sh --no-pull  # rebuild what is checked out, without pulling
```

A restart kills in-flight conversions; re-run them from the browser.

If the project's git history was ever rewritten upstream (force-push), a
plain pull refuses. Reset once, then deploy as usual:

```bash
git fetch origin && git reset --hard origin/main && ./deploy.sh
```

## Step 7 — Tune it (optional)

All settings live in `docker-compose.yml`; edit, then `docker compose up -d`.

| Setting | Default | Meaning |
|---|---|---|
| `ports` | `8321:8000` | LAN port (change the left side) |
| `STL2PRISM_JOB_TTL` | `86400` | seconds before job directories are deleted |
| `STL2PRISM_CONCURRENCY` | `1` | parallel conversions; keep 1 unless RAM is plentiful |
| `STL2PRISM_MAX_UPLOAD` | 200 MB | maximum upload size |
| `mem_limit` | `12g` | container memory cap; lower it if other services suffer |

`data/` is disposable: `rm -rf data/*` with the container stopped is always
safe. There is no database and nothing to back up.

## Step 8 — HTTPS over Tailscale (optional)

If the NAS is on your tailnet, Tailscale can front the app with HTTPS without
opening anything to the internet:

```bash
tailscale serve --bg --https=8443 http://127.0.0.1:<port>
tailscale serve status
```

The app is then at `https://<your-nas>.<your-tailnet>.ts.net:8443`. Use a
port other than 443 if something else already serves it. The app has no
login: anyone on your LAN or tailnet can use it, which is the intended scope.

## Step 9 — Run the test suite on the NAS (optional)

`tests/` and `samples/` are not baked into the image. Mount them from the
checkout (copy your sample meshes into `samples/` first):

```bash
docker compose run --rm \
  -v "$PWD/tests:/app/tests:ro" -v "$PWD/samples:/app/samples:ro" \
  stl2prism sh -c 'pip install -q pytest httpx && python -m pytest -q -m "not slow"'
# use `-m slow` instead for the scan end-to-ends (takes minutes)
```

This is also where the scan path gets proven: Poisson repair does not work on
macOS arm64, so a large scan converting to a valid solid here is the platform
acceptance test.

## After a reboot

`restart: unless-stopped` brings the container back by itself. Check
`docker compose ps`; if you use the Tailscale front, run
`tailscale serve status` and repeat the `serve` command if its entry is gone.

## TOS 6 notes (TerraMaster)

- **Docker CLI** lives under `/Volume1/@apps/DockerEngine/dockerd/bin`; add
  it to `PATH` in `~/.bashrc`. If the daemon is down, start **DockerEngine**
  in App Center and the containers come back.
- **git** installed from App Center is at `/Volume1/@apps/git/bin/git`, and
  interactive shells may only see it as an alias that scripts cannot use —
  `deploy.sh` reads `.git/HEAD` itself and pulls through an `alpine/git`
  container when needed.
- **Share ACLs strip file modes** on clone/pull: run `chmod -R a+rX .` in the
  repo directory afterwards (`deploy.sh` does this for you).
- **SSH auto-block**: a burst of SSH connections can blacklist your client IP
  (ping works, SSH times out). Unblock under Control Panel → Security, or
  connect from your other address (LAN vs tailnet).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails in `apt-get` / `pip` with a timeout | Mirror hiccup — rerun `./deploy.sh`. |
| Upload rejected with 413 | File exceeds `STL2PRISM_MAX_UPLOAD` — raise it in compose `environment`. |
| Button stuck on "Waiting in queue…" | A previous conversion is still running (they are serialised). Scans take minutes on a small CPU — `docker compose logs -f`. |
| Conversion dies with no result, container fine | The worker was OOM-killed under `mem_limit` — raise the limit or convert a decimated mesh. |
| Errors mentioning pymeshlab / Qt on scan uploads | The image installs `libgl1 libglu1-mesa libxrender1 libxext6 libsm6 libx11-6 fontconfig libcom-err2 libp11-kit0 libgpg-error0`; `docker compose exec stl2prism python -c "import pymeshlab"` should print nothing. Rebuild with `--no-cache` if not. |
| UI loads but every API call 404s | Stale image — `docker compose build --no-cache`. |
| Port already in use | Change the left side of `ports:` in compose (and the `tailscale serve` target). |
