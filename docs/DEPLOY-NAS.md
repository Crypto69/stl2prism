# NAS deployment runbook — stl2prism on x86_64 NAS (TOS 6)

The STL→STEP converter runs on the NAS (`nas-host` on the tailnet) as a
single container:

- app code + job data on the **`apps` shared folder on Volume 5**
  (unencrypted — job files are uploaded STLs and generated STEPs, nothing
  sensitive, and everything is auto-purged after 24 h)
- one container serving both the API and the web UI on port **8321**
  (portfolio owns 8090 and `https=443`; this stack stays clear of both)
- **no database, no secrets, no backups** — if the data directory vanished
  entirely, the only loss would be not-yet-downloaded conversions
- conversions run **one at a time** (`STL2PRISM_CONCURRENCY=1`) with a
  `mem_limit` so a huge scan repair can't starve the portfolio stack

## One-time TOS prerequisites

1. **Shared folder** — `apps` on **Volume 5** (already created; plain,
   no encryption needed).
2. **Docker** — already installed for portfolio (DockerEngine in App Center).
   Confirm over SSH: `docker compose version`.
3. **Tailscale** — only needed for the optional HTTPS front; already
   installed for portfolio.

## First deploy

```bash
# on the NAS (SSH)
cd <share-path>
git clone https://github.com/Crypto69/stl2prism.git
# (public repo, https clone — no deploy key needed, unlike portfolio)

cd stl2prism
chmod -R a+rX .                    # share ACLs strip file modes on checkout
mkdir -p data                      # job storage, bind-mounted to /data
```

Build natively on the NAS (x86_64) — never copy images built on an ARM Mac
without `--platform linux/amd64`:

```bash
docker compose build     # pulls the ~2 GB CAD stack; ~2.5 min measured on the NAS CPU
docker compose up -d
```

That's the whole deploy. Check it:

```bash
docker compose ps                      # state: running
curl -s http://127.0.0.1:8321/ | head  # serves the UI's index.html
```

Then from a browser on the LAN: `http://nas-host:8321` (or the NAS IP).
Drop a sample STL, convert, download — the report card appearing with PASS
gates is the acceptance test.

### Optional HTTPS front (tailnet access)

Portfolio already occupies `https=443`, so serve this app on **8443**:

```bash
tailscale serve --bg --https=8443 http://127.0.0.1:8321
tailscale serve status               # confirm both 443 and 8443 entries exist
```

App URL becomes `https://nas-host.your-tailnet.ts.net:8443`. The app has
no login — anyone on the LAN or tailnet can use it, which is the intended
scope.

### Platform acceptance test (worth doing once)

pymeshlab's Poisson scan repair is broken on macOS arm64 but works on Linux
x86_64 — this box is the first place the full scan path can be proven
end-to-end. `tests/` and `samples/` are not baked into the image
(.dockerignore), so mount them from the checkout — with `samples/` copied
into the repo directory on the NAS first:

```bash
docker compose run --rm \
  -v "$PWD/tests:/app/tests:ro" -v "$PWD/samples:/app/samples:ro" \
  stl2prism sh -c 'pip install -q pytest && python -m pytest -q'            # fast tests
# same command with `python -m pytest -q -m slow` for the scan end-to-ends (long)
```

Or simply upload `Mesh_90p.stl` through the UI — a completed faceted (or
forced-prismatic) result proves Poisson works here.

## Configuration knobs

All set in `docker-compose.yml` (edit + `docker compose up -d` to apply):

| Setting | Default | Meaning |
|---|---|---|
| `ports` | `8321:8000` | LAN port |
| `STL2PRISM_JOB_TTL` | `86400` | seconds before job dirs are auto-deleted |
| `STL2PRISM_CONCURRENCY` | `1` | parallel conversions — leave at 1 on 16 GB shared with portfolio |
| `STL2PRISM_MAX_UPLOAD` | 200 MB (app default) | max STL upload size |
| `mem_limit` | `12g` | drop to `8g` if the portfolio stack ever feels starved |

The `data/` directory is disposable: `rm -rf data/*` while the container is
stopped is always safe.

## TOS gotchas (inherited from the portfolio migration)

Same box, same quirks — full detail in the portfolio repo's
`docs/DEPLOY-NAS.md`:

- **Docker CLI location**: binaries under
  `<app-path>/DockerEngine/dockerd/bin` (already on PATH via `~/.bashrc`).
  If the daemon is down, start **DockerEngine** in App Center — containers
  come back.
- **Git**: TOS now ships git (2.54 as of this deploy, 2026-08), so the
  portfolio runbook's `alpine/git` container workaround is no longer needed.
  If a TOS update ever drops it again:
  `docker run --rm -v "$PWD":/git alpine/git <clone|pull ...>`. This repo is
  public https, so none of portfolio's deploy-key/SSH-alias setup applies.
- **Share ACLs strip file modes**: after every clone/pull run
  `chmod -R a+rX .` in the repo dir. The image itself is immune (everything
  is COPY'd at build time); only the `data/` bind mount and build context
  matter, and the app runs as root in-container so ownership is a non-issue.
- **TOS SSH auto-block**: bursts of SSH connections can blacklist your source
  IP (ping works, SSH times out). Unblock in Control Panel → Security, or
  come in via the other address (LAN vs tailnet).

## Updating the app

```bash
cd <share-path>/stl2prism
git pull
chmod -R a+rX .                                # ACLs strip modes on pull
docker compose build && docker compose up -d
```

In-flight conversions are killed by the restart (they're subprocesses of the
container); re-run them from the browser — uploads under the TTL are gone
only if the job dir was purged, but re-uploading is cheap.

## After a NAS reboot

Nothing to unlock (the share isn't encrypted) and `restart: unless-stopped`
brings the container back on its own. Only two checks:

1. `docker compose ps` in `<share-path>/stl2prism` — running.
2. If using the HTTPS front: `tailscale serve status` — re-run the
   `tailscale serve --bg --https=8443 …` command if the 8443 entry is gone
   (tmnascommunity service not auto-starting is the usual culprit).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `docker compose build` fails in `apt-get` | Transient mirror issue — retry; the package list (`libgl1 libglu1-mesa libxrender1 libxext6 libsm6 libx11-6 fontconfig`) is known-good on `python:3.12-slim` |
| Upload rejected with 413 | STL exceeds `STL2PRISM_MAX_UPLOAD` — raise it in compose `environment` |
| Button stuck on "Waiting in queue…" | A previous conversion is still running (they're serialized). Big scans take minutes on the N-series CPU — watch `docker compose logs -f` |
| Conversion dies with no result, container fine | Worker was OOM-killed inside `mem_limit` — raise the limit or convert a decimated mesh |
| Scan upload errors mentioning pymeshlab | The image builds with the `[scan]` extra, so this means a broken build — rebuild; `docker compose exec stl2prism python -c "import pymeshlab"` should be silent |
| UI loads but every API call 404s | Stale image where `frontend/dist` was baked without the backend — rebuild with `docker compose build --no-cache` |
| Port 8321 already in use | Another stack claimed it — change the left side of `ports:` in compose and the tailscale serve target |
