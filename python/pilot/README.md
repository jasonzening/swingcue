# Phase 2 — bone-center pilot

Track 2 of the PR-6.0 Phase 1B verdict. Pure 3D-body-fitting exploration
on Modal GPU. **NEVER touches production code** (`python/analyzer.py`,
`python/pose_timeline.py`, `python/main.py`, `src/*`).

## Status

| Sub-PR | Phase | Status |
|---|---|---|
| phase2a | Modal scaffold (Volume + Image defs + setup function) | ✓ done (`25f0fae`) |
| phase2b | WHAM-first smoke (real inference on `b3fea3f0`) | code in place; awaiting Jason prep |
| phase2c | Expand to Human3R / SMPLest-X / EasyMocap / SMPLify-X | pending |

See `docs/files/PHASE_2_BONE_CENTER_PILOT_SPEC_v2.md` for the
full spec.

## File map

```
python/pilot/
├── README.md                  ← this file
├── __init__.py
├── requirements_pilot.txt     ← local dev deps (Modal client only)
├── modal_app.py               ← Modal App + Volume + per-library Images
├── setup_models.py            ← one-time Volume populator
└── runners/
    ├── __init__.py
    └── _base.py               ← shared PilotRunResult schema (20 joints)
    └── wham_runner.py       ← WHAM Modal entrypoint + result writer (phase2b)
    # phase2c adds:
    # └── human3r_runner.py    ← Human3R Modal entrypoint
    # └── smplest_x_runner.py  ← SMPLest-X Modal entrypoint
    # └── easymocap_runner.py  ← EasyMocap Modal entrypoint
    # └── smplify_x_runner.py  ← SMPLify-X Modal entrypoint
```

## Bootstrap (one-time, Jason runs locally)

```powershell
# 1. Fresh venv — must NOT mix with .venv-benchmark or production
python3.11 -m venv .venv-pilot
.\.venv-pilot\Scripts\Activate.ps1

# 2. Install Modal client
pip install -r python/pilot/requirements_pilot.txt

# 3. Authenticate with Modal (browser flow; writes ~/.modal.toml)
modal token new

# 4. Verify scaffold parses cleanly (no Modal cost yet)
python -m python.pilot.modal_app
#   → prints APP_NAME, VOLUME_NAME, defined images, etc.
```

## SMPL family weights — `modal volume put` (phase2b prerequisite)

Phase 2 needs SMPL / SMPL-H / SMPL-X model weights, which live behind a
registration wall (research-license, non-commercial). **Transport into
Modal: `modal volume put`** — no scraping, no credentials-in-Modal-
Secret. Decision rationale: license-clean + site form structure can
change without notice.

Jason workflow (one-time):

1. Register at:
   - https://smpl.is.tue.mpg.de/  → SMPL (Neutral / Male / Female)
   - https://mano.is.tue.mpg.de/  → SMPL-H (WHAM's actual requirement)
   - https://smpl-x.is.tue.mpg.de/ → SMPL-X (phase2c nice-to-have)

2. Download + unzip locally into this exact layout (the names matter —
   `setup_models._verify_body_models` checks them):

   ```
   ./local-body-models/
   ├── smpl/
   │   ├── SMPL_NEUTRAL.pkl
   │   ├── SMPL_MALE.pkl
   │   └── SMPL_FEMALE.pkl
   ├── smplh/
   │   └── SMPLH_NEUTRAL.npz
   └── smplx/                       (phase2c)
       └── SMPLX_NEUTRAL.npz
   ```

3. Upload to the Modal Volume in one shot:

   ```powershell
   modal volume put swingcue-pilot-models ./local-body-models /models/body_models
   ```

After upload, `setup_models.py::setup_all_models` cross-checks the
layout and warns if anything is missing/too-small. Volume is persistent
across Modal deploys — this upload is one-time.

(The earlier draft of this doc mentioned a `smpl-research-creds` Modal
Secret. That was an alternative scraping-based design; the `volume put`
path replaces it. If you've already created the Secret, it's now
unused — harmless to leave or delete.)

## Cost model (per spec §3)

| Phase | Cost |
|---|---|
| phase2a — scaffold + Volume creation | $0 |
| phase2b — WHAM smoke (1 video × A10G ~30s × few runs) | $0-30 (mostly free tier) |
| phase2c — multi-library expand + retries | $50-150 (engineering cost dominates inference) |
| Production target | ≤ $0.05 per analysis (TBD from phase2c logs) |

Volume storage + Modal Image builds are essentially free; real $$ is
GPU runtime.

## Why isolated from production

Per Verdict v2 §9 anti-pattern guard: "Do NOT delay Track 2 until after
Track 1." Pilot lives in its own Modal app, its own Modal Volume, its
own dependency tree. The winning library identified by phase2c will
move into production via a **separate** PR-7 spec — pilot code itself
is NOT for production deploy.

## Anti-patterns (do NOT)

- Do NOT import `python.pilot.*` from `python/analyzer.py`,
  `python/main.py`, `python/pose_timeline.py`, or any other production
  module. Pilot must remain orphaned from the production import graph.
- Do NOT scrape smpl.is.tue.mpg.de or smpl-x.is.tue.mpg.de. SMPL
  transport is `modal volume put` from your local download only.
- Do NOT combine multiple libraries into one Modal Image. Per-library
  isolation is a hard rule (spec §3 + CC review §2).
- Do NOT touch the rtmpose runner here. Phase 2 uses current production
  mediapipe 2D output as input until rtmpose ships in PR-6.1e.

## Next action — phase2b WHAM smoke

phase2b code is in place. Sequenced with Jason's one-time prep:

**Jason — ~10 min prep:**
1. `python3.11 -m venv .venv-pilot` + activate + `pip install -r python/pilot/requirements_pilot.txt`
2. `modal token new` (browser flow)
3. SMPL family research-license downloads + `modal volume put` per the
   layout above
4. Signal CC: "ready"

**CC drives autonomously after "ready":**
1. `modal volume create swingcue-pilot-models` (or auto-created by
   `create_if_missing=True` on first deploy)
2. `modal run python/pilot/setup_models.py::setup_all_models` —
   downloads 6 WHAM weight files (~1.5 GB) to /models/wham via gdown
   and cross-checks Jason's SMPL upload
3. → GO/NO-GO gate: Jason ACK "WHAM weights downloaded" (next step
   spends GPU money)
4. `modal run python/pilot/runners/wham_runner.py::run_wham_local
   --video-id b3fea3f0-… --video-url <signed URL>` — A10G GPU
   inference, ~30 s wall clock, ~$0.01 per run
5. Local result fetch → `python/pilot/output/wham/b3fea3f0-…/joint_centers_3d.json`
6. 2D back-projection overlay render (local, no Modal cost) — phase2b
   second commit

After step 6 → comparison.mp4 → Jason watches → decide on phase2c
expansion or pivot.
