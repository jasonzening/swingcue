# Phase 2 — bone-center pilot

Track 2 of the PR-6.0 Phase 1B verdict. Pure 3D-body-fitting exploration
on Modal GPU. **NEVER touches production code** (`python/analyzer.py`,
`python/pose_timeline.py`, `python/main.py`, `src/*`).

## Status

| Sub-PR | Phase | Status |
|---|---|---|
| phase2a | Modal scaffold (Volume + Image defs + setup function) | ✓ in progress |
| phase2b | WHAM-first smoke (real inference on `b3fea3f0`) | pending |
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
    # phase2b adds:
    # └── wham_runner.py       ← WHAM Modal entrypoint + result writer
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

## SMPL research-license credentials (phase2b prerequisite)

Phase 2 needs SMPL/SMPL-X model weights, which live behind a
registration wall (research-license, non-commercial). Jason registers
at:

1. https://smpl.is.tue.mpg.de/  → register, accept license
2. https://smpl-x.is.tue.mpg.de/ → same

Then store credentials as a Modal Secret (NOT in `.env.local`, NOT in
repo):

```powershell
modal secret create smpl-research-creds USERNAME="email@..." PASSWORD="..."
```

The Secret is mounted into `setup_models.py::setup_all_models` only,
never into inference runners.

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
- Do NOT put SMPL credentials in `.env.local` or `python/.env`. Modal
  Secret only.
- Do NOT combine multiple libraries into one Modal Image. Per-library
  isolation is a hard rule (spec §3 + CC review §2).
- Do NOT touch the rtmpose runner here. Phase 2 uses current production
  mediapipe 2D output as input until rtmpose ships in PR-6.1e.

## Next action

phase2a status: scaffolding files in place. Awaiting:

1. Jason runs `modal token new` (browser flow)
2. Jason runs `modal volume create swingcue-pilot-models` (or lets
   `create_if_missing=True` auto-create on first deploy)
3. Jason registers SMPL research license + creates Modal Secret
4. Then phase2b starts: write `runners/wham_runner.py` with the actual
   WHAM Modal entrypoint that runs inference on `b3fea3f0`.
