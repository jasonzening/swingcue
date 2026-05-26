# PR-8d.1 visual palette (LOCKED 2026-05-26)

Locked color palette for PR-8d.1 frontend (the WHAM-driven coaching
overlay). Implementation lands in PR-8d; this file just freezes the
values so PR-8d.0 + PR-8d.1 + PR-8d.2 split cannot drift on color choices.

## Anatomy

| Semantic | Hex      | Notes |
|---|---|---|
| Left side  | `#00C2FF` | Cyan. Anatomical left (the golfer's actual left side, NOT image-left). Applies to left_shoulder / left_elbow / left_wrist / left_hip / left_knee / left_ankle. |
| Right side | `#FFB000` | Amber. Anatomical right. |
| Centerline | `#FFFFFF` | White. Spine / neck / pelvis / head / head_crown — the midsagittal joints. |

Chirality reminder (PR-7a.2): WHAM joint NAMES (`left_*` / `right_*`)
follow image-orientation convention AFTER our upper-body arm-chain
swap. For face-on cameras the relabel makes "left" mean image-left
(== anatomical right for the golfer). PR-8d.1 must decide once which
convention it shows the user and document it; the palette colors are
neutral on this — they just tag each side consistently.

## Status / feedback

| Semantic | Hex      | Notes |
|---|---|---|
| Error / failure / out-of-bounds   | red (TBD exact hex, e.g. `#FF3030`) | Reserved. Never use red for joint anchors or coaching cues — red is exclusively for error state UX (e.g., `wham_status='failed'`, off-camera body parts). |
| Ideal / target / "you nailed it"  | green (TBD exact hex, e.g. `#00E676`) | Reserved. Never use green for live anchors — green is "this is correct" feedback only. |

Concrete red/green hex values to be decided when PR-8d.1 actually
renders status UX. The semantic reservation is the lock; the exact
shade can iterate visually.

## Why these specifics

- **`#00C2FF` (cyan) + `#FFB000` (amber)** — complementary-ish across
  the color wheel without being too garish; both legible on green grass
  + tree backgrounds (the typical golf-video background). Distinguishable
  for the most common color-vision deficiencies (deuteranopia /
  protanopia) where pure red-vs-green fails.
- **`#FFFFFF` (white)** for centerline — neutral, doesn't bias the
  left/right scheme; high contrast for spine + crown emphasis.
- **Red/green reserved** — frees future cuing without re-tinting the
  anatomy palette. If we later want "good rotation = green hip arrow",
  the colors are pre-allocated.

## NOT in scope for PR-8d.1

- Per-phase color modulation (setup/top/impact different anchors)
- Confidence-tied opacity gradients
- Brand color override (no SwingCue-brand-magenta on body anchors)
- Disc / surface overlays (PR-5/6 territory; superseded by PR-8d.1)

If any of those need to land later, they get their own palette extension
PR — they do NOT clobber these four anchor semantics.
