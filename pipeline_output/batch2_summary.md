# Batch2 Face-On Pipeline Results

**GT iron rule**: no fault labels. 'ok'/'wrong' are file names only.
**Date**: 2026-06-11

## Layer 0 Gate Summary

| Stem | Verdict | Angle | Reason (brief) |
|---|---|---|---|
| dtl-ok-1 | needs_human | mixed | 4/5 face-on, 1 DTL (follow-through). Prefix dtl conflicts with predominantly fac... |
| dtl-ok-2 | needs_human | mixed | 3 face-on, 2 DTL. Prefix dtl conflicts with mixed VLM result. Paused per angle-c... |
| dtl-wrong-1 | needs_human | face-on | 5/5 face-on. Prefix dtl-wrong expected DTL; VLM consistently face-on → angle con... |
| dtl-wrong-2 | needs_human | mixed | 3 face-on, 2 DTL. Prefix dtl-wrong with mixed VLM result → angle conflict → paus... |
| dtl-wrong-3 | needs_human | face-on | 5/5 face-on. Prefix dtl-wrong expected DTL; VLM consistently face-on → angle con... |
| fo-ok-1 | PASS | face-on | 5/5 q1=YES,q2=1,q4=YES, consistently face-on. Matches prefix fo-ok. |
| fo-ok-2 | PASS | face-on | 5/5 q1=YES,q2=1,q4=YES, consistently face-on. Matches prefix fo-ok. |
| fo-wrong-1 | PASS | face-on | 5/5 pass. Consistently face-on. Note: sampled frames all at address/setup; may c... |
| fo-wrong-2 | PASS | face-on | 4/5 hard pass (fr127 feet marginally cropped). Consistently face-on. Meets >=4 t... |
| fo-wrong-3 | needs_human | mixed | 3 face-on, 2 DTL. Prefix fo expected face-on; fr118/fr157 read as DTL → angle co... |
| fo-wrong-4 | needs_human | mixed | 3 face-on, 2 DTL. Prefix fo expected face-on; fr125/fr167 read as DTL → angle co... |

## Face-On Measurements (4 PASS videos)

| Stem | addr | top | impact | ic | head_lat+ (fr/%) | head_lat- (fr/%) | head_vert (fr/%) | elbow_min (CW_win / fr / deg) |
|---|---|---|---|---|---|---|---|---|
| fo-ok-1 | 42 | 76 | 88 | 0.914 | fr53/+2.3% | fr82/-15.7% | fr79/-6.2% | fr88-fr91/fr91/166.2deg |
| fo-ok-2 | 33 | 65 | 77 | 0.929 | fr36/+0.5% | fr75/-70.9% | fr75/+72.5% | fr77-fr79/fr79/167.2deg |
| fo-wrong-1 | 180 | 296 | 331 | 0.929 | fr180/+0.0% | fr322/-27.0% | fr317/+7.2% | fr331-fr334/fr331/160.7deg |
| fo-wrong-2 | 78 | 115 | 127 | 0.929 | fr101/+1.2% | fr121/-18.8% | fr118/+0.7% | fr127-fr130/fr130/151.3deg |

## needs_human (7 videos — paused per angle-conflict rule)

VLM angle conflicts with filename prefix. Cannot proceed without human confirmation.

| Stem | Prefix angle | VLM result | Reason |
|---|---|---|---|
| dtl-ok-1 | DTL | mixed | 4/5 face-on, 1 DTL (follow-through). Prefix dtl conflicts with predominantly fac... |
| dtl-ok-2 | DTL | mixed | 3 face-on, 2 DTL. Prefix dtl conflicts with mixed VLM result. Paused per angle-c... |
| dtl-wrong-1 | DTL | face-on | 5/5 face-on. Prefix dtl-wrong expected DTL; VLM consistently face-on → angle con... |
| dtl-wrong-2 | DTL | mixed | 3 face-on, 2 DTL. Prefix dtl-wrong with mixed VLM result → angle conflict → paus... |
| dtl-wrong-3 | DTL | face-on | 5/5 face-on. Prefix dtl-wrong expected DTL; VLM consistently face-on → angle con... |
| fo-wrong-3 | face-on | mixed | 3 face-on, 2 DTL. Prefix fo expected face-on; fr118/fr157 read as DTL → angle co... |
| fo-wrong-4 | face-on | mixed | 3 face-on, 2 DTL. Prefix fo expected face-on; fr125/fr167 read as DTL → angle co... |

## DTL processing
Not run — all 5 DTL-prefix videos are needs_human.
