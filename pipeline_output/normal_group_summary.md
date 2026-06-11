# Normal Group — Screening & Ingest Summary

**Date**: 2026-06-10  
**Source**: OneDrive/Documents/stodownload*.mp4 (11 files) + test-dwontheline.mp4  
**GT iron rule**: no fault labels in this document.

## Screening Table

| File | Res | Frames | FPS | Angle | ShRatio | Swings | Complete | PipelineReady |
|---|---|---|---|---|---|---|---|---|
| stodownload(20).mp4 | 1080x1920 | 492 | 30 | no_data | None | 0 | True | no |
| stodownload(24).mp4 | 1080x1920 | 470 | 30 | face-on | 0.556 | 3 | True | no |
| stodownload(28).mp4 | 1080x1920 | 318 | 30 | no_data | None | 2 | True | no |
| stodownload(32).mp4 | 1080x1920 | 291 | 30 | no_data | None | 0 | True | no |
| stodownload(43).mp4 | 1080x1920 | 309 | 30 | no_data | None | 2 | True | no |
| stodownload(45).mp4 | 1080x1920 | 395 | 30 | no_data | None | 4 | True | no |
| stodownload(49).mp4 | 1080x1920 | 446 | 30 | no_data | None | 5 | True | no |
| stodownload(53).mp4 | 1080x1920 | 256 | 30 | DTL | 0.186 | 1 | True | YES |
| stodownload(60).mp4 | 1080x1440 | 258 | 30 | no_data | None | 0 | True | no |
| stodownload(64).mp4 | 1080x1920 | 355 | 30 | no_data | None | 1 | True | no |
| stodownload(91).mp4 | 1080x1920 | 401 | 30 | face-on | 0.774 | 2 | True | no |
| test-dwontheline.mp4 | 720x1280 | 120 | 30 | DTL | 0.07 | 1 | True | YES |

## DTL Pipeline Results — Per-Swing

| File | Swings | addr | top | impact | impact_conf | hip_win_peak (fr/%) | spine_win_peak (fr/°) | Diagnosis (verbatim) |
|---|---|---|---|---|---|---|---|---|
| stodownload(53).mp4 | 1 | 116 | 208 | 221 | 0.960 | fr216 / -103.4% | fr216 / -13.29° | none / none |
| test-dwontheline.mp4 | 1 | 10 | 31 | 46 | 0.900 | fr34 / 20.4% | fr34 / -13.43° | none / none |

## Normal Group Distribution Statistics

hip_disp window peak (n=2 < 5, no stats): ['-103.4%', '20.4%']
spine_delta window peak (n=2 < 5, no stats): ['-13.29°', '-13.43°']

## False-Alarm Count (申报正常段中引擎有输出的)

申报正常杆数: 2  
引擎输出非 none 的杆数: 0  
(是否真误报由人工看图裁决，本文件不含判断)


## Face-On Videos (registered, not processed — features not online)

| File | Frames | angle | sh_ratio |
|---|---|---|---|
| stodownload(24).mp4 | 470 | face-on | 0.556 |
| stodownload(91).mp4 | 401 | face-on | 0.774 |