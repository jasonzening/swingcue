# SwingPhaseDetector — 検出報告

生成日時: 2026-06-03  
入力データ: RTMPose-x 関節点 JSON (test-faceon / test-dwontheline)

---

## 1. 検出結果まとめ

| キーフレーム | 側面 (downtheline) | 正面 (faceon) |
|---|---|---|
| address  | frame **10** (333ms)  conf=0.53 | frame **60** (2143ms) conf=0.56 |
| top      | frame **31** (1033ms) conf=1.00 | frame **100** (3572ms) conf=1.00 |
| impact   | frame **47** (1567ms) conf=0.77 | frame **111** (3965ms) conf=0.87 |
| finish   | frame **68** (2266ms) conf=0.57 | frame **136** (4857ms) conf=0.00* |

\* 正面動画はビデオ末尾まで完全に安定しないため fallback 使用 → conf=0.0 (penalised)

---

## 2. アルゴリズム (SwingPhaseDetector v3)

### 共通パイプライン
1. **前処理**: 左右手首の信頼度加重平均で中点 (x,y) を算出。信頼度 < 0.4 のフレームは線形補間。
2. **平滑化**: Savitzky-Golay フィルタ (窓 = fps × 200ms, poly=3)
3. **速度**: 平滑化座標の差分 → √(dx²+dy²) をさらに SG 平滑化
4. **垂直速度 vy**: np.gradient(ys_smooth)

### address
- 静止閾値 = max(spd の第 20 百分位 × 3.0,  1.0)
- ビデオ前半 50% を走査。閾値以下の最後のフレームを選択。

### top
- 探索範囲 address ~ ビデオ 82% (82% により prominence 右底辺確保)
- find_peaks(-ys, prominence≥30px, distance≥fps×0.25) で最初の山
- prominence=0 のフォールバック: 領域端からの高さで推定

### impact (三条件ルール)
- 速度ピーク (`spd_peak`) を下スイング [top ~ 80%] で検出
- 探索窓 = [spd_peak ± 0.4s]  ← **両視角に対応する核心パラメータ**
  - 側面: impact は spd_peak より少し *後* (手首が球位置に戻る)
  - 正面: impact は spd_peak より少し *前* (手首が球をスイープする)
- 窓内で address 錨点 (xs_addr, ys_addr) への XY ユークリッド距離最小フレームを選択
- **サニティチェック**: (addr_y - impact_y) > torso_h × 0.35 → conf × 0.3

### finish
- 随振後の高点 (ft_top) を impact 後の ys 局所最小で検出
- ft_top 以降で settle_win (≥fps×0.35) 連続フレームが settle_thr 以下 → finish
- ビデオ末尾で未安定の場合: ft_top 以降の速度最小フレームを fallback 選択 (conf × 0.5)

---

## 3. 両視角のパラメータ比較

| パラメータ | 側面 | 正面 | 差異の理由 |
|---|---|---|---|
| address 静止区間 | 帧 0–10 (短い) | 帧 0–60 (長い約2秒) | 撮影前の静止時間の差 |
| impact 方向 | spd_peak **後** (+4fr) | spd_peak **前** (-1fr) | 正面は手首が横断, 側面は手首が折り返す |
| top prominence | 166px (強い) | 285px (より強い) | 正面のほうが垂直変位が大きく検出容易 |
| finish 安定性 | 安定確認 (conf=0.57) | ビデオ末まで未安定 (conf=0.00) | 正面動画がフォロースルー終端を含まない |
| impact dist_to_addr | 47px (torso の 34%) | 30px (torso の 19%) | 正面では両手首が球周辺でより近い |

**共通パラメータ (調整不要)**:
- sg_window_ms=200, static_percentile=20, static_multiplier=3.0
- top_prominence_px=30, impact_window_s=0.40, impact_sanity_frac=0.35
- finish_settle_s=0.35

---

## 4. 信頼度スコア定義

| キー | 算出式 |
|---|---|
| address | clip(1 - speed_at_addr / (2×static_thr), 0, 1) |
| top | clip(top_prominence / 150px, 0, 1) |
| impact | clip(1 - dist_to_addr / (1.5×torso_h), 0, 1) × sanity_flag |
| finish | clip(1 - settle_mean_speed / settle_thr, 0, 1) × (0.5 if fallback) |

---

## 5. 既知の限界 (検出失敗リスク)

### 高リスク
- **複数人が映る動画**: 最初の person_id=0 のみ使用。gallery など非検出
- **カメラが動く動画**: 背景動作が速度信号に混入。手腕追跡が乱れる
- **ハーフスイング / チップショット**: top の垂直変位が 30px 未満 → top prominence 閾値未満で失敗
- **ビデオがアドレス直前から始まる**: 静止区間が 5 フレーム未満 → address が推定値にずれる

### 中リスク
- **高速ビデオ (120fps+)**: SG 窓の計算は fps 基準なので自動スケール。ただし未検証
- **低解像度・ブレが強いビデオ**: 手首信頼度が 0.4 を大量に下回る → 補間区間が多くなり信号劣化
- **左利きゴルファー**: 今回のロジックは利き手依存なし (両手首の中点を使用) → 理論上対応可能だが未検証
- **ダウンスイングで手首が address より大幅に前に出る正面動画**: dist_to_addr が増大 → impact conf 低下

### 低リスク (対策済み)
- 随振後高点が 2 回ある (上杆 top + 随振 top): find_peaks の first-peak 選択で対処
- 正面と側面で impact タイミングがずれる: ±0.4s 窓で両方カバー
- ビデオ末尾まで安定しない: fallback で速度最小フレームを選択 (conf 低下で明示)

---

## 6. 出力ファイル一覧

```
keyframes/
├── detect_keyframes.py              # SwingPhaseDetector モジュール (CLI + library)
├── test-dwontheline_keyframes.json  # 側面検出結果 + confidence
├── test-dwontheline_speed_curve.png # 速度曲線 3 面グラフ
├── test-dwontheline_verify.mp4      # 検収動画 (4 定格フレーム各 1 秒)
├── test-faceon_keyframes.json
├── test-faceon_speed_curve.png
└── test-faceon_verify.mp4
```

### ライブラリとしての使い方
```python
from keyframes.detect_keyframes import SwingPhaseDetector

det = SwingPhaseDetector()
result = det.detect("output/rtmpose/test-faceon_keypoints.json")

kf   = result["keyframes"]    # {"address":60, "top":100, "impact":111, "finish":136}
conf = result["confidence"]   # {"address":0.56, "top":1.0, "impact":0.87, "finish":0.0}
```
