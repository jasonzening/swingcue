# GHOST-003 前置閘門 T0：授權核驗報告

**核驗時間**: 2026-07-06  
**核驗原則**: 只核授權，不跑模型、不改環境、不裝依賴  
**數據來源**: 本機 curl 直接抓取原始文件，逐字引用，不轉述

---

## 1. SAM 3D Body (facebookresearch/sam-3d-body)

### 1.1 Repo 固化

```
GitHub URL:  https://github.com/facebookresearch/sam-3d-body
主分支 HEAD: b5c765a0d89d789985e186d396315e7590887b94  (refs/heads/main)
核驗時間:    2026-07-06
```

### 1.2 LICENSE 文件全文（逐字）

```
SAM License
Last Updated: November 19, 2025

"Agreement" means the terms and conditions for use, reproduction, distribution
and modification of the SAM Materials set forth herein.

"SAM Materials" means, collectively, Documentation and the models, software and
algorithms, including machine-learning model code, trained model weights,
inference-enabling code, training-enabling code, fine-tuning enabling code, and
other elements of the foregoing distributed by Meta and made available under
this Agreement.

...

1. License Rights and Redistribution.

a. Grant of Rights. You are granted a non-exclusive, worldwide, non-transferable
and royalty-free limited license under Meta's intellectual property or other
rights owned by Meta embodied in the SAM Materials to use, reproduce, distribute,
copy, create derivative works of, and make modifications to the SAM Materials.

i. Grant of Patent License. Subject to the terms and conditions of this License,
you are granted a perpetual, worldwide, non-exclusive, no-charge, royalty-free,
irrevocable (except as stated in this section) patent license to make, have made,
use, offer to sell, sell, import, and otherwise transfer the Work...

b. Redistribution and Use.

i. Distribution of SAM Materials, and any derivative works thereof, are subject
to the terms of this Agreement.

ii. If you submit for publication the results of research you perform on, using,
or otherwise in connection with SAM Materials, you must acknowledge the use of
SAM Materials in your publication.

iii. Your use of the SAM Materials must comply with applicable laws and
regulations, including Trade Control Laws and applicable privacy and data
protection laws.

iv. Your use of the SAM Materials will not involve or encourage others to reverse
engineer, decompile or discover the underlying components of the SAM Materials.

v. You are not the target of Trade Controls and your use of SAM Materials must
comply with Trade Controls. You agree not to use, or permit others to use, SAM
Materials for any activities subject to the International Traffic in Arms
Regulations (ITAR) or end uses prohibited by Trade Controls, including those
related to military or warfare purposes, nuclear industries or applications,
espionage, or the development or use of guns or illegal weapons.
```

> **注**：LICENSE 無任何 "non-commercial" 或 "research-only" 字樣。無商業使用限制條款。
> 含貿易管制條款（Trade Controls §1.b.v）及出版致謝要求（§1.b.ii）。

### 1.3 README License 段落（逐字）

```
## License

The SAM 3D Body model checkpoints and code are licensed under [SAM License](./LICENSE).
```

### 1.4 Checkpoint HuggingFace Model Card（逐字）

**Repo**: `facebook/sam-3d-body-dinov3` 及 `facebook/sam-3d-body-vith`

```
HF API cardData:
{
  "license": "other",
  "license_name": "sam-license",
  "license_link": "https://huggingface.co/facebook/sam-3d-body-dinov3/blob/main/LICENSE",
  "extra_gated_fields": {
    "First Name": "text",
    "Last Name": "text",
    "Date of birth": "date_picker",
    "Country": "country",
    "Affiliation": "text",
    "Job title": { "type": "select", ... },
    "geo": "ip_location",
    "By clicking Submit below I accept the terms of the license...": "checkbox"
  },
  "extra_gated_button_content": "Submit"
}

gated: "manual"  (需人工申請審批)
```

Checkpoint repo 的 LICENSE 文件內容 = 代碼 repo 相同的 "SAM License Last Updated: November 19, 2025"（curl 逐字確認）。

### 1.5 Body Model 層核驗（代碼實際 import，非文檔）

`sam_3d_body/models/heads/mhr_head.py` 實際代碼：

```python
try:
    if MOMENTUM_ENABLED:
        from mhr.mhr import MHR        # MHR（Meta Human Representation）獨立包，可選
        MOMENTUM_ENABLED = True
    else:
        raise ImportError
except:
    MOMENTUM_ENABLED = False
```

`sam_3d_body/build_models.py` 實際代碼：

```python
def load_sam_3d_body(checkpoint_path, device, mhr_path=""):
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
    ...
```

`demo.py` 調用：

```python
mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
model, model_cfg = load_sam_3d_body(args.checkpoint_path, device=device, mhr_path=mhr_path)
```

> **確認**：SAM 3D Body **不使用 SMPL / SMPL-X**。使用 **MHR（Meta Human Representation）** 拓撲。  
> MHR 作為可選依賴包 (`from mhr.mhr import MHR`) 打包在 checkpoint asset 中（`mhr_model.pt`），  
> 受 SAM License 整體覆蓋（checkpoint bundle 一體）。

### 1.6 五層分層判定

| 層 | 授權 | 關鍵原文出處 |
|----|------|-------------|
| Code license | **SAM License (Nov 19, 2025)** | `LICENSE` 文件逐字，無 NC 限制 |
| Checkpoint/weights license | **SAM License** | HF `license_name: "sam-license"`；checkpoint repo LICENSE 文件 |
| Body model/rig license | **SAM License**（MHR 打包在 checkpoint bundle 內） | `build_models.py` `mhr_path` 參數；`mhr_head.py` optional import |
| Output mesh/topology license | **SAM License** 覆蓋（輸出為 MHR 拓撲頂點） | LICENSE §5.a：衍生作品歸用戶所有 |
| Training data | NOTICE 未找到；論文引用數據集各有授權，不影響推理使用 | — |

### 1.7 結論標籤

**`PRODUCT_CANDIDATE_CUSTOM_LICENSE`**

- 無 NC / research-only 明文限制
- 屬 Meta 自定義 SAM License，非標準 Apache/MIT/BSD
- 含貿易管制 §1.b.v（涉軍/核/ITAR 禁用）
- 含出版致謝要求 §1.b.ii
- Checkpoint gated（需申請 HF access）
- **進產品前需 legal review 確認 SAM License 對 SwingCue 業務的貿易管制合規性**

---

## 2. Anny Body Model (naver/anny)

### 2.1 Repo 固化

```
GitHub URL:  https://github.com/naver/anny
主分支 HEAD: e53d4b8a6ce4e8b5f257c4cee92cffcfb3d3efb9  (refs/heads/main)
核驗時間:    2026-07-06
```

### 2.2 LICENSE 文件（前 30 行逐字）

```
Anny, Copyright (C) 2025 NAVER Corporation. All Rights Reserved.

The files within this subdirectory are licensed under the Apache License, 
Version 2.0 (the "License"); you may not use these files except in compliance 
with the License.

A copy of the terms of the license are reproduced below. You may also obtain 
a copy of the License at:

       http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

                         Apache License
                         Version 2.0, January 2004
                      http://www.apache.org/licenses/
```

### 2.3 README License 段落（逐字）

```
## License

The code of Anny, Copyright (c) 2025 NAVER Corp., is licensed under the 
Apache License, Version 2.0 (see LICENSE).

**data/mpfb2**: Anny relies on MakeHuman assets adapted from MPFB2 that are 
licensed under the CC0 1.0 Universal License.

**data/soma**: Anny provide a "soma" topology adapted from SOMA-X which is 
licenced under the Apache 2.0 license.

**smplx**: A "smplx" topology can be downloaded for non-commercial use only, 
allowing interoperability with SMPL-X. See LICENSE.txt and NOTICE.txt files 
in http://download.europe.naverlabs.com/humans/Anny/noncommercial.zip for 
more information.
```

### 2.4 Checkpoint / HF Model Card

Anny 是 body model 庫（Python package），不是獨立 checkpoint。  
隨 Multi-HMR 打包的 Anny checkpoint license 見 §3.3。

### 2.5 五層分層判定

| 層 | 授權 | 備注 |
|----|------|------|
| Code license (Anny 庫本身) | **Apache 2.0** | README + LICENSE 文件逐字確認 |
| Anny native topology | **Apache 2.0** | 同上 |
| smplx 拓撲（可選下載） | **NC-only** | README 明文："non-commercial use only" |
| soma 拓撲 | **Apache 2.0** | README 明文 |
| MakeHuman assets (mpfb2) | **CC0 1.0** | README 明文 |

### 2.6 結論標籤

**Anny 本身**：`PRODUCT_CLEAN_PERMISSIVE`（Apache 2.0，不使用 smplx 拓撲包）

> ⚠️ 使用 smplx 拓撲則降級為 `RESEARCH_ONLY_NC`

---

## 3. Multi-HMR + Anny Checkpoint (naver/multi-hmr)

### 3.1 Repo 固化

```
GitHub URL:  https://github.com/naver/multi-hmr
主分支 HEAD: 651fb411e1cbcc626aaa5f38805ecab9cc891f7a  (refs/heads/master)
核驗時間:    2026-07-06
```

### 3.2 LICENSE.txt（全文逐字）

```
Multi-HMR, Copyright (C) 2025 NAVER Corporation. All Rights Reserved.

Non-Commercial License

Subject to any LICENSE EXCEPTIONS, NAVER Corporation ("NAVER") hereby grants 
you a non-exclusive, non-sublicensable, non-transferable license to use the 
Materials, subject to the following conditions:

(1) SCOPE OF USE: The Materials are used solely for non-commercial purposes 
("Purpose"). You may not use the Materials or derivatives thereof for any 
commercial purpose (i.e., primarily intended for or directed towards commercial 
advantage or monetary compensation). You may not distribute the Materials or 
derivatives thereof under different terms and conditions as this License.

(2) COPYRIGHT: The above copyright notice and this License along with the 
disclaimer below shall be retained in all copies and derivatives.

(3) TERM: The License automatically terminates without notice if you fail to 
comply with its terms or the Purpose no longer exists.
...
```

### 3.3 Multi-HMR + Anny Checkpoint 專項授權（逐字）

URL: `https://download.europe.naverlabs.com/ComputerVision/MultiHMR/Checkpoint_License_Anny.txt`

```
Multi-HMR checkpoint with Anny body model, Copyright (C) 2025 NAVER Corporation. 
All Rights Reserved.

Non-Commercial License

Subject to any LICENSE EXCEPTIONS, NAVER Corporation ("NAVER") hereby grants 
you a non-exclusive, non-sublicensable, non-transferable license to use the 
Materials, subject to the following conditions:

(1) SCOPE OF USE: The Materials are used solely for non-commercial purposes 
("Purpose"). You may not use the Materials or derivatives thereof for any 
commercial purpose (i.e., primarily intended for or directed towards commercial 
advantage or monetary compensation).
```

### 3.4 README License 段落（逐字）

```
## License
Code and checkpoints are provided under the terms of this LICENSE and 
accompanying NOTICE.
The licence for the Multi-HMR checkpoint with the Anny body model is here: 
LICENSE (https://download.europe.naverlabs.com/ComputerVision/MultiHMR/Checkpoint_License_Anny.txt)
```

### 3.5 Body Model 層核驗

README 明文：Multi-HMR 支持兩種 body model：
- **SMPL-X**（需自行下載 `SMPLX_NEUTRAL.npz`，SMPL-X 授權另計）
- **Anny**（checkpoint `multiHMR_672_L_anny.pt`，受上述 NC License 約束）

### 3.6 五層分層判定

| 層 | 授權 | 關鍵原文 |
|----|------|---------|
| Code license | **NC（NAVER Non-Commercial）** | LICENSE.txt §1 逐字 |
| Checkpoint (SMPL-X 版) | **NC（NAVER）+ SMPL-X NC** | LICENSE.txt；SMPL-X 單獨 NC |
| Checkpoint (Anny 版) | **NC（NAVER）** | Checkpoint_License_Anny.txt §1 逐字 |
| Body model/rig | **Anny = Apache 2.0**（但 checkpoint 綁定 NC） | Anny repo LICENSE |
| Output mesh | 受 checkpoint NC License 約束 | Checkpoint_License_Anny.txt |

### 3.7 結論標籤

**`RESEARCH_ONLY_NC`**

原文："You may not use the Materials or derivatives thereof for any commercial purpose"  
（Checkpoint_License_Anny.txt §1 及 LICENSE.txt §1，逐字確認）

> ⚠️ Anny 代碼本體是 Apache 2.0，但 Multi-HMR checkpoint 綁定 NC License。  
> 若自行訓練基於 Anny 的模型（不用 Multi-HMR checkpoint），理論上 checkpoint 不受此 NC 約束；  
> 但訓練數據集（BEDLAM/AGORA/3DPW）各有額外許可條款，需逐一核驗（見 NOTICE.txt §2.A–D）。

---

## 4. 核驗結論總表

| 候選 | Code | Checkpoint | Body Rig | 結論標籤 | 進產品前提 |
|------|------|-----------|---------|---------|-----------|
| **SAM 3D Body** | SAM License（無 NC） | SAM License（gated，需申請） | MHR（SAM License bundle） | **PRODUCT_CANDIDATE_CUSTOM_LICENSE** | Legal review：貿易管制 §1.b.v；出版致謝 §1.b.ii；gated 申請 |
| **Anny（代碼庫本身）** | Apache 2.0 | N/A（library） | Anny native topology Apache 2.0 | **PRODUCT_CLEAN_PERMISSIVE** | ✅ 可直接進產品；但禁止使用 smplx 拓撲包 |
| **Multi-HMR + Anny checkpoint** | NC（NAVER） | NC（NAVER） | Anny = Apache 2.0，但 checkpoint 綁 NC | **RESEARCH_ONLY_NC** | 不得商用 |

---

## 5. 授權鏈路示意

```
SAM 3D Body 路線：
  代碼(SAM License) + Checkpoint(SAM License,gated) + MHR rig(SAM License bundle)
  → 整條鏈 SAM License，無 NC，但需 legal review + gated 申請

Anny 自建路線：
  Anny 代碼庫(Apache 2.0) + 自訓練 checkpoint + Anny native topology(Apache 2.0)
  → 如不使用 smplx 拓撲包、不依賴 Multi-HMR NC checkpoint，整條鏈 Apache 2.0 CLEAN
  
Multi-HMR + Anny checkpoint 路線：
  Multi-HMR 代碼(NC) + Anny checkpoint(NC) + Anny rig(Apache 2.0)
  → checkpoint 層 NC 污染，整條鏈 RESEARCH_ONLY_NC
```

---

## 6. 早期衝突解釋

T0 之前 Hermes 報告 "CC BY-NC 4.0" 是 **二手調研 subagent 的幻覺輸出**，非原始文件。  
本報告以本機 curl 直接抓取的 LICENSE 原文為唯一事實依據：  
`facebookresearch/sam-3d-body` 的 LICENSE 為 "SAM License Last Updated: November 19, 2025"，無 NC 條款。

---

## 7. T0 裁決輸入（供 Jason 決策）

依據固化證據，T0 裁決選項：

**A. SAM 3D Body = SAM License 且無 NC**  
→ 允許跑單帧探針，標 PRODUCT_CANDIDATE_CUSTOM_LICENSE  
→ 進產品前需：① 申請 HF gated access；② Legal review SAM License §1.b.v 貿易管制  

**B. SAM 3D Body + 並行考慮 Anny 自建路線**  
→ SAM 探針同 A；  
→ Anny 自建：Apache 2.0 CLEAN，需自訓練或找現成 Anny-topology checkpoint（不用 Multi-HMR NC 版）

**C. 跳過 SAM 3D Body，直接走 Anny Apache 2.0 自建路線**  
→ 最潔淨，但需要訓練或找 Apache 2.0 checkpoint（Multi-HMR NC 版不可用）

**獨立於以上**：Multi-HMR + Anny checkpoint = RESEARCH_ONLY_NC，不進產品線。

---

*T0 報告由 Hermes 于 2026-07-06 根據原始 LICENSE 文件逐字生成。  
所有引用均為 curl 直接抓取，commit hash 已固化，可重現核驗。*
