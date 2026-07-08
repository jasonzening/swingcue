"""
ghost004_step1_finish_align.py  —  GHOST-004 Step 1 收尾
finish 段两机位对齐 (impact 为公共锚点，裁齐 follow-through 尾段)

输入:
  output/ghost004/fo_pose_sequence.npz   (NF=97,  impact=fr052, finish=fr074)
  output/ghost004/dtl_pose_sequence.npz  (NF=119, impact=fr059, finish=fr105)

输出:
  output/ghost004/fo_pose_sequence_aligned.npz   — 已对齐版 (NF 裁至 impact + follow_len)
  output/ghost004/dtl_pose_sequence_aligned.npz  — 已对齐版 (NF 裁至 impact + follow_len)

对齐策略:
  以 impact 为公共锚点（地面真相），两条各自保留从 frame 0 到 impact+follow_len 的段。
  follow_len = min(fo_follow_frames, dtl_follow_frames)
  这样两条时序长度相同、impact 帧绝对位置相同 (=impact_fo, impact_dtl 分别不变)。
  归一化到 impact 后帧数相同即可满足 Step 2 retarget 时序对齐要求。
"""
import numpy as np
from pathlib import Path

OUT = Path("output/ghost004")

fo  = np.load(OUT / "fo_pose_sequence.npz",  allow_pickle=False)
dtl = np.load(OUT / "dtl_pose_sequence.npz", allow_pickle=False)

fo_imp  = int(fo["anchors_impact"]);   fo_fin  = int(fo["anchors_finish"]);   fo_nf  = fo["body_pose_params"].shape[0]
dtl_imp = int(dtl["anchors_impact"]); dtl_fin = int(dtl["anchors_finish"]); dtl_nf = dtl["body_pose_params"].shape[0]

fo_follow  = fo_fin  - fo_imp
dtl_follow = dtl_fin - dtl_imp
follow_len = min(fo_follow, dtl_follow)

# Ensure we don't go past actual NF
fo_end  = min(fo_imp  + follow_len + 1, fo_nf)
dtl_end = min(dtl_imp + follow_len + 1, dtl_nf)

print(f"FO:  NF={fo_nf}  impact=fr{fo_imp:03d}  finish=fr{fo_fin:03d}  follow={fo_follow}")
print(f"DTL: NF={dtl_nf}  impact=fr{dtl_imp:03d}  finish=fr{dtl_fin:03d}  follow={dtl_follow}")
print(f"follow_len (min) = {follow_len}")
print(f"Cropping: FO [0..{fo_end})  DTL [0..{dtl_end})")

def crop_npz(npz, end_frame, **extra_kwargs):
    result = {}
    for k in npz.files:
        v = npz[k]
        if hasattr(v, '__len__') and len(v.shape) > 0 and v.shape[0] == npz["body_pose_params"].shape[0]:
            result[k] = v[:end_frame]
        else:
            result[k] = v
    result.update(extra_kwargs)
    return result

fo_aligned  = crop_npz(fo,  fo_end,  follow_len_cropped=np.int32(follow_len))
dtl_aligned = crop_npz(dtl, dtl_end, follow_len_cropped=np.int32(follow_len))

np.savez(str(OUT / "fo_pose_sequence_aligned.npz"),  **fo_aligned)
np.savez(str(OUT / "dtl_pose_sequence_aligned.npz"), **dtl_aligned)

print(f"\nSaved: fo_pose_sequence_aligned.npz   (NF={fo_end})")
print(f"Saved: dtl_pose_sequence_aligned.npz  (NF={dtl_end})")
print(f"\nPost-alignment check:")
print(f"  FO:  impact at fr{fo_imp:03d},  end at fr{fo_end-1:03d}  follow_frames={fo_end-1-fo_imp}")
print(f"  DTL: impact at fr{dtl_imp:03d},  end at fr{dtl_end-1:03d}  follow_frames={dtl_end-1-dtl_imp}")
print(f"  Follow delta: {abs((fo_end-1-fo_imp)-(dtl_end-1-dtl_imp))} frames  (should be 0)")
print("Step 1 finish alignment done.")
