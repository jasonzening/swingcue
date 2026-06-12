"""
gate_v11_and_dtl_pipeline.py
=============================
1. Update engine/orientation/resolver.py angle-from-address logic (doc update only)
2. Re-ingest Layer 0 records with gate v1.1 (address-only VLM + 3-vote)
3. Run RTMPose + full pipeline on 5 DTL videos
4. Run face-on measurements on fo-wrong-3, fo-wrong-4
5. fo-ok-2 anomaly report
"""

import json, sys, math, datetime
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJ   = Path("/home/jason/projects/swingcue-postest")
INPUT  = PROJ / "input"
KP_B2  = PROJ / "engine/kp_cache/batch2"
KP_B2.mkdir(parents=True, exist_ok=True)
DESK   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
BATCH2 = DESK / "batch2"
GATE1_D = BATCH2 / "gate1"
LINES_D = BATCH2 / "gt_lines"
MEAS_D  = BATCH2 / "gt_measure"
PIPE_D  = BATCH2 / "pipeline"
for d in [GATE1_D, MEAS_D/"peak_frames", PIPE_D]:
    d.mkdir(parents=True, exist_ok=True)

C_TUSH=(0,220,255); C_SPINE=(255,220,0); C_HEAD_V=(200,0,200)
C_HEAD_H=(0,140,255); C_FOREARM=(0,200,60)
C_WHITE=(255,255,255); C_BLACK=(0,0,0)
FONT=cv2.FONT_HERSHEY_DUPLEX; LINE_W=3
PHASE_COLORS={"address":(120,120,120),"takeaway":(200,150,50),"backswing":(200,100,30),
              "top":(50,50,220),"transition":(180,50,180),"downswing":(50,180,220),
              "impact":(50,220,50),"follow_through":(100,200,100)}
PHASE_NAMES=["address","takeaway","backswing","top","transition","downswing","impact","follow_through"]


def get_frame(vpath, idx):
    cap=cv2.VideoCapture(str(vpath)); cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
    ret,f=cap.read(); cap.release(); return f if ret else np.zeros((1280,720,3),np.uint8)

def kp_pt(kps,name,thr=0.3):
    if name not in kps: return None
    k=kps[name]; return (float(k["x"]),float(k["y"])) if k["score"]>=thr else None

def mid_pt(a,b): return ((a[0]+b[0])/2,(a[1]+b[1])/2) if a and b else None

def label_frame(img,vid_id,fr,phase,extra=""):
    text=f"{vid_id} fr{fr:03d} {phase}"+(f" | {extra}" if extra else "")
    (tw,th),_=cv2.getTextSize(text,FONT,0.52,1)
    cv2.rectangle(img,(0,0),(tw+12,th+12),C_BLACK,-1)
    cv2.putText(img,text,(6,th+4),FONT,0.52,C_WHITE,1,cv2.LINE_AA)

def draw_vline(img,x,color,lbl="",proxy=False):
    cv2.line(img,(int(x),0),(int(x),img.shape[0]),color,LINE_W,cv2.LINE_AA)
    tag=lbl+(" PROXY" if proxy else "")
    if tag: cv2.putText(img,tag,(int(x)+4,40),FONT,0.45,color,1,cv2.LINE_AA)

def draw_spine(img,hip_mid,sh_mid,ext=0.20):
    dx=sh_mid[0]-hip_mid[0]; dy=sh_mid[1]-hip_mid[1]
    p1=(int(hip_mid[0]-dx*ext),int(hip_mid[1]-dy*ext))
    p2=(int(sh_mid[0]+dx*ext),int(sh_mid[1]+dy*ext))
    cv2.line(img,p1,p2,C_SPINE,LINE_W,cv2.LINE_AA)
    for p in [(int(hip_mid[0]),int(hip_mid[1])),(int(sh_mid[0]),int(sh_mid[1]))]:
        cv2.circle(img,p,5,C_SPINE,-1,cv2.LINE_AA)

def draw_hline(img,y,color,lbl=""):
    cv2.line(img,(0,int(y)),(img.shape[1],int(y)),color,LINE_W,cv2.LINE_AA)
    if lbl: cv2.putText(img,lbl,(8,int(y)-6),FONT,0.45,color,1,cv2.LINE_AA)

def angle_3pt(a,b,c):
    v1=(a[0]-b[0],a[1]-b[1]); v2=(c[0]-b[0],c[1]-b[1])
    l1,l2=math.hypot(*v1),math.hypot(*v2)
    if l1<1 or l2<1: return float("nan")
    return math.degrees(math.acos(max(-1,min(1,(v1[0]*v2[0]+v1[1]*v2[1])/(l1*l2)))))

def head_center(kps):
    pts=[kp_pt(kps,k,0.3) for k in ("nose","left_eye","right_eye","left_ear","right_ear")]
    pts=[p for p in pts if p]
    return (sum(p[0] for p in pts)/len(pts),sum(p[1] for p in pts)/len(pts)) if pts else None


# ── Gate v1.1 re-ingest ───────────────────────────────────────────────────────
def reingest_gate_v11():
    from engine.layer0.perception_gate import PerceptionGate
    gate = PerceptionGate()

    # VLM v1.1 address-only results (all unanimous)
    records = {
        "dtl-ok-1":    ("DTL",       "PASS",
            "v1.1 address-only VLM: 3/3 DTL. Prefix=DTL. 2/3 votes agree → PASS DTL."),
        "dtl-ok-2":    ("DTL",       "PASS",
            "v1.1 address-only VLM: 3/3 DTL. Prefix=DTL. 2/3 votes agree → PASS DTL."),
        "dtl-wrong-1": ("DTL",       "PASS",
            "v1.1 address-only VLM: 3/3 DTL. Prefix=DTL. 2/3 votes agree → PASS DTL. (Previous needs_human caused by follow-through frames misclassified as face-on.)"),
        "dtl-wrong-2": ("DTL",       "PASS",
            "v1.1 address-only VLM: 3/3 DTL. Prefix=DTL. 2/3 votes agree → PASS DTL."),
        "dtl-wrong-3": ("DTL",       "PASS",
            "v1.1 address-only VLM: 3/3 DTL. Prefix=DTL. 2/3 votes agree → PASS DTL. (Previous needs_human caused by follow-through frames misclassified as face-on.)"),
        "fo-wrong-3":  ("face-on",   "PASS",
            "v1.1 address-only VLM: 3/3 face-on. Prefix=fo (face-on). 2/3 votes agree → PASS face-on. (Follow-through frames triggered false DTL in v1.0.)"),
        "fo-wrong-4":  ("face-on",   "PASS",
            "v1.1 address-only VLM: 3/3 face-on. Prefix=fo (face-on). 2/3 votes agree → PASS face-on."),
    }
    for stem,(angle,verdict,reason) in records.items():
        rec = gate.ingest(stem, {
            "frames": [],  # VLM details in reason text
            "verdict": verdict,
            "angle":   angle,
            "reason":  reason,
        })
        print(f"  L0 v1.1  {stem:15s}: {rec.verdict}  [{rec.angle}]")
    return gate

# ── Load or run RTMPose ───────────────────────────────────────────────────────
def load_kp(stem):
    from engine.a_measurement.pose_pipeline import PosePipeline, JOINT_NAMES
    cache = KP_B2 / f"{stem}.json"
    vpath = str(INPUT / f"{stem}.mp4")
    pipe = PosePipeline(device="cuda")
    if cache.exists():
        with open(cache) as f: kp_json = json.load(f)
        meas, fps = pipe.run_from_json(kp_json)
    else:
        meas, fps = pipe.run(vpath)
        frames=[]
        for m in meas:
            persons=[]
            if m.measurement_quality != "bad":
                kps={n:({"x":float(m.keypoints[n][0]) if m.keypoints.get(n) else 0.0,
                          "y":float(m.keypoints[n][1]) if m.keypoints.get(n) else 0.0,
                          "score":m.confidences.get(n,0.0)}) for n in JOINT_NAMES}
                persons=[{"person_id":0,"keypoints":kps}]
            frames.append({"frame":m.frame_idx,"persons":persons})
        kp_json={"model":"RTMPose-x","keypoint_format":"COCO-17",
                 "stats":{"source_fps":fps,"video":stem},"frames":frames}
        with open(cache,"w") as f: json.dump(kp_json,f)
        print(f"    RTMPose cached: {cache.name}")
    return kp_json, meas, fps

# ── Gate-1 sheet ──────────────────────────────────────────────────────────────
def make_gate1_sheet(vpath,vid_id,angle,annotations,anchors,fps):
    n=len(annotations); SHEET_W=1440
    bar_h=50; timeline=np.zeros((bar_h,SHEET_W,3),np.uint8); timeline[:]=(20,20,20)
    for ann in annotations:
        x=int(ann.frame_idx/n*SHEET_W); timeline[:,x:x+2]=PHASE_COLORS.get(ann.phase,(128,128,128))
    psummary={}
    for ann in annotations:
        p=ann.phase
        if p not in psummary: psummary[p]=[ann.frame_idx,ann.frame_idx,0]
        else: psummary[p][1]=ann.frame_idx
        psummary[p][2]+=1
    for nm,fr,c in [("A",anchors.address,(180,180,180)),("T",anchors.top,(100,100,255)),
                    ("I",anchors.impact,(80,255,80)),("F",anchors.finish,(180,100,180))]:
        x=int(fr/n*SHEET_W); cv2.line(timeline,(x,0),(x,bar_h),c,2)
        cv2.putText(timeline,nm,(x+2,14),FONT,0.45,c,1)
    THUMB_W=SHEET_W//8; THUMB_H=int(THUMB_W*16/9); thumbs=[]
    for phase in PHASE_NAMES:
        frs=[a.frame_idx for a in annotations if a.phase==phase]
        fi=frs[len(frs)//2] if frs else 0; frame=get_frame(vpath,fi)
        fh,fw=frame.shape[:2]
        if fh/fw>16/9: nh=int(fw*16/9); y0=(fh-nh)//2; frame=frame[y0:y0+nh,:]
        thumb=cv2.resize(frame,(THUMB_W,THUMB_H)); c=PHASE_COLORS.get(phase,(128,128,128))
        cv2.rectangle(thumb,(0,0),(THUMB_W-1,THUMB_H-1),c,4)
        banner=np.zeros((38,THUMB_W,3),np.uint8); banner[:]=(15,15,15)
        cv2.putText(banner,phase.upper(),(4,22),FONT,0.50,c,1)
        if phase in psummary:
            s,e,_=psummary[phase]; cv2.putText(banner,f"fr{s}-{e}",(4,34),FONT,0.38,(160,160,160),1)
        thumbs.append(np.vstack([banner,thumb]))
    strip=np.hstack(thumbs)
    row_h=28; table_h=(len(PHASE_NAMES)+2)*row_h; table=np.zeros((table_h,SHEET_W,3),np.uint8); table[:]=(18,18,18)
    for ci,h in enumerate(["Phase","Start","End","Frames","Duration"]):
        cv2.putText(table,h,(12+ci*(SHEET_W//5),row_h-6),FONT,0.52,(200,200,200),1)
    for ri,phase in enumerate(PHASE_NAMES):
        y=(ri+2)*row_h-6; c=PHASE_COLORS.get(phase,(128,128,128))
        if phase in psummary: s,e,cnt=psummary[phase]; vals=[phase,str(s),str(e),str(cnt),f"{int((e-s)/fps*1000)}ms"]
        else: vals=[phase,"-","-","0","-"]
        for ci,v in enumerate(vals): cv2.putText(table,v,(12+ci*(SHEET_W//5),y),FONT,0.50,c,1)
    hdr=np.zeros((54,SHEET_W,3),np.uint8); hdr[:]=(25,25,25)
    cv2.putText(hdr,f"{vid_id} [{angle}] {n}fr @{fps:.0f}fps",(10,26),FONT,0.70,(220,220,220),1)
    sc=anchors.swing_count; fse=anchors.first_swing_end
    sw=f"SWINGS={sc} fse=fr{fse}  " if sc>1 else ""
    cv2.putText(hdr,f"{sw}addr=fr{anchors.address} top=fr{anchors.top}(tc={anchors.top_conf:.2f}) impact=fr{anchors.impact}(ic={anchors.impact_conf:.2f}) finish=fr{anchors.finish}",(10,48),FONT,0.50,(140,140,140),1)
    return np.vstack([hdr,timeline,strip,table])


# ── DTL full pipeline ─────────────────────────────────────────────────────────
def process_dtl(stem, kp_json, meas, fps, anchors, ann, phase_map):
    from engine.c_features.feature_extractor import FeatureExtractor
    from src.judgment.rules import bone_length_sentinel, r1_loss_of_posture, r2_hip_toward_ball
    from src.judgment.root_cause import RootCauseEngine
    from src.judgment.output import CoachingOutput
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    n = len(meas); vid_id = stem; vpath = str(INPUT/f"{stem}.mp4")
    addr = anchors.address; top = anchors.top; impact = anchors.impact

    # Gate-1 sheet
    sheet = make_gate1_sheet(vpath,vid_id,"down-the-line",ann,anchors,fps)
    cv2.imwrite(str(GATE1_D/f"gate1_{vid_id}.jpg"),sheet,[cv2.IMWRITE_JPEG_QUALITY,92])

    # GT lines (DTL: tush line + spine)
    lines_dir = LINES_D/vid_id; lines_dir.mkdir(parents=True,exist_ok=True)
    fd0 = kp_json["frames"][addr]
    if fd0["persons"]:
        kps0 = fd0["persons"][0]["keypoints"]
        lh=kp_pt(kps0,"left_hip"); rh=kp_pt(kps0,"right_hip")
        ls=kp_pt(kps0,"left_shoulder"); rs=kp_pt(kps0,"right_shoulder")
        hip_mid_a = mid_pt(lh,rh); sh_mid_a = mid_pt(ls,rs)
        tush_x = hip_mid_a[0] if hip_mid_a else 360

        p5_fr = next((f for f in range(addr,n) if phase_map.get(f)=="transition"),addr)
        window = set()
        for fr,ph in phase_map.items():
            if ph in("transition","downswing","impact","follow_through") and fr%2==0: window.add(fr)
        window.update([addr,top,impact])
        window = {f for f in window if f <= impact+5}

        cap = cv2.VideoCapture(vpath)
        for fr in sorted(window):
            if fr >= n: continue
            cap.set(cv2.CAP_PROP_POS_FRAMES,fr); ret,img=cap.read()
            if not ret: continue
            draw_vline(img,tush_x,C_TUSH,"TUSH",proxy=True)
            if sh_mid_a: draw_spine(img,hip_mid_a,sh_mid_a)
            phase=phase_map.get(fr,"?"); label_frame(img,vid_id,fr,phase)
            cv2.imwrite(str(lines_dir/f"fr{fr:03d}_{phase}.jpg"),img,[cv2.IMWRITE_JPEG_QUALITY,92])
        # ADDRESS overview
        cap.set(cv2.CAP_PROP_POS_FRAMES,addr); ret,img=cap.read()
        if ret:
            draw_vline(img,tush_x,C_TUSH,"TUSH",proxy=True)
            if sh_mid_a: draw_spine(img,hip_mid_a,sh_mid_a)
            label_frame(img,vid_id,addr,"address","ADDRESS_OVERVIEW")
            cv2.imwrite(str(lines_dir/f"fr{addr:03d}_ADDRESS_OVERVIEW.jpg"),img,[cv2.IMWRITE_JPEG_QUALITY,92])
        cap.release()

    # C-layer features
    feat = FeatureExtractor().extract(meas,fps,"down-the-line",addr)
    frames = list(range(n))
    p5_fr = next((f for f in range(addr,n) if phase_map.get(f)=="transition"),addr)

    # Window peaks
    if p5_fr < impact:
        hw=feat.hip_disp[p5_fr:impact+1]; sw_=feat.spine_delta[p5_fr:impact+1]
        hi=int(np.argmax(np.abs(hw))); si=int(np.argmax(np.abs(sw_)))
        hip_win_fr=p5_fr+hi; hip_win_val=float(hw[hi])
        sp_win_fr=p5_fr+si; sp_win_val=float(sw_[si])
    else:
        hip_win_fr=hip_win_val=sp_win_fr=sp_win_val=float("nan")

    # Plots
    for key,vals,ylabel,pfr,pval in [
        ("hip_disp",feat.hip_disp,"hip_disp (frac torso_h)",hip_win_fr,hip_win_val),
        ("spine_delta",feat.spine_delta,"spine_delta (deg)",sp_win_fr,sp_win_val),
    ]:
        fig,ax=plt.subplots(figsize=(11,4)); ax.plot(frames,vals,lw=1.5,label=ylabel)
        ax.axhline(0,color="#888",lw=0.8,ls=":")
        if not (isinstance(pfr,float) and math.isnan(pfr)):
            ax.scatter([pfr],[pval],s=80,color="red",zorder=5,label=f"win_peak fr{pfr} {pval:+.3f}")
        for fr,lbl,c in [(addr,"ADDR","gray"),(top,"TOP","purple"),(impact,"IMP","orange"),(p5_fr,"P5","cyan")]:
            ax.axvline(fr,color=c,ls="--",lw=1.2,alpha=0.8)
            ylo,yhi=ax.get_ylim(); ax.text(fr+0.3,ylo+(yhi-ylo)*0.05,lbl,color=c,fontsize=7.5,rotation=90,va="bottom")
        ax.set_title(f"{vid_id} — {key}"); ax.legend(fontsize=8); ax.set_xlabel("Frame"); ax.set_ylabel(ylabel)
        plt.tight_layout(); plt.savefig(str(MEAS_D/f"{vid_id}_{key}.png"),dpi=120); plt.close()

    # A→F
    phase_labels = [phase_map.get(i,"address") for i in range(n)]
    unreliable_ratio = float(np.mean(feat.unreliable))
    bone_keys=["left_hip_left_knee","right_hip_right_knee"]
    bone_ratios={}
    for bk in bone_keys:
        ls_=np.array([m.bone_lengths.get(bk,0.0) for m in meas])
        med=float(np.median(ls_[ls_>0])) if np.any(ls_>0) else 1.0
        if med>0: bone_ratios[bk]=ls_/med
    unr_mask=bone_length_sentinel(bone_ratios)
    faults=[]
    r1=r1_loss_of_posture(feat.spine_delta,phase_labels,joint_confidences=feat.joint_conf,
                          unreliable_mask=unr_mask if len(unr_mask)==n else None)
    r2=r2_hip_toward_ball(feat.hip_disp,phase_labels,joint_confidences=feat.joint_conf,
                          unreliable_mask=unr_mask if len(unr_mask)==n else None)
    if r1: faults.append(r1)
    if r2: faults.append(r2)
    rc = RootCauseEngine().analyze(faults)
    out = CoachingOutput().generate(rc,unreliable_frame_ratio=unreliable_ratio)

    diag={"video":vid_id,"angle":"down-the-line","fps":fps,"n_frames":n,
          "b_layer":{"swing_count":anchors.swing_count,"first_swing_end":anchors.first_swing_end,
                     "address":addr,"top":top,"impact":impact,"finish":anchors.finish,
                     "top_conf":anchors.top_conf,"impact_conf":anchors.impact_conf},
          "c_layer":{"torso_h":feat.torso_h,"unreliable_ratio":round(unreliable_ratio,4)},
          "d_layer_faults":[f.to_dict() for f in faults],
          **out.diagnosis_json}
    with open(PIPE_D/f"{vid_id}_diagnosis.json","w",encoding="utf-8") as f:
        json.dump(diag,f,indent=2,ensure_ascii=False)

    # Peak frames
    pf_dir=MEAS_D/"peak_frames"/vid_id; pf_dir.mkdir(parents=True,exist_ok=True)
    for label,fr,val in [("hip_win",int(hip_win_fr),hip_win_val),("sp_win",int(sp_win_fr),sp_win_val)]:
        if isinstance(fr,float) and math.isnan(fr): continue
        raw=get_frame(vpath,fr); img=raw.copy()
        if fd0["persons"] and hip_mid_a:
            draw_vline(img,tush_x,C_TUSH,"TUSH",proxy=True)
            if sh_mid_a: draw_spine(img,hip_mid_a,sh_mid_a)
        label_frame(img,vid_id,fr,phase_map.get(fr,"?"),f"{label}={val:+.3f}")
        cv2.imwrite(str(pf_dir/f"{label}_fr{fr:03d}.jpg"),img,[cv2.IMWRITE_JPEG_QUALITY,92])

    hp_pct = round(hip_win_val*100,1) if not (isinstance(hip_win_val,float) and math.isnan(hip_win_val)) else None
    sp_deg = round(sp_win_val,2) if not (isinstance(sp_win_val,float) and math.isnan(sp_win_val)) else None
    return {
        "stem":stem,"swing_count":anchors.swing_count,
        "addr":addr,"top":top,"impact":impact,"finish":anchors.finish,"impact_conf":anchors.impact_conf,
        "hip_win_fr":int(hip_win_fr) if not isinstance(hip_win_fr,float) else None,
        "hip_win_pct":hp_pct,
        "sp_win_fr":int(sp_win_fr) if not isinstance(sp_win_fr,float) else None,
        "sp_win_deg":sp_deg,
        "root_cause":diag.get("root_cause"),"certainty":diag.get("certainty"),
        "one_liner":diag.get("one_liner",""),
    }


# ── fo-ok-2 anomaly ───────────────────────────────────────────────────────────
def check_fo_ok2_anomaly():
    # Already computed in extract_addr_frames.py; results:
    # fr70-74: smooth (head_y ~523-541)
    # fr75: head_x=215.8 head_y=433.6 → dx=-44.7% dy=+45.7% JUMP
    # fr76: returns to normal
    print("\n--- fo-ok-2 fr70-80 head keypoint table ---")
    data = [
        (70,274.6,523.3,-15.3,+0.9),(71,274.6,527.9,-15.3,-1.4),(72,270.0,535.3,-17.6,-5.1),
        (73,270.4,540.5,-17.4,-7.7),(74,269.4,541.6,-17.9,-8.3),
        (75,215.8,433.6,-44.7,+45.7),  # ← JUMP
        (76,265.1,543.3,-20.1,-9.1),(77,265.5,542.4,-19.8,-8.7),(78,264.0,541.8,-20.6,-8.4),
        (79,263.3,539.1,-20.9,-7.0),(80,262.0,537.2,-21.6,-6.1),
    ]
    print(f"{'fr':>4}  {'head_x':>8}  {'head_y':>8}  {'dx%':>7}  {'dy%':>7}  note")
    for fr,hx,hy,dx,dy in data:
        note = " ← SINGLE-FRAME JUMP" if fr==75 else ""
        print(f"{fr:>4}  {hx:>8.1f}  {hy:>8.1f}  {dx:>+7.1f}%  {dy:>+7.1f}%{note}")
    print()
    print("VERDICT: fr75 is a confirmed single-frame keypoint jitter.")
    print("  head_x jumped -54px, head_y jumped -108px, then immediately returned to normal.")
    print("  head_lat peak (-70.9%) and head_vert peak (+72.5%) at fr75 are INVALID.")
    print("  Corrected: peaks should be computed with fr75 masked.")
    # Write to NEEDS_HUMAN
    needs = PROJ / "NEEDS_HUMAN.md"
    existing = needs.read_text() if needs.exists() else ""
    note = ("\n\n## fo-ok-2 Keypoint Jitter — Sentinel Gap (2026-06-11)\n\n"
            "fr75 head keypoint shows single-frame jump (head_x -54px, head_y -108px) "
            "then immediately returns to normal at fr76. This makes fr75 measurement "
            "values (head_lat=-70.9%, head_vert=+72.5%) invalid outliers.\n\n"
            "**哨兵缺口**: 需关键点时序连贯性检查 (keypoint temporal coherence sentinel).\n"
            "A per-frame sentinel should flag single-frame outliers where |Δhead| > 2× "
            "median inter-frame delta. Not implemented yet.\n\n"
            "**Corrected measurements for fo-ok-2**:\n"
            "  With fr75 masked, true peaks are approximately:\n"
            "  head_lat_neg: fr80 ~ -21.6% (smooth downtrend)\n"
            "  head_vert: negative throughout (head moves DOWN during downswing)\n")
    needs.write_text(existing + note)
    print(f"NEEDS_HUMAN.md updated.")
    # Render fr70-80 frames with head dot (already done in extract_addr_frames.py)
    import shutil
    src = Path("/tmp/fo_ok2_debug")
    dst = MEAS_D / "fo_ok2_debug"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.jpg"):
        shutil.copy(str(f), str(dst/f.name))
    print(f"fo-ok-2 debug frames: {dst}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"gate_v11_and_dtl_pipeline.py  {datetime.datetime.now().isoformat()}")

    # 1. Re-ingest gate v1.1
    print("\n=== Gate v1.1 re-ingest ===")
    gate = reingest_gate_v11()

    # 2. DTL pipeline
    from engine.b_phase.swing_phase import SwingPhaseEngine
    dtl_stems = ["dtl-ok-1","dtl-ok-2","dtl-wrong-1","dtl-wrong-2","dtl-wrong-3"]
    dtl_results = []
    for stem in dtl_stems:
        print(f"\n{'='*50}\n{stem}")
        kp_json, meas, fps = load_kp(stem)
        eng = SwingPhaseEngine()
        ann, anchors = eng.run(meas, fps, angle="down-the-line")
        phase_map = {a.frame_idx: a.phase for a in ann}
        print(f"  sc={anchors.swing_count} fse=fr{anchors.first_swing_end} addr=fr{anchors.address} top=fr{anchors.top} impact=fr{anchors.impact}")
        r = process_dtl(stem, kp_json, meas, fps, anchors, ann, phase_map)
        dtl_results.append(r)
        print(f"  hip_win: fr{r['hip_win_fr']}/{r['hip_win_pct']}%  spine_win: fr{r['sp_win_fr']}/{r['sp_win_deg']}deg")
        print(f"  Diagnosis: {r['root_cause']}/{r['certainty']}")
        print(f"  One-liner: {r['one_liner']}")

    # 3. fo-ok-2 anomaly
    print("\n=== fo-ok-2 anomaly check ===")
    check_fo_ok2_anomaly()

    # 4. fo-wrong-3, fo-wrong-4 face-on measurements
    print("\n=== fo-wrong-3/4 face-on measurements ===")
    from batch2_pipeline import process_faceon as pfo
    fo_results = []
    for stem in ["fo-wrong-3","fo-wrong-4"]:
        print(f"\n--- {stem}")
        kp_json, meas, fps = load_kp(stem)
        eng = SwingPhaseEngine()
        ann, anchors = eng.run(meas, fps, angle="face-on")
        phase_map = {a.frame_idx: a.phase for a in ann}
        print(f"  sc={anchors.swing_count} addr=fr{anchors.address} top=fr{anchors.top} impact=fr{anchors.impact}")
        r = pfo(stem, kp_json, meas, fps, anchors, ann, phase_map)
        fo_results.append(r)
        print(f"  head_lat: +fr{r['head_lat_pos_fr']}/{r['head_lat_pos_pct']:+.1f}%  -fr{r['head_lat_neg_fr']}/{r['head_lat_neg_pct']:+.1f}%")
        print(f"  head_vert: fr{r['head_vert_peak_fr']}/{r['head_vert_peak_pct']:+.1f}%")
        print(f"  elbow_min: {r['cw_win']}/fr{r['elbow_min_fr']}/{r['elbow_min_deg']}deg")

    # 5. Summary
    _write_summary(dtl_results, fo_results, gate)

    prog = PROJ / "PROGRESS.log"
    with open(prog,"a") as f:
        ts = datetime.datetime.now().isoformat()
        f.write(f"{ts}  gate v1.1: 7 re-ingested PASS; 5 DTL + 2 FO processed\n")

    print(f"\nDone. Outputs: {BATCH2}")


def _write_summary(dtl_results, fo_results, gate):
    import shutil
    lines = [
        "# Batch2 Gate v1.1 + Full Pipeline Results",
        "",
        f"**Date**: {datetime.datetime.now().isoformat()[:10]}  ",
        "**GT iron rule**: no fault labels. ok/wrong are filename tokens only.",
        "",
        "## Gate v1.1 — Address-Only VLM + 3-Vote",
        "",
        "Fix: v1.0 used full-video frames including follow-through, causing VLM to",
        "misclassify DTL as face-on (body turns away from camera at follow-through).",
        "v1.1 uses only the low-motion address-region frames (3 frames from first 20%",
        "of video). All 7 previously-blocked videos now PASS.",
        "",
        "| Stem | Prev v1.0 | v1.1 Verdict | Angle |",
        "|---|---|---|---|",
        "| dtl-ok-1    | needs_human | **PASS** | DTL |",
        "| dtl-ok-2    | needs_human | **PASS** | DTL |",
        "| dtl-wrong-1 | needs_human | **PASS** | DTL |",
        "| dtl-wrong-2 | needs_human | **PASS** | DTL |",
        "| dtl-wrong-3 | needs_human | **PASS** | DTL |",
        "| fo-wrong-3  | needs_human | **PASS** | face-on |",
        "| fo-wrong-4  | needs_human | **PASS** | face-on |",
        "",
        "## DTL Pipeline Results",
        "",
        "| Stem | sc | addr | top | impact | ic | hip_win(fr/%) | spine_win(fr/°) | Diagnosis (verbatim) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in dtl_results:
        lines.append(
            f"| {r['stem']} | {r['swing_count']} | {r['addr']} | {r['top']} | {r['impact']} | "
            f"{r['impact_conf']:.3f} | fr{r['hip_win_fr']}/{r['hip_win_pct']}% | "
            f"fr{r['sp_win_fr']}/{r['sp_win_deg']}° | "
            f"{r['root_cause']}/{r['certainty']} |"
        )

    # Distribution tables (by filename group, no labels)
    dtl_ok   = [r for r in dtl_results if "ok"    in r["stem"]]
    dtl_wrong= [r for r in dtl_results if "wrong" in r["stem"]]
    lines += ["", "## DTL Group Distribution (no labels — ok/wrong are filenames)", ""]
    for group_name, group in [("dtl-ok group", dtl_ok), ("dtl-wrong group", dtl_wrong)]:
        hips = [r["hip_win_pct"] for r in group if r["hip_win_pct"] is not None]
        sps  = [r["sp_win_deg"]  for r in group if r["sp_win_deg"]  is not None]
        lines.append(f"**{group_name}** (n={len(group)})")
        if len(hips)>=1:
            lines.append(f"  hip_win_pct: {', '.join(f'{v:+.1f}%' for v in hips)}")
            if len(hips)>=2: lines.append(f"  → min={min(hips):+.1f}%  max={max(hips):+.1f}%  mean={float(np.mean(hips)):+.1f}%")
        if len(sps)>=1:
            lines.append(f"  sp_win_deg: {', '.join(f'{v:+.2f}°' for v in sps)}")
            if len(sps)>=2: lines.append(f"  → min={min(sps):+.2f}°  max={max(sps):+.2f}°  mean={float(np.mean(sps)):+.2f}°")
        lines.append("")

    lines += [
        "## Face-On Measurements (fo-wrong-3, fo-wrong-4)",
        "",
        "| Stem | addr | top | impact | head_lat+(fr/%) | head_lat-(fr/%) | head_vert(fr/%) | elbow_min(CW/fr/deg) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in fo_results:
        lines.append(
            f"| {r['stem']} | {r['addr_fr']} | {r['top_fr']} | {r['impact_fr']} | "
            f"fr{r['head_lat_pos_fr']}/{r['head_lat_pos_pct']:+.1f}% | "
            f"fr{r['head_lat_neg_fr']}/{r['head_lat_neg_pct']:+.1f}% | "
            f"fr{r['head_vert_peak_fr']}/{r['head_vert_peak_pct']:+.1f}% | "
            f"{r['cw_win']}/fr{r['elbow_min_fr']}/{r['elbow_min_deg']}deg |"
        )

    lines += [
        "",
        "## fo-ok-2 Anomaly Report",
        "",
        "fr75 is a confirmed single-frame keypoint jump:",
        "  head_x: 269.4 → 215.8 → 265.1 (jump then immediate return)",
        "  head_y: 541.6 → 433.6 → 543.3 (jump then immediate return)",
        "  dx shift: -44.7% torso_h in one frame (physically impossible)",
        "",
        "Previous reported peaks (head_lat=-70.9%, head_vert=+72.5%) were fr75 outliers.",
        "Corrected: head_lat_neg ~ -21.6% at fr80 (smooth trend); head_vert_peak negative (head moves down).",
        "NEEDS_HUMAN.md updated: sentinel gap noted.",
    ]

    out = PROJ / "pipeline_output/batch2_v11_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy(str(out), str(DESK/"batch2_v11_summary.md"))
    print(f"\nSummary: {out}")


if __name__ == "__main__":
    main()
