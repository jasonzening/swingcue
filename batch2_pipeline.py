"""
batch2_pipeline.py
==================
Process batch2 videos:
1. Ingest Layer 0 records for all 11 videos
2. Run Gate-1 + GT-lines + measurements for the 4 PASS face-on videos
3. Output gate1 sheets, gt_lines, measurements, peak frames
4. Write per-swing summary table
NO A→F for face-on (feature not online). NO DTL (all needs_human).
NO fault labels.
"""
import json, sys, math, time, datetime
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJ   = Path("/home/jason/projects/swingcue-postest")
INPUT  = PROJ / "input"
KP_DIR = PROJ / "engine/kp_cache/batch2"
KP_DIR.mkdir(parents=True, exist_ok=True)
DESK   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
BATCH2 = DESK / "batch2"
for d in [BATCH2/"gate1", BATCH2/"gt_lines", BATCH2/"gt_measure"/"peak_frames"]:
    d.mkdir(parents=True, exist_ok=True)

C_HEAD_V = (200,0,200); C_HEAD_H=(0,140,255); C_FOREARM=(0,200,60)
C_WHITE=(255,255,255); C_BLACK=(0,0,0)
FONT = cv2.FONT_HERSHEY_DUPLEX; LINE_W=3

PHASE_COLORS = {
    "address":(120,120,120),"takeaway":(200,150,50),"backswing":(200,100,30),
    "top":(50,50,220),"transition":(180,50,180),"downswing":(50,180,220),
    "impact":(50,220,50),"follow_through":(100,200,100),
}
PHASE_NAMES=["address","takeaway","backswing","top","transition","downswing","impact","follow_through"]

def get_frame(vpath, idx):
    cap = cv2.VideoCapture(str(vpath)); cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
    ret,f=cap.read(); cap.release(); return f if ret else np.zeros((1280,720,3),np.uint8)

def kp_pt(kps,name,thr=0.3):
    if name not in kps: return None
    k=kps[name]; return (float(k["x"]),float(k["y"])) if k["score"]>=thr else None

def mid_pt(a,b):
    return ((a[0]+b[0])/2,(a[1]+b[1])/2) if a and b else None

def angle_3pt(a,b,c):
    v1=(a[0]-b[0],a[1]-b[1]); v2=(c[0]-b[0],c[1]-b[1])
    l1,l2=math.hypot(*v1),math.hypot(*v2)
    if l1<1 or l2<1: return float("nan")
    return math.degrees(math.acos(max(-1,min(1,(v1[0]*v2[0]+v1[1]*v2[1])/(l1*l2)))))

def head_center(kps):
    pts=[kp_pt(kps,k,0.3) for k in ("nose","left_eye","right_eye","left_ear","right_ear")]
    pts=[p for p in pts if p]
    return (sum(p[0] for p in pts)/len(pts),sum(p[1] for p in pts)/len(pts)) if pts else None

def label_frame(img,vid_id,fr,phase,extra=""):
    text=f"{vid_id} fr{fr:03d} {phase}"+(f" | {extra}" if extra else "")
    (tw,th),_=cv2.getTextSize(text,FONT,0.52,1)
    cv2.rectangle(img,(0,0),(tw+12,th+12),C_BLACK,-1)
    cv2.putText(img,text,(6,th+4),FONT,0.52,C_WHITE,1,cv2.LINE_AA)

def draw_vline(img,x,color,lbl=""):
    cv2.line(img,(int(x),0),(int(x),img.shape[0]),color,LINE_W,cv2.LINE_AA)
    if lbl: cv2.putText(img,lbl,(int(x)+4,40),FONT,0.45,color,1,cv2.LINE_AA)

def draw_hline(img,y,color,lbl=""):
    cv2.line(img,(0,int(y)),(img.shape[1],int(y)),color,LINE_W,cv2.LINE_AA)
    if lbl: cv2.putText(img,lbl,(8,int(y)-6),FONT,0.45,color,1,cv2.LINE_AA)

def draw_forearm(img,sh,el,wr):
    sh=(int(sh[0]),int(sh[1])); el=(int(el[0]),int(el[1])); wr=(int(wr[0]),int(wr[1]))
    cv2.line(img,sh,el,C_FOREARM,LINE_W,cv2.LINE_AA); cv2.line(img,el,wr,C_FOREARM,LINE_W,cv2.LINE_AA)
    for p in(sh,el,wr): cv2.circle(img,p,5,C_FOREARM,-1,cv2.LINE_AA)
    ang=angle_3pt(sh,el,wr)
    if not math.isnan(ang): cv2.putText(img,f"{ang:.0f}deg",(el[0]+8,el[1]-8),FONT,0.50,C_FOREARM,1,cv2.LINE_AA)

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
        if fh/fw>16/9: new_h=int(fw*16/9); y0=(fh-new_h)//2; frame=frame[y0:y0+new_h,:]
        thumb=cv2.resize(frame,(THUMB_W,THUMB_H)); c=PHASE_COLORS.get(phase,(128,128,128))
        cv2.rectangle(thumb,(0,0),(THUMB_W-1,THUMB_H-1),c,4)
        banner=np.zeros((38,THUMB_W,3),np.uint8); banner[:]=(15,15,15)
        cv2.putText(banner,phase.upper(),(4,22),FONT,0.50,c,1)
        if phase in psummary: s,e,_=psummary[phase]; cv2.putText(banner,f"fr{s}-{e}",(4,34),FONT,0.38,(160,160,160),1)
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


def process_faceon(stem, kp_json, meas, fps, anchors, ann, phase_map):
    n=len(meas); vid_id=stem; vpath=str(INPUT/f"{stem}.mp4")
    # Gate-1 sheet
    sheet=make_gate1_sheet(vpath,vid_id,"face-on",ann,anchors,fps)
    cv2.imwrite(str(BATCH2/"gate1"/f"gate1_{vid_id}.jpg"),sheet,[cv2.IMWRITE_JPEG_QUALITY,92])
    # Address anchors
    addr_fr=anchors.address; impact_fr=anchors.impact
    fd0=kp_json["frames"][addr_fr]
    kps0=fd0["persons"][0]["keypoints"] if fd0["persons"] else {}
    hc0=head_center(kps0); addr_hx,addr_hy=hc0 if hc0 else (360,200)
    torso_h=meas[addr_fr].torso_height() or 200.0
    # GT lines sub-folder
    lines_dir=BATCH2/"gt_lines"/vid_id; lines_dir.mkdir(parents=True,exist_ok=True)
    # Draw lines on key windows
    def render_with_lines(fr,extra="",show_forearm=False):
        raw=get_frame(vpath,fr); img=raw.copy()
        phase=phase_map.get(fr,"?")
        draw_vline(img,addr_hx,C_HEAD_V,"HEAD-V")
        draw_hline(img,addr_hy,C_HEAD_H,"HEAD-H")
        if show_forearm and fr<len(kp_json["frames"]):
            fd=kp_json["frames"][fr]
            if fd["persons"]:
                kps=fd["persons"][0]["keypoints"]
                sh=kp_pt(kps,"left_shoulder"); el=kp_pt(kps,"left_elbow"); wr=kp_pt(kps,"left_wrist")
                if sh and el and wr: draw_forearm(img,sh,el,wr)
        hc=head_center(kp_json["frames"][fr]["persons"][0]["keypoints"]) if kp_json["frames"][fr]["persons"] else None
        if hc: cv2.circle(img,(int(hc[0]),int(hc[1])),6,C_HEAD_V,-1,cv2.LINE_AA)
        label_frame(img,vid_id,fr,phase,extra)
        return img
    # All frames stride-2 + anchors, full range
    window_bs=set(); window_ds=set(); window_ft=set()
    for fr,ph in phase_map.items():
        if ph in("takeaway","backswing","top") and fr%2==0: window_bs.add(fr)
        if ph in("transition","downswing","impact") and fr%2==0: window_ds.add(fr)
        if impact_fr<=fr<=impact_fr+8: window_ft.add(fr)
    window_bs.add(addr_fr); window_bs.add(anchors.top)
    window_ds.add(impact_fr)
    for fr in sorted(window_bs):
        img=render_with_lines(fr)
        cv2.imwrite(str(lines_dir/f"bs_fr{fr:03d}_{phase_map.get(fr,'?')}.jpg"),img,[cv2.IMWRITE_JPEG_QUALITY,92])
    for fr in sorted(window_ds):
        img=render_with_lines(fr)
        cv2.imwrite(str(lines_dir/f"ds_fr{fr:03d}_{phase_map.get(fr,'?')}.jpg"),img,[cv2.IMWRITE_JPEG_QUALITY,92])
    for fr in sorted(window_ft):
        img=render_with_lines(fr,show_forearm=True)
        cv2.imwrite(str(lines_dir/f"ft_fr{fr:03d}_{phase_map.get(fr,'?')}.jpg"),img,[cv2.IMWRITE_JPEG_QUALITY,92])
    # Measurements
    meas_dir=BATCH2/"gt_measure"; meas_dir.mkdir(parents=True,exist_ok=True)
    # Head lateral: P1→impact
    p5_fr=next((f for f in range(addr_fr,n) if phase_map.get(f)=="transition"),addr_fr)
    lat_frs=list(range(addr_fr,min(impact_fr+1,n)))
    lat_pct=[]; vert_pct=[]
    for fr in lat_frs:
        fd=kp_json["frames"][fr]
        if fd["persons"]:
            hc=head_center(fd["persons"][0]["keypoints"])
            if hc: lat_pct.append((hc[0]-addr_hx)/torso_h*100)
            else: lat_pct.append(float("nan"))
        else: lat_pct.append(float("nan"))
    vert_frs=list(range(p5_fr,min(impact_fr+1,n)))
    for fr in vert_frs:
        fd=kp_json["frames"][fr]
        if fd["persons"]:
            hc=head_center(fd["persons"][0]["keypoints"])
            if hc: vert_pct.append((addr_hy-hc[1])/torso_h*100)
            else: vert_pct.append(float("nan"))
        else: vert_pct.append(float("nan"))
    lat_arr=np.array(lat_pct); vert_arr=np.array(vert_pct)
    def nanpeak(arr,frs,dir="pos"):
        v=arr.copy(); v[np.isnan(v)]=0
        idx=int(np.argmax(v)) if dir=="pos" else int(np.argmin(v))
        return frs[idx], round(float(arr[idx]) if not math.isnan(arr[idx]) else 0,1)
    lat_pos_fr,lat_pos_pct=nanpeak(lat_arr,lat_frs,"pos")
    lat_neg_fr,lat_neg_pct=nanpeak(lat_arr,lat_frs,"neg")
    vert_peak_fr,vert_peak_pct=nanpeak(vert_arr,vert_frs,"pos")
    # CW window (dynamic)
    def cw_window(impact_fr,max_s=20):
        for fr in range(impact_fr+1,min(impact_fr+max_s+1,n)):
            fd=kp_json["frames"][fr]
            if not fd["persons"]: continue
            kps=fd["persons"][0]["keypoints"]
            lw=kp_pt(kps,"left_wrist"); rw=kp_pt(kps,"right_wrist")
            lh=kp_pt(kps,"left_hip"); rh=kp_pt(kps,"right_hip")
            wm=mid_pt(lw,rw) or lw or rw
            hm=mid_pt(lh,rh) or lh or rh
            if wm and hm and wm[1]<hm[1]: return fr
        return min(impact_fr+max_s,n-1)
    cw_end=cw_window(impact_fr)
    elbow_data=[]
    for fr in range(impact_fr,cw_end+1):
        if fr>=n: break
        fd=kp_json["frames"][fr]
        if not fd["persons"]: elbow_data.append((fr,float("nan"))); continue
        kps=fd["persons"][0]["keypoints"]
        sh=kp_pt(kps,"left_shoulder"); el=kp_pt(kps,"left_elbow"); wr=kp_pt(kps,"left_wrist")
        elbow_data.append((fr,angle_3pt(sh,el,wr) if sh and el and wr else float("nan")))
    valid_e=[(fr,a) for fr,a in elbow_data if not math.isnan(a)]
    elbow_min_fr,elbow_min_deg=(valid_e[min(range(len(valid_e)),key=lambda i:valid_e[i][1])]) if valid_e else (impact_fr,float("nan"))
    # Plots
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for key,frs_list,vals,ylabel,peak_info in [
        ("head_lateral",lat_frs,lat_pct,"head_x offset (% torso_h)",
         [(lat_pos_fr,lat_pos_pct,"red","+"),(lat_neg_fr,lat_neg_pct,"blue","-")]),
        ("head_vertical",vert_frs,vert_pct,"head_up (% torso_h)",
         [(vert_peak_fr,vert_peak_pct,"red","up")]),
    ]:
        fig,ax=plt.subplots(figsize=(11,4))
        ax.plot(frs_list,vals,lw=1.5,label=ylabel); ax.axhline(0,color="#888",lw=0.8,ls=":")
        for pfr,pval,pc,plbl in peak_info:
            ax.scatter([pfr],[pval],s=80,color=pc,zorder=5,label=f"{plbl} fr{pfr} {pval:+.1f}%")
        for fr,lbl,c in [(addr_fr,"ADDR","gray"),(anchors.top,"TOP","purple"),(impact_fr,"IMP","orange"),(p5_fr,"P5","cyan")]:
            ax.axvline(fr,color=c,ls="--",lw=1.2,alpha=0.8)
            ylo,yhi=ax.get_ylim(); ax.text(fr+0.3,ylo+(yhi-ylo)*0.05,lbl,color=c,fontsize=7.5,rotation=90,va="bottom")
        ax.set_title(f"{vid_id} — {key}"); ax.legend(fontsize=8); ax.set_xlabel("Frame"); ax.set_ylabel(ylabel)
        plt.tight_layout(); plt.savefig(str(meas_dir/f"{vid_id}_{key}.png"),dpi=120); plt.close()
    # Elbow plot
    ef=[fr for fr,_ in elbow_data]; ea=[a for _,a in elbow_data]
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(ef,ea,color="#d7191c",lw=1.5,marker="o",ms=5,label="lead elbow angle")
    if not math.isnan(elbow_min_deg): ax.scatter([elbow_min_fr],[elbow_min_deg],s=100,color="navy",zorder=5,label=f"min fr{elbow_min_fr} {elbow_min_deg:.0f}deg")
    ax.axvline(impact_fr,color="orange",ls="--",lw=1.2); ax.set_title(f"{vid_id} — elbow_angle (CW window fr{impact_fr}-fr{cw_end})")
    ax.set_xlabel("Frame"); ax.set_ylabel("Elbow angle (deg)"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(str(meas_dir/f"{vid_id}_elbow_angle.png"),dpi=120); plt.close()
    # Peak frames
    pf_dir=BATCH2/"gt_measure"/"peak_frames"/vid_id; pf_dir.mkdir(parents=True,exist_ok=True)
    for fr,lbl,extra in [(lat_pos_fr,"head_lat_pos",f"+{lat_pos_pct:.1f}%"),
                          (lat_neg_fr,"head_lat_neg",f"{lat_neg_pct:.1f}%"),
                          (vert_peak_fr,"head_vert",f"+{vert_peak_pct:.1f}%"),
                          (elbow_min_fr,"elbow_min",f"{elbow_min_deg:.0f}deg")]:
        img=render_with_lines(fr,extra=f"{lbl}={extra}",show_forearm=("elbow" in lbl))
        cv2.imwrite(str(pf_dir/f"{lbl}_fr{fr:03d}.jpg"),img,[cv2.IMWRITE_JPEG_QUALITY,92])
    return {
        "stem":stem, "swing_count":anchors.swing_count,
        "addr_fr":addr_fr,"top_fr":anchors.top,"impact_fr":impact_fr,"finish_fr":anchors.finish,
        "impact_conf":anchors.impact_conf,
        "head_lat_pos_fr":lat_pos_fr,"head_lat_pos_pct":lat_pos_pct,
        "head_lat_neg_fr":lat_neg_fr,"head_lat_neg_pct":lat_neg_pct,
        "head_vert_peak_fr":vert_peak_fr,"head_vert_peak_pct":vert_peak_pct,
        "cw_win":f"fr{impact_fr}-fr{cw_end}",
        "elbow_min_fr":elbow_min_fr,"elbow_min_deg":round(elbow_min_deg,1) if not math.isnan(elbow_min_deg) else None,
        "torso_h":round(torso_h,0),
    }


def main():
    print(f"batch2_pipeline.py  {datetime.datetime.now().isoformat()}")

    # ── 1. Ingest Layer 0 records ─────────────────────────────────────────────
    from engine.layer0.perception_gate import PerceptionGate
    gate=PerceptionGate()
    VLM=json.load(open("/tmp/vlm_batch2.json")) if Path("/tmp/vlm_batch2.json").exists() else {}
    # VLM data is embedded in this script's global — imported from ingest
    from ingest_batch2_layer0 import VLM_BATCH2 as VLM_DATA
    for stem,vlm in VLM_DATA.items():
        rec=gate.ingest(stem,vlm); print(f"  L0 {stem:15s}: {rec.verdict}")

    # ── 2. Process PASS face-on videos ───────────────────────────────────────
    from engine.a_measurement.pose_pipeline import PosePipeline, JOINT_NAMES
    from engine.b_phase.swing_phase import SwingPhaseEngine
    from engine.orientation.resolver import OrientationResolver

    PASS_FO=["fo-ok-1","fo-ok-2","fo-wrong-1","fo-wrong-2"]
    results=[]
    for stem in PASS_FO:
        print(f"\n{'='*50}\n{stem}")
        vpath=INPUT/f"{stem}.mp4"; cache=KP_DIR/f"{stem}.json"
        if cache.exists():
            with open(cache) as f: kp_json=json.load(f)
            pipe=PosePipeline(device="cuda"); meas,fps=pipe.run_from_json(kp_json)
        else:
            pipe=PosePipeline(device="cuda"); meas,fps=pipe.run(str(vpath))
            frames=[]
            for m in meas:
                persons=[]
                if m.measurement_quality!="bad":
                    kps={n:({"x":float(m.keypoints[n][0]) if m.keypoints.get(n) else 0.0,
                             "y":float(m.keypoints[n][1]) if m.keypoints.get(n) else 0.0,
                             "score":m.confidences.get(n,0.0)}) for n in JOINT_NAMES}
                    persons=[{"person_id":0,"keypoints":kps}]
                frames.append({"frame":m.frame_idx,"persons":persons})
            kp_json={"model":"RTMPose-x","keypoint_format":"COCO-17","stats":{"source_fps":fps,"video":stem},"frames":frames}
            with open(cache,"w") as f: json.dump(kp_json,f)
        eng=SwingPhaseEngine(); ann,anchors=eng.run(meas,fps,angle="face-on")
        phase_map={a.frame_idx:a.phase for a in ann}
        print(f"  sc={anchors.swing_count} addr=fr{anchors.address} top=fr{anchors.top} impact=fr{anchors.impact}")
        r=process_faceon(stem,kp_json,meas,fps,anchors,ann,phase_map)
        results.append(r)
        print(f"  head_lat: +fr{r['head_lat_pos_fr']}/{r['head_lat_pos_pct']:+.1f}%  -{r['head_lat_neg_fr']}/{r['head_lat_neg_pct']:+.1f}%")
        print(f"  head_vert: fr{r['head_vert_peak_fr']}/{r['head_vert_peak_pct']:+.1f}%")
        print(f"  elbow_min: {r['cw_win']} → fr{r['elbow_min_fr']} {r['elbow_min_deg']}deg")

    # ── 3. Summary ────────────────────────────────────────────────────────────
    lines=["# Batch2 Face-On Pipeline Results","",
           "**GT iron rule**: no fault labels. 'ok'/'wrong' are file names only.",
           f"**Date**: {datetime.datetime.now().isoformat()[:10]}","",
           "## Layer 0 Gate Summary","",
           "| Stem | Verdict | Angle | Reason (brief) |","|---|---|---|---|"]
    for stem,vlm in VLM_DATA.items():
        rec=gate.load(stem)
        r80=rec.reason[:80]+"..." if len(rec.reason)>80 else rec.reason
        lines.append(f"| {stem} | {rec.verdict} | {rec.angle} | {r80} |")
    lines+=["","## Face-On Measurements (4 PASS videos)","",
            "| Stem | addr | top | impact | ic | head_lat+ (fr/%) | head_lat- (fr/%) | head_vert (fr/%) | elbow_min (CW_win / fr / deg) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['stem']} | {r['addr_fr']} | {r['top_fr']} | {r['impact_fr']} | {r['impact_conf']:.3f} | "
                     f"fr{r['head_lat_pos_fr']}/{r['head_lat_pos_pct']:+.1f}% | "
                     f"fr{r['head_lat_neg_fr']}/{r['head_lat_neg_pct']:+.1f}% | "
                     f"fr{r['head_vert_peak_fr']}/{r['head_vert_peak_pct']:+.1f}% | "
                     f"{r['cw_win']}/fr{r['elbow_min_fr']}/{r['elbow_min_deg']}deg |")
    lines+=["","## needs_human (7 videos — paused per angle-conflict rule)","",
            "VLM angle conflicts with filename prefix. Cannot proceed without human confirmation.",
            "","| Stem | Prefix angle | VLM result | Reason |","|---|---|---|---|"]
    NH={"dtl-ok-1":"DTL","dtl-ok-2":"DTL","dtl-wrong-1":"DTL","dtl-wrong-2":"DTL","dtl-wrong-3":"DTL","fo-wrong-3":"face-on","fo-wrong-4":"face-on"}
    for stem,exp_ang in NH.items():
        rec=gate.load(stem)
        r80=rec.reason[:80]+"..." if len(rec.reason)>80 else rec.reason
        lines.append(f"| {stem} | {exp_ang} | {rec.angle} | {r80} |")
    lines+=["","## DTL processing","Not run — all 5 DTL-prefix videos are needs_human.",""]
    out=PROJ/"pipeline_output/batch2_summary.md"
    Path(out).write_text("\n".join(lines),encoding="utf-8")
    import shutil; shutil.copy(str(out),str(DESK/"batch2_summary.md"))
    print(f"\nSummary: {out}")
    prog=PROJ/"PROGRESS.log"
    with open(prog,"a") as f: f.write(f"{datetime.datetime.now().isoformat()}  batch2: 4 fo PASS processed, 7 needs_human paused\n")


if __name__=="__main__":
    main()
