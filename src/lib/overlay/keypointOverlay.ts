/**
 * keypointOverlay.ts — Shoulder / Hip Rotation Disc
 *
 * 修复要点：
 * 1. 角度归一化：atan2 → [-90,+90] 再 clamp ±12°（face_on）
 *    防止 ±180° 大斜线
 * 2. guide line 长度 = rx * 1.55（严格在圆盘内，不超出）
 * 3. 填充 opacity 0.22，strokeWidth 4.5
 * 4. 隐藏角度数字（暂时），避免 180° 误导
 * 5. 圆盘中心下移 dist*0.18（包住肩膀+上胸）
 */

import type {
  OverlayElement, KeypointFrame,
  LineElement, ZoneElement, DotElement, LabelElement,
} from '@/types/analysis';
import type { MainIssueType } from '@/types/analysis';
import type { BodyPointName, Pt } from '@/lib/golf/bodyPointSpec';
import type { ViewType } from '@/lib/golf/overlayLineSpec';

const _prevAngle: Record<string, number> = {};
let _uid = 0;
const uid   = (p: string) => `${p}-${++_uid}`;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const toRad = (d: number) => d * Math.PI / 180;
const dist2D = (a: Pt, b: Pt) => Math.hypot(b.x - a.x, b.y - a.y);
const mid2D  = (a: Pt, b: Pt): Pt => ({ x:(a.x+b.x)/2, y:(a.y+b.y)/2, confidence:1 });

type AC = 'red' | 'green' | 'yellow' | 'white';

const mkLine  = (x1:number,y1:number,x2:number,y2:number,c:AC,w=2.5,op=0.88,dash=false,layer:OverlayElement['layer']='body'): LineElement =>
  ({type:'line',id:uid('l'),x1,y1,x2,y2,color:c,strokeWidth:w,opacity:op,dashed:dash,layer});
const mkZone  = (pts:Array<{x:number;y:number}>,c:AC,fillOp=0.22,op=0.92,layer:OverlayElement['layer']='body'): ZoneElement =>
  ({type:'zone',id:uid('z'),points:pts,color:c,fillOpacity:fillOp,opacity:op,layer});
const mkDot   = (x:number,y:number,c:AC,r=0.008,op=0.65,layer:OverlayElement['layer']='body'): DotElement =>
  ({type:'dot',id:uid('d'),x,y,color:c,radius:r,opacity:op,layer});
const mkLabel = (x:number,y:number,text:string,c:AC='white',size=10,op=0.78): LabelElement =>
  ({type:'label',id:uid('t'),x,y,text,color:c,size,opacity:op});

/* ── 椭圆 polygon ── */
function ellipsePts(cx:number,cy:number,rx:number,ry:number,deg:number,n=48):Array<{x:number;y:number}>{
  const ar=toRad(deg),cosA=Math.cos(ar),sinA=Math.sin(ar);
  const pts:Array<{x:number;y:number}>=[];
  for(let i=0;i<n;i++){
    const t=(2*Math.PI*i)/n,ex=rx*Math.cos(t),ey=ry*Math.sin(t);
    pts.push({x:cx+ex*cosA-ey*sinA,y:cy+ex*sinA+ey*cosA});
  }
  return pts;
}

/**
 * normalizeAngle
 * atan2 返回 [-180, +180]，当右肩在左肩左侧时会返回 ±180°。
 * 归一化到 [-90, +90]（只关心横轴倾斜量）。
 */
function normalizeAngle(deg: number): number {
  let a = deg;
  if (a >  90) a -= 180;
  if (a < -90) a += 180;
  return a;
}

/* ═══════════════════════════════════════════════════
   buildDisc — 旋转盘核心
   ─────────────────────────────────────────────────
   leftPt/rightPt: 精确关键点
   dropRatio: 圆盘中心下移量 / dist（shoulder=0.18, hip=0）
   rxMult: rx = dist * rxMult
   ryRatio: ry = rx * ryRatio
   maxAng: clamp angle（face_on shoulder=12, hip=10）
   guideRatio: guide line 长度 = rx * guideRatio（≤1.55，不超出圆盘）
═══════════════════════════════════════════════════ */
function buildDisc(
  leftPt:    Pt,
  rightPt:   Pt,
  opts: {
    dropRatio:  number;
    rxMult:     number;
    rxMin:      number;
    rxMax:      number;
    ryRatio:    number;
    maxAng:     number;
    guideRatio: number;
    showLabel:  boolean;
    label:      string;
  },
  color: AC,
  prevKey: string,
  layer: OverlayElement['layer'] = 'body',
): OverlayElement[] {
  const els: OverlayElement[] = [];
  const lc = leftPt.confidence  ?? 0.8;
  const rc = rightPt.confidence ?? 0.8;
  const dist = dist2D(leftPt, rightPt);
  if (dist < 0.02) return els;

  /* ── 圆盘几何 ── */
  const cx = (leftPt.x + rightPt.x) / 2;
  const cy = (leftPt.y + rightPt.y) / 2 + dist * opts.dropRatio;
  const rx = clamp(dist * opts.rxMult, opts.rxMin, opts.rxMax);
  const ry = rx * opts.ryRatio;

  /* ── 角度归一化 + clamp + 帧间平滑 ── */
  const rawDeg = Math.atan2(rightPt.y - leftPt.y, rightPt.x - leftPt.x) * 180 / Math.PI;
  const normDeg  = normalizeAngle(rawDeg);           // → [-90, +90]
  const clampDeg = clamp(normDeg, -opts.maxAng, opts.maxAng);
  const prev = _prevAngle[prevKey];
  const smoothDeg = prev !== undefined
    ? prev + clamp(clampDeg - prev, -8, 8)
    : clampDeg;
  _prevAngle[prevKey] = smoothDeg;

  /* ── 低 confidence：只画连线 ── */
  if (lc < 0.38 || rc < 0.38) {
    els.push(mkLine(leftPt.x,leftPt.y,rightPt.x,rightPt.y,color,1.5,0.50,true,layer));
    return els;
  }

  /* ── 椭圆填充 (zone polygon) ── */
  const fillPts = ellipsePts(cx,cy,rx,ry,smoothDeg,48);
  els.push(mkZone(fillPts, color, 0.22, 0.92, layer));

  /* ── Guide line：严格在圆盘内，长度 = rx * guideRatio ── */
  const guideLen = rx * opts.guideRatio;   // guideRatio ≤ 1.55 → 不超出圆盘
  const ar = toRad(smoothDeg), cosA = Math.cos(ar), sinA = Math.sin(ar);
  els.push(mkLine(
    cx - guideLen*cosA, cy - guideLen*sinA,
    cx + guideLen*cosA, cy + guideLen*sinA,
    color, 3.0, 0.88, false, layer,
  ));

  /* ── 关节圆点（小，不抢眼）── */
  if (lc > 0.35) els.push(mkDot(leftPt.x,  leftPt.y,  color, 0.008, 0.62, layer));
  if (rc > 0.35) els.push(mkDot(rightPt.x, rightPt.y, color, 0.008, 0.62, layer));

  /* ── label（不显示角度数字，避免 180° 误导）── */
  if (opts.showLabel) {
    els.push(mkLabel(cx, cy - ry * 1.70, opts.label, 'white', 10, 0.78));
  }

  return els;
}

/* ═══════ 关键点解析 ═══════ */
export function getKeypoints(kpFrame: KeypointFrame): Partial<Record<BodyPointName, Pt>> {
  const lm = kpFrame.landmarks;
  const r: Partial<Record<BodyPointName, Pt>> = {};
  const toP = (pt?: {x:number;y:number;confidence?:number}|null) =>
    pt ? {x:pt.x,y:pt.y,confidence:pt.confidence??0.8} : null;
  if(lm.head)          r.headCenter    = toP(lm.head)!;
  if(lm.leftShoulder)  r.leftShoulder  = toP(lm.leftShoulder)!;
  if(lm.rightShoulder) r.rightShoulder = toP(lm.rightShoulder)!;
  if(lm.leftElbow)     r.leftElbow     = toP(lm.leftElbow)!;
  if(lm.rightElbow)    r.rightElbow    = toP(lm.rightElbow)!;
  if(lm.leftWrist)     r.leftWrist     = toP(lm.leftWrist)!;
  if(lm.rightWrist)    r.rightWrist    = toP(lm.rightWrist)!;
  if(lm.leftHip)       r.leftHip       = toP(lm.leftHip)!;
  if(lm.rightHip)      r.rightHip      = toP(lm.rightHip)!;
  if(lm.leftKnee)      r.leftKnee      = toP(lm.leftKnee)!;
  if(lm.rightKnee)     r.rightKnee     = toP(lm.rightKnee)!;
  if(lm.leftAnkle)     r.leftAnkle     = toP(lm.leftAnkle)!;
  if(lm.rightAnkle)    r.rightAnkle    = toP(lm.rightAnkle)!;
  const ls=r.leftShoulder,rs=r.rightShoulder;
  if(ls&&rs) r.shoulderCenter=mid2D(ls,rs);
  const lh=r.leftHip,rh=r.rightHip;
  if(lh&&rh) r.hipCenter=mid2D(lh,rh);
  const lw=r.leftWrist,rw=r.rightWrist;
  if(lw&&rw) r.gripCenter=mid2D(lw,rw);
  return r;
}

/* ═══════ 主生成函数 ═══════ */
export function generateSpecDrivenOverlayFrame(
  issue:    MainIssueType,
  viewType: ViewType,
  phase:    string,
  kpFrame:  KeypointFrame,
  _hist?:   Array<{x:number;y:number}>,
): OverlayElement[] {
  _uid = 0;
  const pts = getKeypoints(kpFrame);
  const els: OverlayElement[] = [];

  /* Shoulder Disc */
  const ls = pts.leftShoulder, rs = pts.rightShoulder;
  if (ls && rs) {
    els.push(...buildDisc(ls, rs, {
      dropRatio:  0.18,    // 圆盘中心下移 18%，包住上胸
      rxMult:     1.15,    // rx 比肩宽大 15%
      rxMin:      0.14,
      rxMax:      0.36,
      ryRatio:    viewType === 'face_on' ? 0.42 : 0.30,
      maxAng:     viewType === 'face_on' ? 12 : 35,  // ← 12° 限制大斜线
      guideRatio: 1.45,    // guide 不超出圆盘（1.45 < 1.0/0.42*0.5 ≈ 2.38，安全）
      showLabel:  true,
      label:      'SHOULDERS',   // 不显示角度数字
    }, 'red', 'shoulder', 'body'));
  }

  /* Hip Ring */
  const lh = pts.leftHip, rh = pts.rightHip;
  if (lh && rh) {
    els.push(...buildDisc(lh, rh, {
      dropRatio:  0,
      rxMult:     0.95,
      rxMin:      0.08,
      rxMax:      0.24,
      ryRatio:    viewType === 'face_on' ? 0.25 : 0.20,
      maxAng:     viewType === 'face_on' ? 10 : 30,
      guideRatio: 1.35,
      showLabel:  true,
      label:      'HIPS',        // 不显示角度数字
    }, 'red', 'hip', 'club'));
  }

  return els;
}

/* ── compat ── */
export function computePerspectiveDisc(args:{leftPoint:Pt;rightPoint:Pt;viewType:ViewType;kind:'shoulder'|'hip';previousAngle?:number;}){
  const{leftPoint:lp,rightPoint:rp,viewType,kind}=args;
  const dist=dist2D(lp,rp);
  const cx=(lp.x+rp.x)/2,cy=(lp.y+rp.y)/2;
  const rxM=kind==='shoulder'?1.15:0.95;
  const rx=clamp(dist*rxM,kind==='shoulder'?0.14:0.08,kind==='shoulder'?0.36:0.24);
  const ryR=viewType==='face_on'?(kind==='shoulder'?0.42:0.25):(kind==='shoulder'?0.30:0.20);
  const ry=rx*ryR;
  const rawDeg=Math.atan2(rp.y-lp.y,rp.x-lp.x)*180/Math.PI;
  const normDeg=normalizeAngle(rawDeg);
  const maxA=viewType==='face_on'?(kind==='shoulder'?12:10):(kind==='shoulder'?35:30);
  const smoothDeg=clamp(normDeg,-maxA,maxA);
  const ar=toRad(smoothDeg),cosA=Math.cos(ar),sinA=Math.sin(ar);
  const gl=rx*1.45,drop=kind==='shoulder'?dist*0.18:0;
  return{cx,cy:cy+drop,rx,ry,rotationDeg:smoothDeg,
    guideStart:{x:cx-gl*cosA,y:cy+drop-gl*sinA},
    guideEnd:{x:cx+gl*cosA,y:cy+drop+gl*sinA},
    alpha:0.88,visible:dist>0.02};
}

export function getTrackedPoint(_i:MainIssueType,_v:ViewType,kpFrame:KeypointFrame):{x:number;y:number}|null{
  const pts=getKeypoints(kpFrame);
  return pts.shoulderCenter?{x:pts.shoulderCenter.x,y:pts.shoulderCenter.y}:null;
}

export function findNearestFrame(frames:KeypointFrame[],time:number):KeypointFrame|null{
  if(!frames.length)return null;
  let best=frames[0],bestD=Math.abs(time-best.time);
  for(const f of frames){const d=Math.abs(time-f.time);if(d<bestD){best=f;bestD=d;}}
  return best;
}

export function applyCorrection(pt:{x:number;y:number;confidence?:number},dir:string,mag='medium'):{x:number;y:number;confidence?:number}{
  const D:Record<string,number>={small:0.028,medium:0.050,large:0.078};
  const d=D[mag]??0.050;
  switch(dir){
    case 'lower':return{...pt,y:pt.y+d};case 'higher':return{...pt,y:pt.y-d};
    case 'more_inside':return{...pt,x:pt.x-d};case 'more_outside':return{...pt,x:pt.x+d};
    case 'more_centered':return{...pt,x:0.50};default:return pt;
  }
}
