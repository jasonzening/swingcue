/**
 * keypointOverlay.ts — 样板图风格旋转盘
 *
 * 关键参数（对标样板图）：
 *   肩部盘：rx = shoulderDist × 1.20，ry = rx × 0.20（非常扁平，高宽比 1:5）
 *   髋部环：rx = hipDist     × 1.00，ry = rx × 0.18（更扁）
 *   guide line：白色，长度 = rx × 1.60，严格在椭圆内
 *   center：shoulderCenter 下移 dist × 0.08（轻微下移，不要太多）
 *   angle：clamp ±12° face_on，归一化防 ±180°
 */

import type {
  OverlayElement, KeypointFrame,
  LineElement, DotElement, LabelElement,
} from '@/types/analysis';
import type { MainIssueType } from '@/types/analysis';
import type { BodyPointName, Pt } from '@/lib/golf/bodyPointSpec';
import type { ViewType } from '@/lib/golf/overlayLineSpec';

const _prevAngle: Record<string, number> = {};
let _uid = 0;
const uid   = (p: string) => `${p}-${++_uid}`;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const dist2D = (a: Pt, b: Pt) => Math.hypot(b.x - a.x, b.y - a.y);
const mid2D  = (a: Pt, b: Pt): Pt => ({ x:(a.x+b.x)/2, y:(a.y+b.y)/2, confidence:1 });

type AC = 'red' | 'green' | 'yellow' | 'white';

const mkLine  = (x1:number,y1:number,x2:number,y2:number,c:AC,w=2.5,op=0.88,layer:OverlayElement['layer']='body'): LineElement =>
  ({type:'line',id:uid('l'),x1,y1,x2,y2,color:c,strokeWidth:w,opacity:op,layer});
const mkDot   = (x:number,y:number,c:AC,r=0.008,op=0.70,layer:OverlayElement['layer']='body'): DotElement =>
  ({type:'dot',id:uid('d'),x,y,color:c,radius:r,opacity:op,layer});
const mkLabel = (x:number,y:number,text:string,c:AC='white',size=10,op=0.80): LabelElement =>
  ({type:'label',id:uid('t'),x,y,text,color:c,size,opacity:op});

/** mkEllipse — type:'ellipse' 由 renderer.drawEllipse 处理（三层霓虹发光）*/
function mkEllipse(
  cx:number, cy:number,
  rx:number, ry:number,
  angleDeg:number,
  color: AC,
  strokeWidth=5.0,
  opacity=0.92,
  layer:OverlayElement['layer']='body',
): OverlayElement {
  return {
    type: 'ellipse' as OverlayElement['type'],
    id: uid('e'),
    cx, cy, rx, ry,
    angleDeg,
    color, strokeWidth, opacity, layer,
  } as unknown as OverlayElement;
}

/** 角度归一化到 [-90, +90]，防止 atan2 返回 ±180° */
function normalizeAngle(deg: number): number {
  let a = deg;
  if (a >  90) a -= 180;
  if (a < -90) a += 180;
  return a;
}

/* ═══════════════════════════════════════════════════
   buildDisc — 旋转盘核心
   ─────────────────────────────────────────────────
   对标样板图：
   - 椭圆极度扁平（ryRatio 0.20，高宽比 1:5）
   - 椭圆要比肩宽大（rxMult 1.20）
   - guide line 白色穿越盘面
   - 中心轻微下移（dropRatio 0.08）
═══════════════════════════════════════════════════ */
function buildDisc(
  leftPt: Pt, rightPt: Pt,
  opts: {
    dropRatio:  number;
    rxMult:     number;
    rxMin:      number;
    rxMax:      number;
    ryRatio:    number;   // ← 关键：样板图约 0.20
    maxAngle:   number;
    guideRatio: number;
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
  if (dist < 0.015) return els;

  /* 圆盘几何 */
  const cx = (leftPt.x + rightPt.x) / 2;
  const cy = (leftPt.y + rightPt.y) / 2 + dist * opts.dropRatio;
  const rx = clamp(dist * opts.rxMult, opts.rxMin, opts.rxMax);
  const ry = rx * opts.ryRatio;   // ← 非常扁平

  /* 角度归一化 + clamp + 帧间平滑 */
  const rawDeg   = Math.atan2(rightPt.y - leftPt.y, rightPt.x - leftPt.x) * 180 / Math.PI;
  const normDeg  = normalizeAngle(rawDeg);
  const clampDeg = clamp(normDeg, -opts.maxAngle, opts.maxAngle);
  const prev     = _prevAngle[prevKey];
  const smoothDeg = prev !== undefined
    ? prev + clamp(clampDeg - prev, -8, 8)
    : clampDeg;
  _prevAngle[prevKey] = smoothDeg;

  /* 低 confidence → 只画肩线 */
  if (lc < 0.38 || rc < 0.38) {
    els.push(mkLine(leftPt.x,leftPt.y,rightPt.x,rightPt.y,color,1.5,0.50,layer));
    return els;
  }

  /* ① 霓虹椭圆盘（type:'ellipse'，三层发光）*/
  els.push(mkEllipse(cx, cy, rx, ry, smoothDeg, color, 5.0, 0.92, layer));

  /* ② 白色 guide line：穿越盘面，长 rx * guideRatio */
  const guideLen = rx * opts.guideRatio;
  const ar = smoothDeg * Math.PI / 180;
  const cosA = Math.cos(ar), sinA = Math.sin(ar);
  els.push(mkLine(
    cx - guideLen*cosA, cy - guideLen*sinA,
    cx + guideLen*cosA, cy + guideLen*sinA,
    'white', 2.5, 0.85, layer,
  ));

  /* ③ 端点圆点（小，锚定真实关节）*/
  if (lc > 0.40) els.push(mkDot(leftPt.x,  leftPt.y,  color, 0.008, 0.65, layer));
  if (rc > 0.40) els.push(mkDot(rightPt.x, rightPt.y, color, 0.008, 0.65, layer));

  /* ④ 标签 */
  els.push(mkLabel(cx, cy - ry * 2.0, opts.label, 'white', 10, 0.78));

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

  /* Shoulder Disc — 极度扁平，对标样板图 */
  const ls = pts.leftShoulder, rs = pts.rightShoulder;
  if (ls && rs) {
    els.push(...buildDisc(ls, rs, {
      dropRatio:  0.08,   // 轻微下移，不要太多
      rxMult:     1.20,   // 比肩宽大 20%
      rxMin:      0.15,
      rxMax:      0.40,
      ryRatio:    viewType === 'face_on' ? 0.20 : 0.14, // ← 核心：非常扁平
      maxAngle:   viewType === 'face_on' ? 12 : 35,
      guideRatio: 1.50,   // guide line 长度（在椭圆内）
      label:      'SHOULDERS',
    }, 'red', 'shoulder', 'body'));
  }

  /* Hip Ring — 更扁 */
  const lh = pts.leftHip, rh = pts.rightHip;
  if (lh && rh) {
    els.push(...buildDisc(lh, rh, {
      dropRatio:  0,
      rxMult:     1.00,
      rxMin:      0.08,
      rxMax:      0.26,
      ryRatio:    viewType === 'face_on' ? 0.18 : 0.12,
      maxAngle:   viewType === 'face_on' ? 10 : 30,
      guideRatio: 1.40,
      label:      'HIPS',
    }, 'red', 'hip', 'club'));
  }

  return els;
}

/* ── compat ── */
export function computePerspectiveDisc(args:{leftPoint:Pt;rightPoint:Pt;viewType:ViewType;kind:'shoulder'|'hip';previousAngle?:number;}){
  const{leftPoint:lp,rightPoint:rp,viewType,kind}=args;
  const dist=dist2D(lp,rp);
  const cx=(lp.x+rp.x)/2,cy=(lp.y+rp.y)/2;
  const rxM=kind==='shoulder'?1.20:1.00;
  const rx=clamp(dist*rxM,kind==='shoulder'?0.15:0.08,kind==='shoulder'?0.40:0.26);
  const ryR=viewType==='face_on'?(kind==='shoulder'?0.20:0.18):(kind==='shoulder'?0.14:0.12);
  const ry=rx*ryR;
  const rawDeg=Math.atan2(rp.y-lp.y,rp.x-lp.x)*180/Math.PI;
  const normDeg=normalizeAngle(rawDeg);
  const maxA=viewType==='face_on'?(kind==='shoulder'?12:10):(kind==='shoulder'?35:30);
  const smoothDeg=clamp(normDeg,-maxA,maxA);
  const ar=smoothDeg*Math.PI/180,cosA=Math.cos(ar),sinA=Math.sin(ar);
  const gl=rx*1.50,drop=kind==='shoulder'?dist*0.08:0;
  return{cx,cy:cy+drop,rx,ry,rotationDeg:smoothDeg,
    guideStart:{x:cx-gl*cosA,y:cy+drop-gl*sinA},
    guideEnd:{x:cx+gl*cosA,y:cy+drop+gl*sinA},
    alpha:0.92,visible:dist>0.015};
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
