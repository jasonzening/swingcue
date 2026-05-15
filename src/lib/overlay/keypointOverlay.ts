/**
 * keypointOverlay.ts — 旋转盘（对标样板图）
 *
 * 核心原则：
 *   圆盘中心 = 左右肩中点（不下移！）
 *   guide line 从 leftPt → rightPt 沿方向延伸，经过两个真实肩点
 *   圆盘绕脊椎/颈部中轴旋转
 *
 * guide line 画法：
 *   方向 = leftPt → rightPt
 *   端点 = 肩点位置再往外延伸 extraExt（样板图中线超出肩点）
 *   这样白线一定经过两个肩点
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

const mkLine  = (x1:number,y1:number,x2:number,y2:number,c:AC,w=2.5,op=0.90,layer:OverlayElement['layer']='body'): LineElement =>
  ({type:'line',id:uid('l'),x1,y1,x2,y2,color:c,strokeWidth:w,opacity:op,layer});
const mkDot   = (x:number,y:number,c:AC,r=0.009,op=0.72,layer:OverlayElement['layer']='body'): DotElement =>
  ({type:'dot',id:uid('d'),x,y,color:c,radius:r,opacity:op,layer});
const mkLabel = (x:number,y:number,text:string,c:AC='white',size=10,op=0.80): LabelElement =>
  ({type:'label',id:uid('t'),x,y,text,color:c,size,opacity:op});

function mkEllipse(
  cx:number, cy:number,
  rx:number, ry:number,
  angleDeg:number,
  color: AC, strokeWidth=5.0, opacity=0.92,
  layer:OverlayElement['layer']='body',
): OverlayElement {
  return {
    type: 'ellipse' as OverlayElement['type'],
    id: uid('e'), cx, cy, rx, ry, angleDeg,
    color, strokeWidth, opacity, layer,
  } as unknown as OverlayElement;
}

function normalizeAngle(deg: number): number {
  let a = deg;
  if (a >  90) a -= 180;
  if (a < -90) a += 180;
  return a;
}

/* ═══════════════════════════════════════════════════
   buildDisc
   ─────────────────────────────────────────────────
   圆盘几何原则：
   1. cx/cy = mid(leftPt, rightPt) — 不下移，让guide line经过肩点
   2. rx = dist * rxMult — 椭圆长轴（比肩宽）
   3. ry = rx * ryRatio — 椭圆短轴（很扁）
   4. guide line 从 leftPt 出发，沿方向延伸 extraRatio 至 rightPt 再延伸
      → 保证经过两个真实关键点
═══════════════════════════════════════════════════ */
function buildDisc(
  leftPt: Pt, rightPt: Pt,
  opts: {
    rxMult:     number;
    rxMin:      number;
    rxMax:      number;
    ryRatio:    number;
    maxAngle:   number;
    extraRatio: number;   // guide line 在肩点外延伸 = dist * extraRatio
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

  // 圆盘中心 = 精确肩点中点（不下移）
  const cx = (leftPt.x + rightPt.x) / 2;
  const cy = (leftPt.y + rightPt.y) / 2;

  // 椭圆尺寸
  const rx = clamp(dist * opts.rxMult, opts.rxMin, opts.rxMax);
  const ry = rx * opts.ryRatio;

  // 角度
  const rawDeg   = Math.atan2(rightPt.y - leftPt.y, rightPt.x - leftPt.x) * 180 / Math.PI;
  const normDeg  = normalizeAngle(rawDeg);
  const clampDeg = clamp(normDeg, -opts.maxAngle, opts.maxAngle);
  const prev     = _prevAngle[prevKey];
  const smoothDeg = prev !== undefined
    ? prev + clamp(clampDeg - prev, -8, 8)
    : clampDeg;
  _prevAngle[prevKey] = smoothDeg;

  if (lc < 0.38 || rc < 0.38) {
    els.push(mkLine(leftPt.x,leftPt.y,rightPt.x,rightPt.y,color,1.5,0.50,layer));
    return els;
  }

  // 椭圆
  els.push(mkEllipse(cx, cy, rx, ry, smoothDeg, color, 5.0, 0.92, layer));

  // guide line：从 leftPt → rightPt 方向，两端各额外延伸 extraRatio * dist
  // 这样白线一定经过 leftPt 和 rightPt 两个真实肩点
  const ar = smoothDeg * Math.PI / 180;
  const cosA = Math.cos(ar), sinA = Math.sin(ar);
  const extra = dist * opts.extraRatio;  // 肩点外延伸长度
  // guide line 端点 = 肩点位置 ± extra
  const gx1 = leftPt.x  - extra * cosA;
  const gy1 = leftPt.y  - extra * sinA;
  const gx2 = rightPt.x + extra * cosA;
  const gy2 = rightPt.y + extra * sinA;
  els.push(mkLine(gx1,gy1, gx2,gy2, 'white', 2.5, 0.88, layer));

  // 端点圆点
  if (lc > 0.40) els.push(mkDot(leftPt.x,  leftPt.y,  color, 0.010, 0.80, layer));
  if (rc > 0.40) els.push(mkDot(rightPt.x, rightPt.y, color, 0.010, 0.80, layer));

  // label
  els.push(mkLabel(cx, cy - ry * 2.2, opts.label, 'white', 10, 0.78));

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
      rxMult:     1.85,
      rxMin:      0.20,
      rxMax:      0.50,
      ryRatio:    viewType === 'face_on' ? 0.20 : 0.14,
      maxAngle:   viewType === 'face_on' ? 12 : 35,
      extraRatio: 0.20,  // guide line 在肩点外延伸 20% 肩宽（样板图中线超出）
      label:      'SHOULDERS',
    }, 'red', 'shoulder', 'body'));
  }

  /* Hip Ring */
  const lh = pts.leftHip, rh = pts.rightHip;
  if (lh && rh) {
    els.push(...buildDisc(lh, rh, {
      rxMult:     1.50,
      rxMin:      0.12,
      rxMax:      0.38,
      ryRatio:    viewType === 'face_on' ? 0.18 : 0.12,
      maxAngle:   viewType === 'face_on' ? 10 : 30,
      extraRatio: 0.15,
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
  const rxM=kind==='shoulder'?1.85:1.50;
  const rx=clamp(dist*rxM,kind==='shoulder'?0.20:0.12,kind==='shoulder'?0.50:0.38);
  const ryR=viewType==='face_on'?(kind==='shoulder'?0.20:0.18):(kind==='shoulder'?0.14:0.12);
  const ry=rx*ryR;
  const rawDeg=Math.atan2(rp.y-lp.y,rp.x-lp.x)*180/Math.PI;
  const normDeg=normalizeAngle(rawDeg);
  const maxA=viewType==='face_on'?(kind==='shoulder'?12:10):(kind==='shoulder'?35:30);
  const smoothDeg=clamp(normDeg,-maxA,maxA);
  const ar=smoothDeg*Math.PI/180,cosA=Math.cos(ar),sinA=Math.sin(ar);
  const ext=dist*0.20;
  return{cx,cy,rx,ry,rotationDeg:smoothDeg,
    guideStart:{x:lp.x-ext*cosA,y:lp.y-ext*sinA},
    guideEnd:{x:rp.x+ext*cosA,y:rp.y+ext*sinA},
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
