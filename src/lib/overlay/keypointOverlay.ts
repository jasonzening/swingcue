/**
 * keypointOverlay.ts — Shoulder / Hip Anchored Disc
 *
 * 硬规则（不可违反）：
 *  guideLineStart = left keypoint  (精确）
 *  guideLineEnd   = right keypoint (精确）
 *  disc center    = midpoint(left, right)
 *  disc long axis = angle(left → right)，clamped ±30° face_on / ±45° dtl
 *  disc rx        = dist(left, right) × 0.55  → 椭圆端点刚好包住两个关键点
 *  disc ry        = rx × heightRatio (shoulder 0.32, hip 0.28)
 *
 * 帧间平滑：max 8° delta per frame（防 MediaPipe 抖动）
 * confidence < 0.4 → 只画连线，不画圆盘
 */

import type {
  OverlayElement, KeypointFrame,
  LineElement, ZoneElement, DotElement, LabelElement,
} from '@/types/analysis';
import type { MainIssueType } from '@/types/analysis';
import type { BodyPointName, Pt } from '@/lib/golf/bodyPointSpec';
import type { ViewType } from '@/lib/golf/overlayLineSpec';

/* ── 帧间平滑状态 ── */
const _prevAngle: Record<string, number> = {};

let _uid = 0;
const uid = (p: string) => `${p}-${++_uid}`;

const clamp  = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const toRad  = (deg: number) => deg * Math.PI / 180;
const dist2D = (a: Pt, b: Pt) => Math.hypot(b.x - a.x, b.y - a.y);
const mid2D  = (a: Pt, b: Pt): Pt => ({ x:(a.x+b.x)/2, y:(a.y+b.y)/2, confidence:1 });

type AC = 'red' | 'green' | 'yellow' | 'white'; // AllowedColor

const mkLine = (x1:number,y1:number,x2:number,y2:number, c:AC, w=2.0, op=0.88, dash=false, layer:OverlayElement['layer']='body'): LineElement =>
  ({type:'line',id:uid('l'),x1,y1,x2,y2,color:c,strokeWidth:w,opacity:op,dashed:dash,layer});
const mkZone = (pts:Array<{x:number;y:number}>, c:AC, fillOp=0.14, op=0.85, layer:OverlayElement['layer']='body'): ZoneElement =>
  ({type:'zone',id:uid('z'),points:pts,color:c,fillOpacity:fillOp,opacity:op,layer});
const mkDot  = (x:number,y:number, c:AC, r=0.009, op=0.90, layer:OverlayElement['layer']='body'): DotElement =>
  ({type:'dot',id:uid('d'),x,y,color:c,radius:r,opacity:op,layer});
const mkLabel= (x:number,y:number,text:string,c:AC='white',size=9,op=0.72): LabelElement =>
  ({type:'label',id:uid('t'),x,y,text,color:c,size,opacity:op});

/* ── 椭圆 polygon（40 点）── */
function ellipsePts(cx:number,cy:number,rx:number,ry:number,deg:number,n=40): Array<{x:number;y:number}> {
  const ar=toRad(deg), cosA=Math.cos(ar), sinA=Math.sin(ar);
  const pts:Array<{x:number;y:number}>=[];
  for(let i=0;i<n;i++){
    const t=(2*Math.PI*i)/n, ex=rx*Math.cos(t), ey=ry*Math.sin(t);
    pts.push({x:cx+ex*cosA-ey*sinA, y:cy+ex*sinA+ey*cosA});
  }
  return pts;
}

/* ══════════════════════════════════════════════════
   buildAnchoredDisc
   ──────────────────────────────────────────────────
   所有几何量严格从 leftPoint / rightPoint 推导。
   guideLineStart / guideLineEnd = 原始关键点，不做任何偏移。
══════════════════════════════════════════════════ */
interface AnchoredDisc {
  cx: number; cy: number;
  rx: number; ry: number;
  rotationDeg: number;
  guideLineStart: Pt;   // = leftPoint（精确，无偏移）
  guideLineEnd:   Pt;   // = rightPoint（精确，无偏移）
  alpha: number;
  valid: boolean;
  tooNarrow: boolean;   // confidence 不足时只画线
}

function buildAnchoredDisc(
  leftPoint:  Pt,
  rightPoint: Pt,
  options: {
    heightRatio: number;   // ry = rx × heightRatio
    maxAngleDeg: number;   // clamp rotation
    prevAngleKey: string;  // 帧间平滑 key
  },
): AnchoredDisc {
  const FAIL: AnchoredDisc = {
    cx:0,cy:0,rx:0,ry:0,rotationDeg:0,
    guideLineStart:leftPoint,guideLineEnd:rightPoint,
    alpha:0,valid:false,tooNarrow:false,
  };

  const lc = leftPoint.confidence  ?? 0.8;
  const rc = rightPoint.confidence ?? 0.8;

  // confidence 太低 → 不画圆盘（tooNarrow=true 只画线）
  const tooNarrow = lc < 0.40 || rc < 0.40;

  // 中心 = 精确中点
  const cx = (leftPoint.x + rightPoint.x) / 2;
  const cy = (leftPoint.y + rightPoint.y) / 2;

  // 两点距离
  const dist = dist2D(leftPoint, rightPoint);
  if (dist < 0.01) return FAIL; // 两点几乎重合，跳过

  // rx = dist × 0.55：椭圆半长轴 ≈ 两点间距的一半 × 1.1
  // 这样椭圆端点正好超出两个关键点少许，视觉上包住双点
  const rx = clamp(dist * 0.55, 0.06, 0.30);

  // ry = rx × heightRatio
  const ry = clamp(rx * options.heightRatio, 0.012, 0.12);

  // 旋转角 = left→right 方向（弧度转度）
  const rawDeg = Math.atan2(
    rightPoint.y - leftPoint.y,
    rightPoint.x - leftPoint.x,
  ) * 180 / Math.PI;

  // clamp ±maxAngleDeg
  const clampedDeg = clamp(rawDeg, -options.maxAngleDeg, options.maxAngleDeg);

  // 帧间平滑：max delta 8°
  const prev = _prevAngle[options.prevAngleKey];
  let smoothDeg = clampedDeg;
  if (prev !== undefined) {
    const delta = clamp(clampedDeg - prev, -8, 8);
    smoothDeg = prev + delta;
  }
  _prevAngle[options.prevAngleKey] = smoothDeg;

  const alpha = tooNarrow ? 0 : 0.82;

  return {
    cx, cy, rx, ry,
    rotationDeg: smoothDeg,
    guideLineStart: leftPoint,   // ← 精确关键点
    guideLineEnd:   rightPoint,  // ← 精确关键点
    alpha,
    valid: true,
    tooNarrow,
  };
}

/* ── 把 AnchoredDisc 转成 OverlayElement 列表 ── */
function discToElements(
  disc: AnchoredDisc,
  color: AC,
  label: string,
  layer: OverlayElement['layer'] = 'body',
): OverlayElement[] {
  if (!disc.valid) return [];
  const {cx,cy,rx,ry,rotationDeg,guideLineStart:gS,guideLineEnd:gE,alpha,tooNarrow} = disc;
  const els: OverlayElement[] = [];

  // ① 精确横杠：left keypoint → right keypoint（硬规则）
  els.push(mkLine(gS.x,gS.y, gE.x,gE.y, color, 2.2, alpha, false, layer));

  // ② 端点关节圆点（绑定在真实关键点上）
  if ((gS.confidence ?? 0.8) > 0.35) els.push(mkDot(gS.x,gS.y, color, 0.010, alpha, layer));
  if ((gE.confidence ?? 0.8) > 0.35) els.push(mkDot(gE.x,gE.y, color, 0.010, alpha, layer));

  // confidence 不足 → 只画线，不画圆盘
  if (tooNarrow) return els;

  // ③ 椭圆（zone polygon）：中心=midpoint，长轴=两点连线方向
  const pts = ellipsePts(cx,cy,rx,ry,rotationDeg,40);
  const fillOp = color==='red' ? 0.14 : 0.16;
  els.push(mkZone(pts, color, fillOp, alpha * 0.95, layer));

  // ④ label（角度）
  els.push(mkLabel(cx, cy - ry*1.8, label + '  ' + Math.round(Math.abs(rotationDeg)) + '°', 'white', 9, 0.70));

  return els;
}

/* ══════════════════════════════════════════════════
   关键点解析
══════════════════════════════════════════════════ */
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

/* ══════════════════════════════════════════════════
   主生成函数
══════════════════════════════════════════════════ */
export function generateSpecDrivenOverlayFrame(
  issue: MainIssueType,
  viewType: ViewType,
  phase: string,
  kpFrame: KeypointFrame,
  _historyPts?: Array<{x:number;y:number}>,
): OverlayElement[] {
  _uid = 0;
  const pts = getKeypoints(kpFrame);
  const els: OverlayElement[] = [];
  const maxAng = viewType==='face_on' ? 30 : 45;

  /* Shoulder Disc */
  const ls=pts.leftShoulder, rs=pts.rightShoulder;
  if(ls&&rs){
    const disc = buildAnchoredDisc(ls, rs, {
      heightRatio: 0.32,      // ry = rx × 0.32
      maxAngleDeg: maxAng,
      prevAngleKey: 'shoulder',
    });
    els.push(...discToElements(disc, 'red', 'SHOULDERS', 'body'));
  }

  /* Hip Ring */
  const lh=pts.leftHip, rh=pts.rightHip;
  if(lh&&rh){
    const disc = buildAnchoredDisc(lh, rh, {
      heightRatio: 0.28,      // ry = rx × 0.28（比肩部更扁）
      maxAngleDeg: maxAng,
      prevAngleKey: 'hip',
    });
    els.push(...discToElements(disc, 'red', 'HIPS', 'club'));
  }

  return els;
}

/* ── 兼容性导出 ── */
export function computePerspectiveDisc(args: {
  leftPoint: Pt; rightPoint: Pt;
  viewType: ViewType; kind: 'shoulder'|'hip'; previousAngle?: number;
}) {
  const {leftPoint:lp,rightPoint:rp,viewType,kind,previousAngle} = args;
  const maxAng = viewType==='face_on' ? 30 : 45;
  return buildAnchoredDisc(lp, rp, {
    heightRatio: kind==='shoulder' ? 0.32 : 0.28,
    maxAngleDeg: maxAng,
    prevAngleKey: kind,
  });
}

export function getTrackedPoint(
  _i:MainIssueType,_v:ViewType,kpFrame:KeypointFrame,
):{x:number;y:number}|null{
  const pts=getKeypoints(kpFrame);
  return pts.shoulderCenter?{x:pts.shoulderCenter.x,y:pts.shoulderCenter.y}:null;
}

export function findNearestFrame(frames:KeypointFrame[],time:number):KeypointFrame|null{
  if(!frames.length) return null;
  let best=frames[0],bestD=Math.abs(time-best.time);
  for(const f of frames){const d=Math.abs(time-f.time);if(d<bestD){best=f;bestD=d;}}
  return best;
}

export function applyCorrection(
  pt:{x:number;y:number;confidence?:number},dir:string,mag='medium',
):{x:number;y:number;confidence?:number}{
  const D:Record<string,number>={small:0.028,medium:0.050,large:0.078};
  const d=D[mag]??0.050;
  switch(dir){
    case 'lower':         return{...pt,y:pt.y+d};
    case 'higher':        return{...pt,y:pt.y-d};
    case 'more_inside':   return{...pt,x:pt.x-d};
    case 'more_outside':  return{...pt,x:pt.x+d};
    case 'more_centered': return{...pt,x:0.50};
    default:              return pt;
  }
}
