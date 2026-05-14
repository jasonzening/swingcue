/**
 * keypointOverlay.ts — Shoulder Rotation Disc (AlignSnow style)
 *
 * Disc 设计规格：
 *   中心：shoulderCenter 下移 dist*0.18（包住肩膀+上胸）
 *   rx  = dist * 1.10（比肩宽超出两侧）
 *   ry  = rx  * 0.40（face_on）/ 0.30（dtl）
 *   angle = shoulder line angle, clamped ±18° (face_on) / ±35° (dtl)
 *   guide line 横贯盘面，长度 = rx * 1.80，穿越两肩点
 *   fill 半透明 + 外圈清晰
 *
 * 所有元素用现有类型：LineElement / ZoneElement / DotElement / LabelElement
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
const uid = (p: string) => `${p}-${++_uid}`;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const toRad = (d: number) => d * Math.PI / 180;
const dist2D = (a: Pt, b: Pt) => Math.hypot(b.x - a.x, b.y - a.y);
const mid2D  = (a: Pt, b: Pt): Pt => ({ x:(a.x+b.x)/2, y:(a.y+b.y)/2, confidence: 1 });

type AC = 'red' | 'green' | 'yellow' | 'white';

const mkLine = (x1:number,y1:number,x2:number,y2:number,c:AC,w=2.5,op=0.88,dash=false,layer:OverlayElement['layer']='body'): LineElement =>
  ({type:'line',id:uid('l'),x1,y1,x2,y2,color:c,strokeWidth:w,opacity:op,dashed:dash,layer});
const mkZone = (pts:Array<{x:number;y:number}>,c:AC,fillOp=0.15,op=0.92,layer:OverlayElement['layer']='body'): ZoneElement =>
  ({type:'zone',id:uid('z'),points:pts,color:c,fillOpacity:fillOp,opacity:op,layer});
const mkDot  = (x:number,y:number,c:AC,r=0.010,op=0.90,layer:OverlayElement['layer']='body'): DotElement =>
  ({type:'dot',id:uid('d'),x,y,color:c,radius:r,opacity:op,layer});
const mkLabel= (x:number,y:number,text:string,c:AC='white',size=10,op=0.80): LabelElement =>
  ({type:'label',id:uid('t'),x,y,text,color:c,size,opacity:op});

/* ── 椭圆 polygon（48 点，更圆滑）── */
function ellipsePts(cx:number,cy:number,rx:number,ry:number,deg:number,n=48): Array<{x:number;y:number}> {
  const ar=toRad(deg), cosA=Math.cos(ar), sinA=Math.sin(ar);
  const pts:Array<{x:number;y:number}>=[];
  for(let i=0;i<n;i++){
    const t=(2*Math.PI*i)/n, ex=rx*Math.cos(t), ey=ry*Math.sin(t);
    pts.push({x:cx+ex*cosA-ey*sinA, y:cy+ex*sinA+ey*cosA});
  }
  return pts;
}

/* ═══════════════════════════════════════════════════════
   buildRotationDisc
   ───────────────────────────────────────────────────────
   参数说明：
   - leftPt / rightPt: 精确关键点（左右肩/髋）
   - hipCenter: 用于计算圆盘中心下移量
   - kind: 'shoulder' | 'hip'
   - viewType
   返回 OverlayElement[]
═══════════════════════════════════════════════════════ */
function buildRotationDisc(
  leftPt:   Pt,
  rightPt:  Pt,
  hipCenterY: number | null,  // null = hip ring 自身
  kind:     'shoulder' | 'hip',
  viewType: ViewType,
  color:    AC,
  prevKey:  string,
  layer:    OverlayElement['layer'] = 'body',
): OverlayElement[] {
  const els: OverlayElement[] = [];

  const lc = leftPt.confidence  ?? 0.8;
  const rc = rightPt.confidence ?? 0.8;
  const lowConf = lc < 0.38 || rc < 0.38;

  const shCx = (leftPt.x + rightPt.x) / 2;
  const shCy = (leftPt.y + rightPt.y) / 2;
  const dist  = dist2D(leftPt, rightPt);

  if (dist < 0.02) return els;

  // ── 圆盘中心：shoulder 下移，hip 保持中点 ──
  // shoulder disc 下移 dist*0.18，让盘面包住上胸
  const dropY = kind === 'shoulder' ? dist * 0.18 : 0;
  const cx = shCx;
  const cy = shCy + dropY;

  // ── 尺寸 ──
  const rxMultiplier = kind === 'shoulder' ? 1.10 : 1.00;
  const rxMin  = kind === 'shoulder' ? 0.14 : 0.08;
  const rxMax  = kind === 'shoulder' ? 0.38 : 0.26;
  const rx = clamp(dist * rxMultiplier, rxMin, rxMax);

  const ryRatio = viewType === 'face_on'
    ? (kind === 'shoulder' ? 0.40 : 0.28)
    : (kind === 'shoulder' ? 0.30 : 0.22);
  const ry = rx * ryRatio;

  // ── 角度：shoulder line angle，clamped ──
  const rawDeg = Math.atan2(rightPt.y - leftPt.y, rightPt.x - leftPt.x) * 180 / Math.PI;
  const maxAng = viewType === 'face_on'
    ? (kind === 'shoulder' ? 18 : 15)
    : (kind === 'shoulder' ? 35 : 30);
  const clampedDeg = clamp(rawDeg, -maxAng, maxAng);

  // 帧间平滑（max 8°/frame）
  const prev = _prevAngle[prevKey];
  let smoothDeg = clampedDeg;
  if (prev !== undefined) {
    smoothDeg = prev + clamp(clampedDeg - prev, -8, 8);
  }
  _prevAngle[prevKey] = smoothDeg;

  // ── confidence 不足 → 只画关键点连线 ──
  if (lowConf) {
    els.push(mkLine(leftPt.x,leftPt.y, rightPt.x,rightPt.y, color, 1.5, 0.55, true, layer));
    return els;
  }

  // ── 椭圆填充（半透明 zone polygon）──
  const fillPts = ellipsePts(cx, cy, rx, ry, smoothDeg, 48);
  const fillOp  = kind === 'shoulder' ? (color==='red'?0.14:0.16) : (color==='red'?0.12:0.14);
  els.push(mkZone(fillPts, color, fillOp, 0.92, layer));

  // ── Guide line：横贯盘面，长 rx*1.80，方向=肩线方向 ──
  // 从圆盘中心沿肩线方向延伸，穿越两肩点并超出两侧
  const guideLen = rx * 1.80;
  const ar = toRad(smoothDeg), cosA = Math.cos(ar), sinA = Math.sin(ar);
  const gx1 = cx - guideLen * cosA, gy1 = cy - guideLen * sinA;
  const gx2 = cx + guideLen * cosA, gy2 = cy + guideLen * sinA;
  const guideW = kind === 'shoulder' ? 2.8 : 2.0;
  els.push(mkLine(gx1,gy1, gx2,gy2, color, guideW, 0.90, false, layer));

  // ── 肩/髋 关节点（锚定真实关键点）──
  els.push(mkDot(leftPt.x,  leftPt.y,  color, 0.011, 0.92, layer));
  els.push(mkDot(rightPt.x, rightPt.y, color, 0.011, 0.92, layer));

  // ── 角度标注 ──
  const tiltDeg = Math.round(Math.abs(rawDeg));
  const labelC: AC = 'white';
  const labelY = cy - ry * 1.65;
  const labelText = kind === 'shoulder'
    ? `SHOULDERS  ${tiltDeg}°`
    : `HIPS  ${tiltDeg}°`;
  els.push(mkLabel(cx, labelY, labelText, labelC, 10, 0.80));

  return els;
}

/* ═════ 关键点解析 ═════ */
export function getKeypoints(kpFrame: KeypointFrame): Partial<Record<BodyPointName, Pt>> {
  const lm = kpFrame.landmarks;
  const r: Partial<Record<BodyPointName, Pt>> = {};
  const toP = (pt?: {x:number;y:number;confidence?:number}|null) =>
    pt ? {x:pt.x, y:pt.y, confidence: pt.confidence ?? 0.8} : null;
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
  if(ls&&rs) r.shoulderCenter = mid2D(ls,rs);
  const lh=r.leftHip,rh=r.rightHip;
  if(lh&&rh) r.hipCenter = mid2D(lh,rh);
  const lw=r.leftWrist,rw=r.rightWrist;
  if(lw&&rw) r.gripCenter = mid2D(lw,rw);
  return r;
}

/* ═════ 主生成函数 ═════ */
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

  const hipCY = pts.hipCenter?.y ?? null;

  /* Shoulder Disc */
  const ls = pts.leftShoulder, rs = pts.rightShoulder;
  if (ls && rs) {
    els.push(...buildRotationDisc(
      ls, rs, hipCY, 'shoulder', viewType, 'red', 'shoulder', 'body',
    ));
  }

  /* Hip Ring */
  const lh = pts.leftHip, rh = pts.rightHip;
  if (lh && rh) {
    els.push(...buildRotationDisc(
      lh, rh, null, 'hip', viewType, 'red', 'hip', 'club',
    ));
  }

  return els;
}

/* ─── compat exports ─── */
export function computePerspectiveDisc(args:{
  leftPoint:Pt;rightPoint:Pt;viewType:ViewType;kind:'shoulder'|'hip';previousAngle?:number;
}) {
  const {leftPoint:lp,rightPoint:rp,viewType,kind,previousAngle} = args;
  const dist=dist2D(lp,rp);
  const cx=(lp.x+rp.x)/2, cy=(lp.y+rp.y)/2;
  const maxAng=viewType==='face_on'?18:35;
  const rawDeg=Math.atan2(rp.y-lp.y,rp.x-lp.x)*180/Math.PI;
  const clampedDeg=clamp(rawDeg,-maxAng,maxAng);
  const smoothDeg=previousAngle!==undefined
    ? previousAngle+clamp(clampedDeg-previousAngle,-8,8)
    : clampedDeg;
  const rxM=kind==='shoulder'?1.10:1.00;
  const rx=clamp(dist*rxM, kind==='shoulder'?0.14:0.08, kind==='shoulder'?0.38:0.26);
  const ryR=viewType==='face_on'?(kind==='shoulder'?0.40:0.28):(kind==='shoulder'?0.30:0.22);
  const ry=rx*ryR;
  const ar=toRad(smoothDeg),cosA=Math.cos(ar),sinA=Math.sin(ar);
  const guideLen=rx*1.80;
  const dropY=kind==='shoulder'?dist*0.18:0;
  return {cx,cy:cy+dropY,rx,ry,rotationDeg:smoothDeg,
    guideStart:{x:cx-guideLen*cosA,y:cy+dropY-guideLen*sinA},
    guideEnd:{x:cx+guideLen*cosA,y:cy+dropY+guideLen*sinA},
    alpha:0.88,visible:dist>0.02};
}

export function getTrackedPoint(_i:MainIssueType,_v:ViewType,kpFrame:KeypointFrame):{x:number;y:number}|null{
  const pts=getKeypoints(kpFrame);
  return pts.shoulderCenter?{x:pts.shoulderCenter.x,y:pts.shoulderCenter.y}:null;
}

export function findNearestFrame(frames:KeypointFrame[],time:number):KeypointFrame|null{
  if(!frames.length) return null;
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
