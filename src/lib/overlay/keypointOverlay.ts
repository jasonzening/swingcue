/**
 * keypointOverlay.ts â æ ·æ¿å¾é£æ ¼æè½¬ç
 *
 * å³é®åæ°ï¼å¯¹æ æ ·æ¿å¾ï¼ï¼
 *   è©é¨çï¼rx = shoulderDist Ã 1.20ï¼ry = rx Ã 0.20ï¼éå¸¸æå¹³ï¼é«å®½æ¯ 1:5ï¼
 *   é«é¨ç¯ï¼rx = hipDist     Ã 1.00ï¼ry = rx Ã 0.18ï¼æ´æï¼
 *   guide lineï¼ç½è²ï¼é¿åº¦ = rx Ã 1.60ï¼ä¸¥æ ¼å¨æ¤­åå
 *   centerï¼shoulderCenter ä¸ç§» dist Ã 0.08ï¼è½»å¾®ä¸ç§»ï¼ä¸è¦å¤ªå¤ï¼
 *   angleï¼clamp Â±12Â° face_onï¼å½ä¸åé² Â±180Â°
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

/** mkEllipse â type:'ellipse' ç± renderer.drawEllipse å¤çï¼ä¸å±éè¹ååï¼*/
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

/** è§åº¦å½ä¸åå° [-90, +90]ï¼é²æ­¢ atan2 è¿å Â±180Â° */
function normalizeAngle(deg: number): number {
  let a = deg;
  if (a >  90) a -= 180;
  if (a < -90) a += 180;
  return a;
}

/* âââââââââââââââââââââââââââââââââââââââââââââââââââ
   buildDisc â æè½¬çæ ¸å¿
   âââââââââââââââââââââââââââââââââââââââââââââââââ
   å¯¹æ æ ·æ¿å¾ï¼
   - æ¤­åæåº¦æå¹³ï¼ryRatio 0.20ï¼é«å®½æ¯ 1:5ï¼
   - æ¤­åè¦æ¯è©å®½å¤§ï¼rxMult 1.20ï¼
   - guide line ç½è²ç©¿è¶çé¢
   - ä¸­å¿è½»å¾®ä¸ç§»ï¼dropRatio 0.08ï¼
âââââââââââââââââââââââââââââââââââââââââââââââââââ */
function buildDisc(
  leftPt: Pt, rightPt: Pt,
  opts: {
    dropRatio:  number;
    rxMult:     number;
    rxMin:      number;
    rxMax:      number;
    ryRatio:    number;   // â å³é®ï¼æ ·æ¿å¾çº¦ 0.20
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

  /* åçå ä½ */
  const cx = (leftPt.x + rightPt.x) / 2;
  const cy = (leftPt.y + rightPt.y) / 2 + dist * opts.dropRatio;
  const rx = clamp(dist * opts.rxMult, opts.rxMin, opts.rxMax);
  const ry = rx * opts.ryRatio;   // â éå¸¸æå¹³

  /* è§åº¦å½ä¸å + clamp + å¸§é´å¹³æ» */
  const rawDeg   = Math.atan2(rightPt.y - leftPt.y, rightPt.x - leftPt.x) * 180 / Math.PI;
  const normDeg  = normalizeAngle(rawDeg);
  const clampDeg = clamp(normDeg, -opts.maxAngle, opts.maxAngle);
  const prev     = _prevAngle[prevKey];
  const smoothDeg = prev !== undefined
    ? prev + clamp(clampDeg - prev, -8, 8)
    : clampDeg;
  _prevAngle[prevKey] = smoothDeg;

  /* ä½ confidence â åªç»è©çº¿ */
  if (lc < 0.38 || rc < 0.38) {
    els.push(mkLine(leftPt.x,leftPt.y,rightPt.x,rightPt.y,color,1.5,0.50,layer));
    return els;
  }

  /* â  éè¹æ¤­åçï¼type:'ellipse'ï¼ä¸å±ååï¼*/
  els.push(mkEllipse(cx, cy, rx, ry, smoothDeg, color, 5.0, 0.92, layer));

  /* â¡ ç½è² guide lineï¼ç©¿è¶çé¢ï¼é¿ rx * guideRatio */
  const guideLen = rx * opts.guideRatio;
  const ar = smoothDeg * Math.PI / 180;
  const cosA = Math.cos(ar), sinA = Math.sin(ar);
  els.push(mkLine(
    cx - guideLen*cosA, cy - guideLen*sinA,
    cx + guideLen*cosA, cy + guideLen*sinA,
    'white', 2.5, 0.85, layer,
  ));

  /* â¢ ç«¯ç¹åç¹ï¼å°ï¼éå®çå®å³èï¼*/
  if (lc > 0.40) els.push(mkDot(leftPt.x,  leftPt.y,  color, 0.008, 0.65, layer));
  if (rc > 0.40) els.push(mkDot(rightPt.x, rightPt.y, color, 0.008, 0.65, layer));

  /* â£ æ ç­¾ */
  els.push(mkLabel(cx, cy - ry * 2.0, opts.label, 'white', 10, 0.78));

  return els;
}

/* âââââââ å³é®ç¹è§£æ âââââââ */
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

/* âââââââ ä¸»çæå½æ° âââââââ */
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

  /* Shoulder Disc â æåº¦æå¹³ï¼å¯¹æ æ ·æ¿å¾ */
  const ls = pts.leftShoulder, rs = pts.rightShoulder;
  if (ls && rs) {
    els.push(...buildDisc(ls, rs, {
      dropRatio:  0.08,   // è½»å¾®ä¸ç§»ï¼ä¸è¦å¤ªå¤
      rxMult:     1.20,   // æ¯è©å®½å¤§ 20%
      rxMin:      0.15,
      rxMax:      0.40,
      ryRatio:    viewType === 'face_on' ? 0.20 : 0.14, // â æ ¸å¿ï¼éå¸¸æå¹³
      maxAngle:   viewType === 'face_on' ? 12 : 35,
      guideRatio: 1.50,   // guide line é¿åº¦ï¼å¨æ¤­ååï¼
      label:      'SHOULDERS',
    }, 'red', 'shoulder', 'body'));
  }

  /* Hip Ring â æ´æ */
  const lh = pts.leftHip, rh = pts.rightHip;
  if (lh && rh) {
    els.push(...buildDisc(lh, rh, {
      dropRatio:  0,
      rxMult:     1.50,
      rxMin:      0.12,
      rxMax:      0.38,
      ryRatio:    viewType === 'face_on' ? 0.18 : 0.12,
      maxAngle:   viewType === 'face_on' ? 10 : 30,
      guideRatio: 1.40,
      label:      'HIPS',
    }, 'red', 'hip', 'club'));
  }

  return els;
}

/* ââ compat ââ */
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
