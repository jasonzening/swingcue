/**
 * keypointOverlay.ts — dense MediaPipe-driven disc rendering (fallback path).
 *
 * PR-2C: stripped Phase 7.x stateful post-processing (EMA smoothing, refW
 * lock, cy shift, zAsym phase shift, slew-rate limit, isUltraFlat, low-conf
 * last-known-good). New videos go through sparsePhaseOverlay.ts; this path
 * stays for legacy videos that only have keypoint_timeline_json. UX impact:
 * 4fps samples no longer EMA-smoothed (see commit message for trade-off).
 */

import type {
  OverlayElement, KeypointFrame,
  LineElement, DotElement, LabelElement,
} from '@/types/analysis';
import type { MainIssueType } from '@/types/analysis';
import type { BodyPointName, Pt } from '@/lib/golf/bodyPointSpec';
import type { ViewType } from '@/lib/golf/overlayLineSpec';

let _uid = 0;
const uid    = (p: string) => `${p}-${++_uid}`;
const clamp  = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const dist2D = (a: Pt, b: Pt) => Math.hypot(b.x - a.x, b.y - a.y);
const mid2D  = (a: Pt, b: Pt): Pt => ({ x:(a.x+b.x)/2, y:(a.y+b.y)/2, confidence:1 });

type AC = 'red' | 'green' | 'yellow' | 'white';

const mkLine  = (x1:number,y1:number,x2:number,y2:number,c:AC,w=2.5,op=0.90,layer:OverlayElement['layer']='body'): LineElement =>
  ({type:'line',id:uid('l'),x1,y1,x2,y2,color:c,strokeWidth:w,opacity:op,layer});
const mkDot   = (x:number,y:number,c:AC,r=0.009,op=0.90,layer:OverlayElement['layer']='body'): DotElement =>
  ({type:'dot',id:uid('d'),x,y,color:c,radius:r,opacity:op,layer});
const mkLabel = (x:number,y:number,text:string,c:AC='white',size=10,op=0.80): LabelElement =>
  ({type:'label',id:uid('t'),x,y,text,color:c,size,opacity:op});

// Basic ellipse — no visRatio/zAsym/bodyHalfRatio; renderer applies defaults.
function mkEllipse(cx:number,cy:number,rx:number,ry:number,angleDeg:number,color:AC,sw=5.0,op=0.92,layer:OverlayElement['layer']='body'):OverlayElement{
  return{type:'ellipse'as OverlayElement['type'],id:uid('e'),cx,cy,rx,ry,angleDeg,color,strokeWidth:sw,opacity:op,layer}as unknown as OverlayElement;
}

function normalizeAngle(deg:number):number{let a=deg;if(a>90)a-=180;if(a<-90)a+=180;return a;}

/** Stateless disc builder: 2 endpoint dots + ellipse + center guide + label. */
function buildDisc(
  leftPt: Pt, rightPt: Pt,
  opts: { rxMult:number; rxMin:number; rxMax:number; ryRatio:number; maxAngle:number; label:string; dotExpand:number; },
  color: AC,
  layer: OverlayElement['layer'] = 'body',
): OverlayElement[] {
  const dist = dist2D(leftPt, rightPt);
  if (dist < 0.01) return [];

  const cx = (leftPt.x + rightPt.x) / 2;
  const cy = (leftPt.y + rightPt.y) / 2;
  const dx = rightPt.x - leftPt.x;
  const dy = rightPt.y - leftPt.y;

  const rx = clamp(dist * opts.rxMult, opts.rxMin, opts.rxMax);
  const ry = rx * opts.ryRatio;
  const angleDeg = clamp(normalizeAngle(Math.atan2(dy, dx) * 180 / Math.PI), -opts.maxAngle, opts.maxAngle);

  const expand = dist * opts.dotExpand;
  const ux = dx / dist, uy = dy / dist;
  const lDotX = leftPt.x  - ux * expand, lDotY = leftPt.y  - uy * expand;
  const rDotX = rightPt.x + ux * expand, rDotY = rightPt.y + uy * expand;

  const els: OverlayElement[] = [];
  els.push(mkDot(lDotX, lDotY, 'yellow', 0.006, 0.30, layer));
  els.push(mkDot(rDotX, rDotY, 'white',  0.006, 0.30, layer));
  els.push(mkEllipse(cx, cy, rx, ry, angleDeg, color, 5.0, 0.92, layer));

  const ar = angleDeg * Math.PI / 180;
  const cosA = Math.cos(ar), sinA = Math.sin(ar);
  const guideHalfLen = rx * 0.65;
  els.push(mkLine(cx - guideHalfLen*cosA, cy - guideHalfLen*sinA, cx + guideHalfLen*cosA, cy + guideHalfLen*sinA, 'white', 2.2, 0.70, layer));
  els.push(mkLabel(cx, cy - ry * 2.2, opts.label, 'white', 10, 0.78));
  return els;
}

export function getKeypoints(kpFrame: KeypointFrame): Partial<Record<BodyPointName, Pt>> {
  const lm = kpFrame.landmarks;
  const r: Partial<Record<BodyPointName, Pt>> = {};
  const toP = (pt?: {x:number;y:number;z?:number;confidence?:number}|null) =>
    pt ? {x:pt.x,y:pt.y,z:pt.z,confidence:pt.confidence??0.8} : null;
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

export function generateSpecDrivenOverlayFrame(
  issue:MainIssueType,viewType:ViewType,phase:string,
  kpFrame:KeypointFrame,_hist?:Array<{x:number;y:number}>,
):OverlayElement[]{
  _uid=0;
  const pts=getKeypoints(kpFrame);
  const els:OverlayElement[]=[];
  const maxAng=viewType==='face_on'?30:50;

  const ls=pts.leftShoulder,rs=pts.rightShoulder;
  if(ls&&rs){els.push(...buildDisc(ls,rs,{
    rxMult:1.85,rxMin:0.20,rxMax:0.50,ryRatio:0.20,maxAngle:maxAng,
    label:'SHOULDERS',dotExpand:0,
  },'white','body'));}

  const lh=pts.leftHip,rh=pts.rightHip;
  if(lh&&rh){els.push(...buildDisc(lh,rh,{
    rxMult:2.10,rxMin:0.16,rxMax:0.50,ryRatio:0.18,maxAngle:maxAng,
    label:'HIPS',dotExpand:0.52,
  },'white','club'));}

  void issue; void phase;
  return els;
}

export function computePerspectiveDisc(args:{leftPoint:Pt;rightPoint:Pt;viewType:ViewType;kind:'shoulder'|'hip';previousAngle?:number;}){
  const{leftPoint:lp,rightPoint:rp,viewType,kind}=args;
  const dist=dist2D(lp,rp),cx=(lp.x+rp.x)/2,cy=(lp.y+rp.y)/2;
  const rx=clamp(dist*(kind==='shoulder'?1.85:2.10),kind==='shoulder'?0.20:0.16,0.50);
  const ry=rx*(kind==='shoulder'?0.20:0.18);
  const raw=Math.atan2(rp.y-lp.y,rp.x-lp.x)*180/Math.PI;
  const sd=clamp(normalizeAngle(raw),viewType==='face_on'?-30:-50,viewType==='face_on'?30:50);
  const ar=sd*Math.PI/180,ca=Math.cos(ar),sa=Math.sin(ar);
  return{cx,cy,rx,ry,rotationDeg:sd,guideStart:{x:cx-rx*ca,y:cy-rx*sa},guideEnd:{x:cx+rx*ca,y:cy+rx*sa},alpha:0.92,visible:dist>0.01};
}

export function getTrackedPoint(_i:MainIssueType,_v:ViewType,kpFrame:KeypointFrame):{x:number;y:number}|null{
  const pts=getKeypoints(kpFrame);return pts.shoulderCenter?{x:pts.shoulderCenter.x,y:pts.shoulderCenter.y}:null;
}
export function findNearestFrame(frames:KeypointFrame[],time:number):KeypointFrame|null{
  if(!frames.length)return null;let best=frames[0],bestD=Math.abs(time-best.time);
  for(const f of frames){const d=Math.abs(time-f.time);if(d<bestD){best=f;bestD=d;}}return best;
}
export function applyCorrection(pt:{x:number;y:number;confidence?:number},dir:string,mag='medium'):{x:number;y:number;confidence?:number}{
  const D:Record<string,number>={small:0.028,medium:0.050,large:0.078};const d=D[mag]??0.050;
  switch(dir){case 'lower':return{...pt,y:pt.y+d};case 'higher':return{...pt,y:pt.y-d};
    case 'more_inside':return{...pt,x:pt.x-d};case 'more_outside':return{...pt,x:pt.x+d};
    case 'more_centered':return{...pt,x:0.50};default:return pt;}
}
