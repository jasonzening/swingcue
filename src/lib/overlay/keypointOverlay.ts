/**
 * keypointOverlay.ts — 旋转盘（透视版）
 *
 * 关键修复：
 * 1. 关节点向"远离圆盘中心"方向扩展（不依赖 MediaPipe 的 left/right X 顺序）
 *    face_on: leftShoulder.x > rightShoulder.x（左肩在画面右边）
 *    所以必须用 sign(pt.x - cx) 决定扩展方向
 * 2. 透视压缩：ry *= visRatio，侧身时圆盘变细
 * 3. 左点=黄色，右点=白色
 * 4. 髋部外扩量 = dist * 0.52
 */

import type {
  OverlayElement, KeypointFrame,
  LineElement, DotElement, LabelElement,
} from '@/types/analysis';
import type { MainIssueType } from '@/types/analysis';
import type { BodyPointName, Pt } from '@/lib/golf/bodyPointSpec';
import type { ViewType } from '@/lib/golf/overlayLineSpec';

const _prevAngle: Record<string, number> = {};
const _refWidth:  Record<string, number> = {};

let _uid = 0;
const uid   = (p: string) => `${p}-${++_uid}`;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const dist2D = (a: Pt, b: Pt) => Math.hypot(b.x - a.x, b.y - a.y);
const mid2D  = (a: Pt, b: Pt): Pt => ({ x:(a.x+b.x)/2, y:(a.y+b.y)/2, confidence:1 });

type AC = 'red' | 'green' | 'yellow' | 'white';

const mkLine  = (x1:number,y1:number,x2:number,y2:number,c:AC,w=2.5,op=0.90,layer:OverlayElement['layer']='body'): LineElement =>
  ({type:'line',id:uid('l'),x1,y1,x2,y2,color:c,strokeWidth:w,opacity:op,layer});
const mkDot   = (x:number,y:number,c:AC,r=0.009,op=0.90,layer:OverlayElement['layer']='body'): DotElement =>
  ({type:'dot',id:uid('d'),x,y,color:c,radius:r,opacity:op,layer});
const mkLabel = (x:number,y:number,text:string,c:AC='white',size=10,op=0.80): LabelElement =>
  ({type:'label',id:uid('t'),x,y,text,color:c,size,opacity:op});

function mkEllipse(cx:number,cy:number,rx:number,ry:number,angleDeg:number,color:AC,sw=5.0,op=0.92,layer:OverlayElement['layer']='body',bhr=0.27):OverlayElement{
  return{type:'ellipse'as OverlayElement['type'],id:uid('e'),cx,cy,rx,ry,angleDeg,color,strokeWidth:sw,opacity:op,layer,bodyHalfRatio:bhr}as unknown as OverlayElement;
}

function normalizeAngle(deg:number):number{let a=deg;if(a>90)a-=180;if(a<-90)a+=180;return a;}

function buildDisc(
  leftPt: Pt, rightPt: Pt,
  opts: {
    rxMult:number; rxMin:number; rxMax:number;
    ryRatio:number; maxAngle:number;
    refKey:string; angleKey:string;
    label:string; dotExpand:number;
  },
  color: AC,
  layer: OverlayElement['layer'] = 'body',
): OverlayElement[] {
  const els: OverlayElement[] = [];
  const lc = leftPt.confidence  ?? 0.8;
  const rc = rightPt.confidence ?? 0.8;
  const dist = dist2D(leftPt, rightPt);
  if (dist < 0.01) return els;

  const cx = (leftPt.x + rightPt.x) / 2;
  const cy = (leftPt.y + rightPt.y) / 2;

  const apparentW = Math.abs(rightPt.x - leftPt.x);
  const prevRef   = _refWidth[opts.refKey] ?? 0;
  if (apparentW > prevRef) _refWidth[opts.refKey] = apparentW;
  const refW     = _refWidth[opts.refKey] ?? Math.max(apparentW, 0.10);
  const visRatio = clamp(apparentW / refW, 0, 1.0);

  // 关节点向远离中心方向扩展
  const expand = dist * opts.dotExpand;
  const lDir   = leftPt.x  >= cx ? 1 : -1;
  const rDir   = rightPt.x >= cx ? 1 : -1;
  const lDotX  = leftPt.x  + lDir * expand;
  const rDotX  = rightPt.x + rDir * expand;

  if (lc < 0.35 || rc < 0.35) {
    els.push(mkDot(cx, cy, color, 0.012, 0.55, layer));
    return els;
  }

  els.push(mkDot(lDotX, leftPt.y,  'yellow', 0.011, 0.90, layer));
  els.push(mkDot(rDotX, rightPt.y, 'white',  0.011, 0.90, layer));

  if (visRatio < 0.18) {
    els.push(mkLine(leftPt.x,leftPt.y,rightPt.x,rightPt.y,color,1.5,0.45,layer));
    return els;
  }

  const rx = clamp(dist * opts.rxMult, opts.rxMin, opts.rxMax);
  const ry = rx * opts.ryRatio * visRatio;

  const rawDeg   = Math.atan2(rightPt.y - leftPt.y, rightPt.x - leftPt.x) * 180 / Math.PI;
  const normDeg  = normalizeAngle(rawDeg);
  const clampDeg = clamp(normDeg, -opts.maxAngle, opts.maxAngle);
  const prev     = _prevAngle[opts.angleKey];
  const smoothDeg = prev !== undefined ? prev + clamp(clampDeg - prev, -15, 15) : clampDeg;
  _prevAngle[opts.angleKey] = smoothDeg;

  const bhr   = clamp((dist / 2) / rx, 0.05, 0.95);
  const alpha = 0.50 + visRatio * 0.42;
  els.push(mkEllipse(cx, cy, rx, ry, smoothDeg, color, 5.0, alpha, layer, bhr));

  const ar = smoothDeg * Math.PI / 180;
  const cosA = Math.cos(ar), sinA = Math.sin(ar);
  els.push(mkLine(cx - rx*cosA, cy - rx*sinA, cx + rx*cosA, cy + rx*sinA, 'white', 2.2, alpha * 0.90, layer));

  if (visRatio > 0.50) {
    els.push(mkLabel(cx, cy - ry * 2.2, opts.label, 'white', 10, 0.78));
  }
  return els;
}

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
    refKey:'sRef',angleKey:'sAng',label:'SHOULDERS',dotExpand:0,
  },'red','body'));}

  const lh=pts.leftHip,rh=pts.rightHip;
  if(lh&&rh){els.push(...buildDisc(lh,rh,{
    rxMult:2.10,rxMin:0.16,rxMax:0.50,ryRatio:0.18,maxAngle:maxAng,
    refKey:'hRef',angleKey:'hAng',label:'HIPS',dotExpand:0.52,
  },'red','club'));}

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
