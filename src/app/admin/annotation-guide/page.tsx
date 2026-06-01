/**
 * src/app/admin/annotation-guide/page.tsx
 *
 * PR-7A.1: Anatomical Annotation Guide — Bone-Top-Centerline Principle
 *
 * Fixes baked in (vs. v1):
 *   1. Acknowledgement enforced via React useState — button disabled
 *      until the checkbox is ticked. ESC keypress is a no-op until
 *      ackChecked is true.
 *   2. Neutral admin UI (no brand blue) — only green/red appear, and
 *      ONLY inside the bone schematics (correct vs wrong click target).
 *   3. Hip card explicitly states "estimated internal joint center"
 *      and warns against clicking the visible trochanter bump.
 *
 * Routes / usage:
 *   - GET /admin/annotation-guide  → renders <AnnotationGuideBody mode="standalone" />
 *   - Workbench imports <AnnotationGuideBody mode="modal" onAcknowledge={...} />
 *     and shows it on first visit when the localStorage key is absent.
 *
 * Styling note: the architect's reference template was written in
 * Tailwind. This file adapts the class strings to match the existing
 * codebase convention — a single `const css = `…`` template literal +
 * <style>{css}</style>, since the project has Tailwind v4 installed but
 * not wired into globals.css yet. The four SVG diagrams, the bilingual
 * text, the neutral color scheme, and the React state-based
 * acknowledgement enforcement are PRESERVED verbatim.
 */

'use client';

import { useCallback, useEffect, useState } from 'react';

export const ANNOTATION_GUIDE_STORAGE_KEY = 'swingcue.annotation-guide.read-v2';

// ─────────────────────────────────────────────────────────────────────────────
// BONE SCHEMATIC SVGs — small simplified diagrams, NOT realistic anatomy
// solid line = bone; dashed = skin/muscle outline; green dot = click target;
// red X = common wrong click. Each SVG = 200x160 viewBox. VERBATIM from spec.
// ─────────────────────────────────────────────────────────────────────────────

export function ShoulderDiagram() {
  return (
    <svg viewBox="0 0 200 160" width="190" height="152" role="img" aria-label="Shoulder click target">
      <path d="M 20 30 Q 30 18, 60 18 L 110 22 Q 135 28, 150 55 L 165 95 L 158 130"
            stroke="#B4B2A9" strokeWidth="1" strokeDasharray="3,3" fill="none"/>
      <line x1="50" y1="48" x2="118" y2="60" stroke="#5F5E5A" strokeWidth="2.5" strokeLinecap="round"/>
      <circle cx="124" cy="64" r="13" fill="none" stroke="#5F5E5A" strokeWidth="2.2"/>
      <line x1="128" y1="74" x2="148" y2="130" stroke="#5F5E5A" strokeWidth="2.5" strokeLinecap="round"/>
      <text x="20" y="155" fontSize="9" fill="#888780">clavicle + humerus</text>
      <circle cx="124" cy="55" r="5" fill="#1D9E75"/>
      <circle cx="124" cy="55" r="9" fill="none" stroke="#1D9E75" strokeOpacity="0.5"/>
      <text x="135" y="48" fontSize="10" fill="#0F6E56" fontWeight="500">bone apex</text>
      <g transform="translate(155,75)">
        <line x1="-4" y1="-4" x2="4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
        <line x1="4" y1="-4" x2="-4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
      </g>
      <text x="163" y="80" fontSize="9" fill="#A32D2D">deltoid peak</text>
      <g transform="translate(48,42)">
        <line x1="-4" y1="-4" x2="4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
        <line x1="4" y1="-4" x2="-4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
      </g>
      <text x="3" y="36" fontSize="9" fill="#A32D2D">neck base</text>
    </svg>
  );
}

export function ElbowDiagram() {
  return (
    <svg viewBox="0 0 200 160" width="190" height="152" role="img" aria-label="Elbow click target">
      <path d="M 60 12 Q 90 18, 105 50 Q 118 78, 95 105 Q 82 130, 105 152"
            stroke="#B4B2A9" strokeWidth="1" strokeDasharray="3,3" fill="none"/>
      <path d="M 138 12 Q 130 28, 122 55 Q 115 80, 130 105 Q 145 130, 140 152"
            stroke="#B4B2A9" strokeWidth="1" strokeDasharray="3,3" fill="none"/>
      <line x1="92" y1="20" x2="112" y2="78" stroke="#5F5E5A" strokeWidth="2.5" strokeLinecap="round"/>
      <circle cx="115" cy="85" r="12" fill="none" stroke="#5F5E5A" strokeWidth="2.2"/>
      <line x1="118" y1="95" x2="128" y2="152" stroke="#5F5E5A" strokeWidth="2.5" strokeLinecap="round"/>
      <text x="22" y="155" fontSize="9" fill="#888780">humerus + ulna</text>
      <circle cx="115" cy="78" r="5" fill="#1D9E75"/>
      <circle cx="115" cy="78" r="9" fill="none" stroke="#1D9E75" strokeOpacity="0.5"/>
      <text x="128" y="74" fontSize="10" fill="#0F6E56" fontWeight="500">bone pivot</text>
      <g transform="translate(85,55)">
        <line x1="-4" y1="-4" x2="4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
        <line x1="4" y1="-4" x2="-4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
      </g>
      <text x="34" y="60" fontSize="9" fill="#A32D2D">biceps bulge</text>
      <g transform="translate(150,105)">
        <line x1="-4" y1="-4" x2="4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
        <line x1="4" y1="-4" x2="-4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
      </g>
      <text x="158" y="110" fontSize="9" fill="#A32D2D">forearm</text>
    </svg>
  );
}

export function WristDiagram() {
  return (
    <svg viewBox="0 0 200 160" width="190" height="152" role="img" aria-label="Wrist click target">
      <path d="M 30 15 Q 50 18, 70 35 L 90 75 L 100 100 Q 105 115, 130 130 L 175 145"
            stroke="#B4B2A9" strokeWidth="1" strokeDasharray="3,3" fill="none"/>
      <path d="M 30 35 Q 55 38, 75 60 L 90 95 L 100 115 Q 110 130, 140 145"
            stroke="#B4B2A9" strokeWidth="1" strokeDasharray="3,3" fill="none"/>
      <line x1="60" y1="25" x2="98" y2="92" stroke="#5F5E5A" strokeWidth="2.2" strokeLinecap="round"/>
      <line x1="68" y1="45" x2="106" y2="105" stroke="#5F5E5A" strokeWidth="2.2" strokeLinecap="round"/>
      <circle cx="102" cy="98" r="3.5" fill="#5F5E5A"/>
      <circle cx="106" cy="105" r="3.5" fill="#5F5E5A"/>
      <line x1="108" y1="100" x2="155" y2="135" stroke="#5F5E5A" strokeWidth="2" strokeLinecap="round"/>
      <text x="22" y="155" fontSize="9" fill="#888780">ulna + radius</text>
      <circle cx="104" cy="101.5" r="5" fill="#1D9E75"/>
      <circle cx="104" cy="101.5" r="9" fill="none" stroke="#1D9E75" strokeOpacity="0.5"/>
      <text x="116" y="98" fontSize="10" fill="#0F6E56" fontWeight="500">midpoint of bone-tops</text>
      <g transform="translate(140,128)">
        <line x1="-4" y1="-4" x2="4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
        <line x1="4" y1="-4" x2="-4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
      </g>
      <text x="148" y="133" fontSize="9" fill="#A32D2D">glove cuff</text>
      <g transform="translate(78,68)">
        <line x1="-4" y1="-4" x2="4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
        <line x1="4" y1="-4" x2="-4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
      </g>
      <text x="32" y="68" fontSize="9" fill="#A32D2D">watch / shirt cuff</text>
    </svg>
  );
}

// PR-7A.2: head_crown — top of skull at cranial midline. Solid skull
// outline (bone). Dashed hair contour for context. Correct = apex of
// cranial midline; wrong = hairline / forehead bulge (skin-only).
export function HeadCrownDiagram() {
  return (
    <svg viewBox="0 0 200 160" width="190" height="152" role="img" aria-label="Head crown click target">
      {/* hair / scalp outline (dashed) */}
      <path d="M 50 35 Q 60 12, 100 8 Q 140 12, 152 38 Q 155 55, 150 70"
            stroke="#B4B2A9" strokeWidth="1" strokeDasharray="3,3" fill="none"/>
      {/* skull bone — rounded calvarium */}
      <path d="M 55 45 Q 65 22, 100 18 Q 135 22, 148 48 Q 152 70, 148 90 L 138 110 L 65 110 L 55 90 Q 51 70, 55 45 Z"
            fill="none" stroke="#5F5E5A" strokeWidth="2"/>
      {/* ear hint (right side) */}
      <path d="M 148 78 Q 155 82, 154 92 Q 152 100, 145 100"
            fill="none" stroke="#5F5E5A" strokeWidth="1.5"/>
      {/* spinal/neck axis dashed reference */}
      <line x1="100" y1="110" x2="100" y2="155" stroke="#B4B2A9" strokeWidth="1" strokeDasharray="3,3"/>
      <text x="22" y="155" fontSize="9" fill="#888780">calvarium (skull)</text>
      {/* correct click — cranial midline apex */}
      <circle cx="100" cy="18" r="5" fill="#1D9E75"/>
      <circle cx="100" cy="18" r="9" fill="none" stroke="#1D9E75" strokeOpacity="0.5"/>
      <text x="108" y="14" fontSize="10" fill="#0F6E56" fontWeight="500">cranial apex</text>
      {/* wrong — hairline */}
      <g transform="translate(90,38)">
        <line x1="-4" y1="-4" x2="4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
        <line x1="4" y1="-4" x2="-4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
      </g>
      <text x="20" y="38" fontSize="9" fill="#A32D2D">hairline</text>
      {/* wrong — forehead bulge */}
      <g transform="translate(110,62)">
        <line x1="-4" y1="-4" x2="4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
        <line x1="4" y1="-4" x2="-4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
      </g>
      <text x="118" y="62" fontSize="9" fill="#A32D2D">forehead</text>
    </svg>
  );
}

// PR-7A.2: chin — midpoint of mandibular symphysis (bottom-center of
// jaw BONE, not soft tissue below it). Solid jaw line + correct apex;
// red X for "soft chin tissue" and "lower lip".
export function ChinDiagram() {
  return (
    <svg viewBox="0 0 200 160" width="190" height="152" role="img" aria-label="Chin click target">
      {/* face outline (skin) dashed */}
      <path d="M 60 18 Q 50 60, 60 100 Q 70 140, 100 148 Q 130 140, 140 100 Q 150 60, 140 18"
            stroke="#B4B2A9" strokeWidth="1" strokeDasharray="3,3" fill="none"/>
      {/* mandibular arch (jaw bone) */}
      <path d="M 65 65 Q 70 100, 100 122 Q 130 100, 135 65"
            fill="none" stroke="#5F5E5A" strokeWidth="2.2"/>
      {/* upper mouth line (lips dashed) */}
      <path d="M 78 96 Q 100 102, 122 96"
            stroke="#B4B2A9" strokeWidth="1" strokeDasharray="3,3" fill="none"/>
      <text x="22" y="155" fontSize="9" fill="#888780">mandibular bone</text>
      {/* correct click — bone-bottom midline (symphysis) */}
      <circle cx="100" cy="122" r="5" fill="#1D9E75"/>
      <circle cx="100" cy="122" r="9" fill="none" stroke="#1D9E75" strokeOpacity="0.5"/>
      <text x="108" y="118" fontSize="10" fill="#0F6E56" fontWeight="500">symphysis</text>
      {/* wrong — soft chin tissue below bone */}
      <g transform="translate(100,140)">
        <line x1="-4" y1="-4" x2="4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
        <line x1="4" y1="-4" x2="-4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
      </g>
      <text x="108" y="144" fontSize="9" fill="#A32D2D">soft tissue</text>
      {/* wrong — lower lip */}
      <g transform="translate(100,102)">
        <line x1="-4" y1="-4" x2="4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
        <line x1="4" y1="-4" x2="-4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
      </g>
      <text x="40" y="98" fontSize="9" fill="#A32D2D">lower lip</text>
    </svg>
  );
}

// PR-7A.2: knee — lateral femoral epicondyle apex (the visible bony
// bump on the OUTSIDE of the knee at the joint line). NOT the patella
// (kneecap) center, NOT the inner side.
export function KneeDiagram() {
  return (
    <svg viewBox="0 0 200 160" width="190" height="152" role="img" aria-label="Knee click target">
      {/* skin / pant leg silhouette */}
      <path d="M 75 8 Q 70 50, 78 75 Q 88 100, 80 130 L 78 155"
            stroke="#B4B2A9" strokeWidth="1" strokeDasharray="3,3" fill="none"/>
      <path d="M 130 8 Q 145 50, 142 75 Q 148 100, 150 130 L 152 155"
            stroke="#B4B2A9" strokeWidth="1" strokeDasharray="3,3" fill="none"/>
      {/* femur (upper) */}
      <line x1="98" y1="15" x2="110" y2="78" stroke="#5F5E5A" strokeWidth="2.5" strokeLinecap="round"/>
      {/* tibia (lower) */}
      <line x1="112" y1="92" x2="120" y2="155" stroke="#5F5E5A" strokeWidth="2.5" strokeLinecap="round"/>
      {/* knee joint capsule */}
      <ellipse cx="111" cy="85" rx="20" ry="12" fill="none" stroke="#5F5E5A" strokeWidth="2"/>
      {/* patella (kneecap) — inside the capsule */}
      <circle cx="108" cy="84" r="6" fill="none" stroke="#5F5E5A" strokeWidth="1.5"/>
      <text x="22" y="155" fontSize="9" fill="#888780">femur + tibia (right leg, lateral view)</text>
      {/* correct click — lateral epicondyle apex (outside) */}
      <circle cx="131" cy="85" r="5" fill="#1D9E75"/>
      <circle cx="131" cy="85" r="9" fill="none" stroke="#1D9E75" strokeOpacity="0.5"/>
      <text x="140" y="82" fontSize="10" fill="#0F6E56" fontWeight="500">lat. epicondyle</text>
      {/* wrong — patella center */}
      <g transform="translate(108,84)">
        <line x1="-4" y1="-4" x2="4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
        <line x1="4" y1="-4" x2="-4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
      </g>
      <text x="60" y="80" fontSize="9" fill="#A32D2D">patella ctr</text>
      {/* wrong — inner knee */}
      <g transform="translate(91,85)">
        <line x1="-4" y1="-4" x2="4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
        <line x1="4" y1="-4" x2="-4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
      </g>
      <text x="50" y="100" fontSize="9" fill="#A32D2D">inner knee</text>
    </svg>
  );
}

// PR-7A.2: ankle — lateral malleolus apex (visible bony bump on the
// OUTSIDE of the ankle, distal fibula tip). Bone-surface landmark, NOT
// an internal joint center (unlike hip).
export function AnkleDiagram() {
  return (
    <svg viewBox="0 0 200 160" width="190" height="152" role="img" aria-label="Ankle click target">
      {/* pant leg / sock dashed */}
      <path d="M 65 5 Q 60 50, 72 90 L 72 110 L 60 145"
            stroke="#B4B2A9" strokeWidth="1" strokeDasharray="3,3" fill="none"/>
      <path d="M 135 5 Q 140 50, 130 90 L 130 110 L 145 145"
            stroke="#B4B2A9" strokeWidth="1" strokeDasharray="3,3" fill="none"/>
      {/* tibia + fibula */}
      <line x1="90" y1="10" x2="95" y2="110" stroke="#5F5E5A" strokeWidth="2.5" strokeLinecap="round"/>
      <line x1="115" y1="10" x2="118" y2="105" stroke="#5F5E5A" strokeWidth="2" strokeLinecap="round"/>
      {/* foot outline */}
      <path d="M 70 122 Q 90 118, 130 120 L 175 130 Q 178 142, 165 148 L 70 148 Q 60 140, 70 122 Z"
            fill="none" stroke="#5F5E5A" strokeWidth="2"/>
      {/* lateral malleolus — visible bone bump on fibula end */}
      <circle cx="118" cy="115" r="6" fill="none" stroke="#5F5E5A" strokeWidth="1.5"/>
      {/* medial malleolus hint (tibia, smaller, inner) */}
      <circle cx="95" cy="115" r="4" fill="none" stroke="#5F5E5A" strokeWidth="1"/>
      <text x="22" y="158" fontSize="9" fill="#888780">tibia + fibula + foot (lateral view)</text>
      {/* correct click — lateral malleolus apex */}
      <circle cx="123" cy="120" r="5" fill="#1D9E75"/>
      <circle cx="123" cy="120" r="9" fill="none" stroke="#1D9E75" strokeOpacity="0.5"/>
      <text x="132" y="117" fontSize="10" fill="#0F6E56" fontWeight="500">lat. malleolus</text>
      {/* wrong — shoe edge */}
      <g transform="translate(160,138)">
        <line x1="-4" y1="-4" x2="4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
        <line x1="4" y1="-4" x2="-4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
      </g>
      <text x="138" y="146" fontSize="9" fill="#A32D2D">shoe edge</text>
      {/* wrong — inner ankle */}
      <g transform="translate(90,116)">
        <line x1="-4" y1="-4" x2="4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
        <line x1="4" y1="-4" x2="-4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
      </g>
      <text x="30" y="116" fontSize="9" fill="#A32D2D">inner ankle</text>
    </svg>
  );
}

export function HipDiagram() {
  return (
    <svg viewBox="0 0 200 160" width="190" height="152" role="img" aria-label="Hip click target">
      <path d="M 30 25 Q 50 22, 75 30 Q 105 40, 140 60 L 165 100 Q 170 130, 162 150"
            stroke="#B4B2A9" strokeWidth="1" strokeDasharray="3,3" fill="none"/>
      <path d="M 30 60 L 60 65 Q 95 75, 125 85"
            stroke="#B4B2A9" strokeWidth="1" strokeDasharray="3,3" fill="none"/>
      <path d="M 35 25 Q 60 30, 95 45 Q 120 55, 130 75 L 132 88 L 110 88 L 85 70 Q 60 58, 35 55 Z"
            fill="none" stroke="#5F5E5A" strokeWidth="2"/>
      <circle cx="105" cy="68" r="11" fill="none" stroke="#5F5E5A" strokeWidth="2.2"/>
      <line x1="112" y1="74" x2="155" y2="148" stroke="#5F5E5A" strokeWidth="2.5" strokeLinecap="round"/>
      <path d="M 132 78 L 145 80 Q 150 82, 148 88 L 140 88 Z" fill="none" stroke="#5F5E5A" strokeWidth="2"/>
      <text x="22" y="155" fontSize="9" fill="#888780">pelvis + femur</text>
      <circle cx="105" cy="68" r="5" fill="#1D9E75"/>
      <circle cx="105" cy="68" r="9" fill="none" stroke="#1D9E75" strokeOpacity="0.5"/>
      <text x="50" y="98" fontSize="10" fill="#0F6E56" fontWeight="500">femoral head ctr</text>
      <g transform="translate(146,84)">
        <line x1="-4" y1="-4" x2="4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
        <line x1="4" y1="-4" x2="-4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
      </g>
      <text x="153" y="80" fontSize="9" fill="#A32D2D">trochanter bump</text>
      <g transform="translate(78,32)">
        <line x1="-4" y1="-4" x2="4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
        <line x1="4" y1="-4" x2="-4" y2="4" stroke="#A32D2D" strokeWidth="1.8"/>
      </g>
      <text x="50" y="22" fontSize="9" fill="#A32D2D">waistband</text>
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CARD CONTENT — VERBATIM bilingual text from spec.
// ─────────────────────────────────────────────────────────────────────────────

type CardSpec = {
  key: string;
  title: string;
  requirement: string;
  definitionZh: string;
  definitionEn: string;
  blurb: string;
  views: { face_on: string; dtl: string };
  diagram: () => React.ReactElement;
  optional: boolean;
  warning?: string;
};

const CARDS: CardSpec[] = [
  {
    key: 'shoulder',
    title: 'Shoulder · 肩',
    requirement: '2 per phase · required',
    definitionZh: '肱骨头顶点',
    definitionEn: 'humeral head apex',
    blurb: 'the top-center of the upper-arm bone where it meets the shoulder socket',
    views: {
      face_on: 'Inside of the shoulder curve peak (where the bone arch crests, just below the trapezius line)',
      dtl: 'The visible top of the shoulder bone, sitting just above the visible armpit angle',
    },
    diagram: ShoulderDiagram,
    optional: false,
  },
  {
    key: 'elbow',
    title: 'Elbow · 肘',
    requirement: '2 per phase · required',
    definitionZh: '尺骨鹰嘴顶 / 肱骨远端中线',
    definitionEn: 'olecranon apex / distal humerus centerline',
    blurb: 'the bony bump at the back of a bent elbow, on the bone rotation axis',
    views: {
      face_on: 'Outer corner of the elbow joint, on the bone axis between upper-arm and forearm lines',
      dtl: 'The back of the elbow tip — the visible bony bump that pokes out when the arm bends',
    },
    diagram: ElbowDiagram,
    optional: false,
  },
  {
    key: 'wrist',
    title: 'Wrist · 腕',
    requirement: '2 per phase · required',
    definitionZh: '尺骨头 + 桡骨远端中点的顶',
    definitionEn: 'midpoint of the two distal wrist-bone tops',
    blurb: 'on the forearm bone-axis line',
    views: {
      face_on: 'Center of the wrist crease, not the glove edge, not the club grip',
      dtl: 'Forearm bone-axis line where it meets the hand (just before the bone disappears under the glove)',
    },
    diagram: WristDiagram,
    optional: false,
  },
  {
    key: 'hip',
    title: 'Hip · 髋',
    requirement: '2 per phase · optional',
    definitionZh: '股骨头中心 (估计点)',
    definitionEn: 'femoral head center — ESTIMATED',
    blurb: 'the rotation center of the hip ball-joint, hidden inside the pelvis',
    views: {
      face_on: 'About midway between ASIS (hipbone front) and pubic bone, slightly inside the trochanter line',
      dtl: '~3–5 cm above the visible trochanter bump, on the femur bone-axis',
    },
    diagram: HipDiagram,
    optional: true,
    warning:
      'Hip is an estimated internal joint center — you cannot see this point directly, you must estimate it. NEVER click the visible outer trochanter bump (the lateral hip bone bulge). 髋点是内部旋转中心的估计点,绝不是外侧看得见的髋骨凸起。',
  },
  // PR-7A.2: head + leg clusters. Both are bone-surface landmarks
  // (unlike hip's internal-center estimation): the annotator clicks
  // visible bone, not an inferred internal point.
  {
    key: 'head_crown',
    title: 'Head crown · 颅顶',
    requirement: '1 per phase · required',
    definitionZh: '颅顶正中点',
    definitionEn: 'top of skull at cranial midline',
    blurb: 'the apex of the calvarium, on the body’s vertical axis through the neck',
    views: {
      face_on: 'Highest center of the skull — extend the body axis up through the neck to the apex',
      dtl: 'Topmost point of the skull silhouette, on the cranial midline (above the ear)',
    },
    diagram: HeadCrownDiagram,
    optional: false,
    warning:
      'NOT the hairline, NOT the forehead. The point of the skull bone at the top of the head — imagine the body’s vertical axis extending up through the neck and skull to its apex. 不是发际线,不是额头,是颅骨最高点。',
  },
  {
    key: 'chin',
    title: 'Chin · 下颌',
    requirement: '1 per phase · required',
    definitionZh: '下颌正中点',
    definitionEn: 'midpoint of mandibular symphysis',
    blurb: 'the lowest center of the jaw BONE (not the soft tissue below it)',
    views: {
      face_on: 'Lowest center point of the jaw bone (mandibular symphysis), on the facial midline',
      dtl: 'Bottom of the jaw bone on the facial midline (where bone ends, before soft chin tissue)',
    },
    diagram: ChinDiagram,
    optional: false,
    warning:
      'Click where the BONE is, NOT where the soft chin tissue ends. The jaw bone often sits a few mm above the apparent skin edge. 点骨头位置,不要点下面的软组织。',
  },
  {
    key: 'knee',
    title: 'Knee · 膝',
    requirement: '2 per phase (lead + trail) · required',
    definitionZh: '外侧股骨上髁顶',
    definitionEn: 'lateral femoral epicondyle apex',
    blurb: 'the visible bony bump on the OUTSIDE of the knee at the joint line',
    views: {
      face_on: 'Outer side of the knee joint at the joint line, on the bone axis between thigh and shin',
      dtl: 'Camera-facing knee: the visible bony bump on the outside; for the far-side knee, estimate the symmetric point',
    },
    diagram: KneeDiagram,
    optional: false,
    warning:
      'NOT the patella (kneecap) center, NOT the inner side. The lateral epicondyle is the visible bony bump on the OUTSIDE of the knee. 是外侧骨突,不是膝盖骨中心,不是内侧。',
  },
  {
    key: 'ankle',
    title: 'Ankle · 踝',
    requirement: '2 per phase (lead + trail) · required',
    definitionZh: '外踝顶',
    definitionEn: 'lateral malleolus apex',
    blurb: 'the visible bony bump on the OUTSIDE of the ankle (distal fibula tip)',
    views: {
      face_on: 'Outer ankle bone bump on each side, on the leg bone axis',
      dtl: 'Camera-facing ankle: the bony bump on the outside; for the far-side ankle, estimate the symmetric point',
    },
    diagram: AnkleDiagram,
    optional: false,
    warning:
      'Bone-surface landmark (visible), NOT an internal joint center estimate (unlike hip). Click the bony bump, NOT the shoe edge, NOT the inner ankle. 是看得见的骨突,不像髋点是内部估计点。',
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// Page component (full-page standalone mode)
// ─────────────────────────────────────────────────────────────────────────────

export default function AnnotationGuidePage() {
  return (
    <main className="ag-page">
      <AnnotationGuideBody mode="standalone" />
      <style>{css}</style>
    </main>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Reusable body — also used as modal content from workbench.
// React-state-based acknowledgement: button disabled until ackChecked,
// ESC no-op until ackChecked (force user to read + tick).
// ─────────────────────────────────────────────────────────────────────────────

export function AnnotationGuideBody({
  mode,
  onAcknowledge,
}: {
  mode: 'standalone' | 'modal';
  onAcknowledge?: () => void;
}) {
  const [ackChecked, setAckChecked] = useState(false);

  // ESC handler — only effective once acknowledgement is checked.
  // No-op otherwise (force user to read and tick).
  useEffect(() => {
    if (mode !== 'modal') return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && ackChecked) {
        if (typeof window !== 'undefined') {
          localStorage.setItem(ANNOTATION_GUIDE_STORAGE_KEY, '1');
        }
        onAcknowledge?.();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [mode, ackChecked, onAcknowledge]);

  const handleStart = useCallback(() => {
    if (!ackChecked) return; // guard — should not trigger because button disabled
    if (typeof window !== 'undefined') {
      localStorage.setItem(ANNOTATION_GUIDE_STORAGE_KEY, '1');
    }
    onAcknowledge?.();
  }, [ackChecked, onAcknowledge]);

  return (
    <div className="ag-body">
      <p className="ag-eyebrow">Admin · Annotation guide</p>

      {/* HERO — neutral gray, not blue */}
      <section className="ag-hero">
        <h2 className="ag-hero-title">Annotation guide — bone-top-centerline</h2>
        <p className="ag-hero-text">
          Every keypoint = top-center of the <em>bone</em>, not muscle peak, not skin bulge.
          Bones are rigid and view-invariant; muscles move with camera angle, lighting, and clothing.
        </p>
      </section>

      {/* PRINCIPLE callout */}
      <section className="ag-callout">
        {/* Simple inline bone icon — replace with lucide-react Bone if the lib lands later. */}
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             strokeWidth="2" strokeLinecap="round" className="ag-callout-icon" aria-hidden="true">
          <path d="M17 10c.7-.7 1.69 0 2.5 0a2.5 2.5 0 1 0 0-5A2.5 2.5 0 0 0 17 7c0 .81-.7 1.8 0 2.5l-7 7c-.7.7-1.69 0-2.5 0a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 2.5-2.5c0-.81.7-1.8 0-2.5Z"/>
        </svg>
        <p className="ag-callout-text">
          <strong>Click the bone, not the body.</strong>{' '}
          If a click point would shift when the camera rotates 90°, it&apos;s wrong. The point
          you want is the joint&apos;s rotation axis — the bone end-centerline — which sits at
          the same anatomical place from any view.
        </p>
      </section>

      {/* LEGEND */}
      <div className="ag-legend">
        <span className="ag-legend-item">
          <span className="ag-legend-swatch" style={{ background: '#1D9E75' }}/>
          Correct click
        </span>
        <span className="ag-legend-item">
          <svg width="10" height="10" aria-hidden="true">
            <line x1="2" y1="2" x2="8" y2="8" stroke="#A32D2D" strokeWidth="1.5"/>
            <line x1="8" y1="2" x2="2" y2="8" stroke="#A32D2D" strokeWidth="1.5"/>
          </svg>
          Common wrong click
        </span>
        <span className="ag-legend-meta">solid line = bone · dashed = skin/muscle outline</span>
      </div>

      {/* 4 CARDS in 2x2 grid */}
      <div className="ag-grid">
        {CARDS.map((card) => {
          const Diagram = card.diagram;
          return (
            <article key={card.key} className="ag-card">
              <header className="ag-card-header">
                <h3 className="ag-card-title">{card.title}</h3>
                <span className="ag-card-req">{card.requirement}</span>
              </header>
              <p className="ag-card-def">
                <strong>{card.definitionZh}</strong>{' '}
                · {card.definitionEn} — {card.blurb}
              </p>
              {card.warning ? (
                <p className="ag-card-warning">⚠ {card.warning}</p>
              ) : null}
              <div className="ag-diagram-box">
                <Diagram />
              </div>
              <div className="ag-views">
                <span className="ag-view-label">FO</span>
                <span className="ag-view-text">{card.views.face_on}</span>
                <span className="ag-view-label">DTL</span>
                <span className="ag-view-text">{card.views.dtl}</span>
              </div>
            </article>
          );
        })}
      </div>

      {/* FOOTER */}
      <footer className="ag-footer">
        {mode === 'modal' ? (
          <>
            <label className="ag-ack-label">
              <input
                type="checkbox"
                className="ag-checkbox"
                checked={ackChecked}
                onChange={(e) => setAckChecked(e.target.checked)}
              />
              <span>
                I have read and understand these definitions.
                <span className="ag-ack-hint">30 tasks per video (10 arm + 5 hip + 5 head + 10 leg)</span>
              </span>
            </label>
            <button
              type="button"
              disabled={!ackChecked}
              onClick={handleStart}
              className="ag-start-btn"
            >
              Start annotating →
            </button>
          </>
        ) : (
          <p className="ag-standalone-msg">
            Read once, then start at{' '}
            <code className="ag-inline-code">/admin/annotate</code>.
          </p>
        )}
      </footer>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline CSS — neutral admin palette, mirrors Tailwind reference exactly.
// No brand blue. Green / red appear only inside SVG diagrams (correct vs
// wrong click). Used by both standalone page and the workbench modal.
// ─────────────────────────────────────────────────────────────────────────────

export const ANNOTATION_GUIDE_CSS = `
  .ag-page {
    max-width: 56rem;
    margin: 0 auto;
    padding: 2.5rem 1.5rem;
    color: #111827;
    font-family: 'DM Sans', system-ui, sans-serif;
    background: #ffffff;
    min-height: 100vh;
  }
  .ag-body { color: #111827; font-family: 'DM Sans', system-ui, sans-serif; }
  .ag-eyebrow { font-size: 0.75rem; color: #6b7280; margin: 0 0 0.75rem; }

  .ag-hero {
    border-radius: 0.75rem;
    background: #f3f4f6;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
  }
  .ag-hero-title {
    font-size: 1.125rem;
    font-weight: 500;
    color: #111827;
    margin: 0 0 0.5rem;
  }
  .ag-hero-text {
    font-size: 0.875rem;
    color: #374151;
    line-height: 1.6;
    margin: 0;
  }

  .ag-callout {
    border-radius: 0.375rem;
    background: #f9fafb;
    padding: 1rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
  }
  .ag-callout-icon {
    color: #6b7280;
    margin-top: 0.125rem;
    flex-shrink: 0;
  }
  .ag-callout-text {
    font-size: 0.875rem;
    color: #374151;
    line-height: 1.6;
    margin: 0;
  }
  .ag-callout-text strong { color: #111827; font-weight: 500; }

  .ag-legend {
    display: flex;
    align-items: center;
    gap: 1rem;
    font-size: 0.75rem;
    color: #4b5563;
    margin-bottom: 1.25rem;
    flex-wrap: wrap;
  }
  .ag-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
  }
  .ag-legend-swatch {
    width: 0.625rem;
    height: 0.625rem;
    border-radius: 9999px;
    display: inline-block;
  }
  .ag-legend-meta { color: #9ca3af; }

  .ag-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
  }
  @media (min-width: 768px) {
    .ag-grid { grid-template-columns: 1fr 1fr; }
  }

  .ag-card {
    border-radius: 0.75rem;
    border: 1px solid #e5e7eb;
    background: #ffffff;
    padding: 1rem;
  }
  .ag-card-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 0.5rem;
  }
  .ag-card-title { font-size: 1rem; font-weight: 500; margin: 0; }
  .ag-card-req { font-size: 0.75rem; color: #9ca3af; font-style: italic; }
  .ag-card-def {
    font-size: 0.875rem;
    color: #4b5563;
    line-height: 1.6;
    margin: 0 0 0.75rem;
  }
  .ag-card-def strong { color: #111827; font-weight: 500; }
  .ag-card-warning {
    font-size: 0.75rem;
    color: #1f2937;
    line-height: 1.6;
    margin: 0 0 0.75rem;
    border-left: 2px solid #9ca3af;
    padding-left: 0.75rem;
    font-style: italic;
  }
  .ag-diagram-box {
    width: 100%;
    height: 10rem;
    background: #f9fafb;
    border-radius: 0.375rem;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.75rem;
  }
  .ag-views {
    display: grid;
    grid-template-columns: 36px 1fr;
    column-gap: 0.5rem;
    row-gap: 0.25rem;
    font-size: 0.75rem;
  }
  .ag-view-label { color: #9ca3af; font-weight: 500; }
  .ag-view-text { color: #4b5563; line-height: 1.4; }

  .ag-footer {
    border-radius: 0.375rem;
    background: #f9fafb;
    padding: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
  }
  .ag-ack-label {
    display: flex;
    align-items: center;
    gap: 0.625rem;
    font-size: 0.875rem;
    color: #374151;
    cursor: pointer;
  }
  .ag-checkbox { width: 1rem; height: 1rem; cursor: pointer; }
  .ag-ack-hint { font-size: 0.75rem; color: #9ca3af; margin-left: 0.5rem; }

  .ag-start-btn {
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    border: 1px solid #d1d5db;
    background: #ffffff;
    color: #111827;
    font-size: 0.875rem;
    font-family: inherit;
    cursor: pointer;
    transition: background 0.12s;
  }
  .ag-start-btn:hover:not(:disabled) { background: #f9fafb; }
  .ag-start-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .ag-standalone-msg { font-size: 0.875rem; color: #6b7280; margin: 0; }
  .ag-inline-code {
    padding: 0.0625rem 0.375rem;
    background: #f3f4f6;
    border-radius: 0.25rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.8125rem;
  }
`;

const css = ANNOTATION_GUIDE_CSS;
