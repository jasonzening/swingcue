import type { SwingIssueType } from '@/lib/types/swing';

interface Props {
  issueType: SwingIssueType;
  width?: number;
  height?: number;
}

// Each overlay shows: dark background + red (current wrong) + green (target correct)
// Stylised stick-figure golf swing diagrams — illustrative, not photo-realistic
export function SwingOverlay({ issueType, width = 320, height = 400 }: Props) {
  const W = width;
  const H = height;

  switch (issueType) {
    case 'early_extension':
      return <EarlyExtensionOverlay W={W} H={H} />;
    case 'steep_downswing':
      return <SteepDownswingOverlay W={W} H={H} />;
    case 'head_movement':
      return <HeadMovementOverlay W={W} H={H} />;
    case 'weight_shift_issue':
      return <WeightShiftOverlay W={W} H={H} />;
    case 'hand_path_issue':
      return <HandPathOverlay W={W} H={H} />;
    case 'steep_backswing_plane':
      return <SteepBackswingOverlay W={W} H={H} />;
    default:
      return <EarlyExtensionOverlay W={W} H={H} />;
  }
}

/* ─── SHARED HELPERS ─── */
const RED = '#ff4040';
const GREEN = '#40e040';
const RED_DIM = 'rgba(255,64,64,0.25)';
const GREEN_DIM = 'rgba(64,224,64,0.25)';

function Label({ x, y, text, color }: { x: number; y: number; text: string; color: string }) {
  return (
    <text x={x} y={y} fill={color} fontSize={11} fontFamily="DM Sans, system-ui" fontWeight="700"
      textAnchor="middle" style={{ textShadow: '0 1px 2px #000' }}>
      {text}
    </text>
  );
}

function Legend({ W, H }: { W: number; H: number }) {
  return (
    <g>
      <rect x={10} y={H - 38} width={120} height={28} rx={6} fill="rgba(0,0,0,0.6)" />
      <circle cx={24} cy={H - 24} r={5} fill={RED} />
      <text x={33} y={H - 20} fill={RED} fontSize={10} fontWeight="700" fontFamily="DM Sans, system-ui">Current</text>
      <circle cx={80} cy={H - 24} r={5} fill={GREEN} />
      <text x={89} y={H - 20} fill={GREEN} fontSize={10} fontWeight="700" fontFamily="DM Sans, system-ui">Target</text>
    </g>
  );
}

/* ─── EARLY EXTENSION ─── */
// Body rises / hips thrust toward ball at impact
function EarlyExtensionOverlay({ W, H }: { W: number; H: number }) {
  const cx = W / 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ display: 'block' }}>
      <rect width={W} height={H} fill="#0a1008" rx={12} />

      {/* Ground line */}
      <line x1={20} y1={H - 60} x2={W - 20} y2={H - 60} stroke="rgba(255,255,255,0.15)" strokeWidth={2} />

      {/* RED — current: body rises, hips thrust out, spine angle lost */}
      <g opacity={0.9}>
        {/* Hip thrust out — hips pushed toward camera */}
        <ellipse cx={cx - 12} cy={H - 130} rx={28} ry={14} fill={RED_DIM} stroke={RED} strokeWidth={2} />
        {/* Spine — upright, lost angle */}
        <line x1={cx - 12} y1={H - 130} x2={cx - 4} y2={H - 280} stroke={RED} strokeWidth={3} strokeLinecap="round" />
        {/* Head — risen up */}
        <circle cx={cx - 4} cy={H - 295} r={18} fill="none" stroke={RED} strokeWidth={2.5} />
        {/* Shoulders — lifted, tilted wrong */}
        <line x1={cx - 48} y1={H - 268} x2={cx + 28} y2={H - 262} stroke={RED} strokeWidth={3} strokeLinecap="round" />
        {/* Trail arm */}
        <line x1={cx - 48} y1={H - 268} x2={cx - 30} y2={H - 200} stroke={RED} strokeWidth={2.5} strokeLinecap="round" />
        {/* Club path — steep */}
        <path d={`M${cx - 30} ${H - 200} Q${cx + 10} ${H - 155} ${cx + 40} ${H - 95}`}
          fill="none" stroke={RED} strokeWidth={2.5} strokeDasharray="5,3" strokeLinecap="round" />
        <Label x={cx + 60} y={H - 120} text="Rises" color={RED} />
      </g>

      {/* GREEN — target: maintain spine angle, hips rotate not thrust */}
      <g opacity={0.9}>
        {/* Hips — rotating correctly, not thrusting */}
        <ellipse cx={cx + 60} cy={H - 120} rx={22} ry={12} fill={GREEN_DIM} stroke={GREEN} strokeWidth={2} />
        {/* Spine — maintained angle through impact */}
        <line x1={cx + 60} y1={H - 120} x2={cx + 52} y2={H - 280} stroke={GREEN} strokeWidth={3} strokeLinecap="round" strokeDasharray="6,3" />
        {/* Head — stays down */}
        <circle cx={cx + 52} cy={H - 296} r={18} fill="none" stroke={GREEN} strokeWidth={2.5} strokeDasharray="5,3" />
        {/* Shoulders — better plane */}
        <line x1={cx + 22} y1={H - 262} x2={cx + 82} y2={H - 270} stroke={GREEN} strokeWidth={3} strokeLinecap="round" strokeDasharray="6,3" />
        {/* Club path — shallower */}
        <path d={`M${cx + 22} ${H - 240} Q${cx + 60} ${H - 170} ${cx + 72} ${H - 95}`}
          fill="none" stroke={GREEN} strokeWidth={2.5} strokeDasharray="5,3" strokeLinecap="round" />
        <Label x={cx + 95} y={H - 80} text="Stay down" color={GREEN} />
      </g>

      {/* Angle indicator — spine stays forward */}
      <path d={`M${cx + 52} ${H - 130} A30 30 0 0 0 ${cx + 52} ${H - 160}`}
        fill="none" stroke={GREEN} strokeWidth={1.5} strokeDasharray="3,2" />

      <Legend W={W} H={H} />
    </svg>
  );
}

/* ─── STEEP DOWNSWING ─── */
// Over-the-top swing path
function SteepDownswingOverlay({ W, H }: { W: number; H: number }) {
  const cx = W / 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ display: 'block' }}>
      <rect width={W} height={H} fill="#0a1008" rx={12} />
      <line x1={20} y1={H - 60} x2={W - 20} y2={H - 60} stroke="rgba(255,255,255,0.15)" strokeWidth={2} />

      {/* Body center — shared */}
      <circle cx={cx} cy={H - 290} r={18} fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth={1.5} />
      <line x1={cx} y1={H - 272} x2={cx} y2={H - 120} stroke="rgba(255,255,255,0.2)" strokeWidth={2} />
      <line x1={cx - 30} y1={H - 260} x2={cx + 30} y2={H - 252} stroke="rgba(255,255,255,0.2)" strokeWidth={1.5} />

      {/* RED — steep over-the-top path */}
      <g opacity={0.9}>
        <path d={`M${cx - 80} ${H - 320} Q${cx - 20} ${H - 240} ${cx + 20} ${H - 80}`}
          fill="none" stroke={RED} strokeWidth={3.5} strokeLinecap="round" />
        <path d={`M${cx + 20} ${H - 80} L${cx + 35} ${H - 65}`}
          fill="none" stroke={RED} strokeWidth={3} strokeLinecap="round" />
        {/* Arrow head on steep path */}
        <polygon points={`${cx + 35},${H - 65} ${cx + 25},${H - 78} ${cx + 40},${H - 80}`} fill={RED} />
        <text x={cx - 90} y={H - 330} fill={RED} fontSize={11} fontWeight="700" fontFamily="DM Sans, system-ui">Over-the-top</text>
        <Label x={cx - 50} y={H - 240} text="Steep path" color={RED} />
      </g>

      {/* GREEN — inside-out shallower path */}
      <g opacity={0.9}>
        <path d={`M${cx - 30} ${H - 300} Q${cx + 40} ${H - 210} ${cx + 55} ${H - 80}`}
          fill="none" stroke={GREEN} strokeWidth={3.5} strokeDasharray="8,4" strokeLinecap="round" />
        <path d={`M${cx + 55} ${H - 80} L${cx + 70} ${H - 65}`}
          fill="none" stroke={GREEN} strokeWidth={3} strokeLinecap="round" />
        <polygon points={`${cx + 70},${H - 65} ${cx + 57},${H - 75} ${cx + 72},${H - 78}`} fill={GREEN} />
        <Label x={cx + 90} y={H - 190} text="Shallow" color={GREEN} />
      </g>

      {/* Swing plane indicator */}
      <line x1={cx - 90} y1={H - 330} x2={cx + 80} y2={H - 60}
        stroke="rgba(255,255,255,0.12)" strokeWidth={1.5} strokeDasharray="4,4" />
      <text x={cx + 85} y={H - 55} fill="rgba(255,255,255,0.35)" fontSize={9} fontFamily="DM Sans, system-ui">Correct plane</text>

      <Legend W={W} H={H} />
    </svg>
  );
}

/* ─── HEAD MOVEMENT ─── */
function HeadMovementOverlay({ W, H }: { W: number; H: number }) {
  const cx = W / 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ display: 'block' }}>
      <rect width={W} height={H} fill="#0a1008" rx={12} />
      <line x1={20} y1={H - 60} x2={W - 20} y2={H - 60} stroke="rgba(255,255,255,0.15)" strokeWidth={2} />

      {/* Center line — head should stay here */}
      <line x1={cx} y1={40} x2={cx} y2={H - 60}
        stroke="rgba(64,224,64,0.3)" strokeWidth={1.5} strokeDasharray="6,4" />
      <text x={cx + 6} y={55} fill="rgba(64,224,64,0.6)" fontSize={9} fontFamily="DM Sans, system-ui">Stay here</text>

      {/* RED — head drifting left (away from target) during backswing */}
      <g opacity={0.9}>
        <circle cx={cx - 55} cy={H - 295} r={20} fill={RED_DIM} stroke={RED} strokeWidth={2.5} />
        <line x1={cx - 55} y1={H - 275} x2={cx - 45} y2={H - 140} stroke={RED} strokeWidth={3} strokeLinecap="round" />
        <line x1={cx - 78} y1={H - 260} x2={cx - 22} y2={H - 250} stroke={RED} strokeWidth={2.5} strokeLinecap="round" />
        {/* Hip line */}
        <line x1={cx - 68} y1={H - 165} x2={cx - 18} y2={H - 155} stroke={RED} strokeWidth={2} strokeLinecap="round" />
        {/* Arrow showing movement */}
        <path d={`M${cx - 10} ${H - 295} L${cx - 45} ${H - 295}`}
          fill="none" stroke={RED} strokeWidth={2} markerEnd="url(#arrowRed)" />
        <defs>
          <marker id="arrowRed" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
            <polygon points="0 0, 6 3, 0 6" fill={RED} />
          </marker>
        </defs>
        <Label x={cx - 80} y={H - 315} text="Drifts" color={RED} />
      </g>

      {/* GREEN — head stays centered */}
      <g opacity={0.9}>
        <circle cx={cx + 40} cy={H - 295} r={20} fill={GREEN_DIM} stroke={GREEN} strokeWidth={2.5} strokeDasharray="5,3" />
        <line x1={cx + 40} y1={H - 275} x2={cx + 35} y2={H - 140} stroke={GREEN} strokeWidth={3} strokeDasharray="6,3" strokeLinecap="round" />
        <line x1={cx + 15} y1={H - 260} x2={cx + 65} y2={H - 250} stroke={GREEN} strokeWidth={2.5} strokeDasharray="5,3" strokeLinecap="round" />
        <line x1={cx + 12} y1={H - 165} x2={cx + 60} y2={H - 155} stroke={GREEN} strokeWidth={2} strokeDasharray="4,2" strokeLinecap="round" />
        <Label x={cx + 80} y={H - 315} text="Centered" color={GREEN} />
      </g>

      {/* Lateral drift indicator */}
      <path d={`M${cx} ${H - 200} Q${cx - 30} ${H - 230} ${cx - 55} ${H - 260}`}
        fill="none" stroke={RED} strokeWidth={1.5} strokeDasharray="3,2" opacity={0.6} />

      <Legend W={W} H={H} />
    </svg>
  );
}

/* ─── WEIGHT SHIFT (REVERSE PIVOT) ─── */
function WeightShiftOverlay({ W, H }: { W: number; H: number }) {
  const cx = W / 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ display: 'block' }}>
      <rect width={W} height={H} fill="#0a1008" rx={12} />
      <line x1={20} y1={H - 60} x2={W - 20} y2={H - 60} stroke="rgba(255,255,255,0.15)" strokeWidth={2} />

      {/* Foot markers */}
      <ellipse cx={cx - 50} cy={H - 62} rx={22} ry={7} fill="rgba(255,255,255,0.08)" stroke="rgba(255,255,255,0.2)" strokeWidth={1} />
      <ellipse cx={cx + 50} cy={H - 62} rx={22} ry={7} fill="rgba(255,255,255,0.08)" stroke="rgba(255,255,255,0.2)" strokeWidth={1} />
      <text x={cx - 50} y={H - 45} fill="rgba(255,255,255,0.3)" fontSize={9} textAnchor="middle" fontFamily="DM Sans, system-ui">Lead</text>
      <text x={cx + 50} y={H - 45} fill="rgba(255,255,255,0.3)" fontSize={9} textAnchor="middle" fontFamily="DM Sans, system-ui">Trail</text>

      {/* RED — weight stays back (reverse pivot), no shift through */}
      <g opacity={0.9}>
        {/* Body leaning back */}
        <circle cx={cx + 25} cy={H - 285} r={19} fill={RED_DIM} stroke={RED} strokeWidth={2.5} />
        <line x1={cx + 25} y1={H - 266} x2={cx + 40} y2={H - 130} stroke={RED} strokeWidth={3} strokeLinecap="round" />
        <line x1={cx - 2} y1={H - 250} x2={cx + 52} y2={H - 242} stroke={RED} strokeWidth={2.5} strokeLinecap="round" />
        <line x1={cx + 12} y1={H - 155} x2={cx + 60} y2={H - 148} stroke={RED} strokeWidth={2} strokeLinecap="round" />
        {/* Weight indicator — on back foot */}
        <ellipse cx={cx + 50} cy={H - 72} rx={22} ry={7} fill={RED} opacity={0.5} />
        <text x={cx + 50} y={H - 70} fill={RED} fontSize={9} textAnchor="middle" fontWeight="700" fontFamily="DM Sans, system-ui">80%</text>
        <Label x={cx + 60} y={H - 305} text="Stuck back" color={RED} />
      </g>

      {/* GREEN — weight shifts to lead side */}
      <g opacity={0.9}>
        <circle cx={cx - 35} cy={H - 285} r={19} fill={GREEN_DIM} stroke={GREEN} strokeWidth={2.5} strokeDasharray="5,3" />
        <line x1={cx - 35} y1={H - 266} x2={cx - 24} y2={H - 130} stroke={GREEN} strokeWidth={3} strokeDasharray="6,3" strokeLinecap="round" />
        <line x1={cx - 58} y1={H - 250} x2={cx - 10} y2={H - 244} stroke={GREEN} strokeWidth={2.5} strokeDasharray="5,3" strokeLinecap="round" />
        <line x1={cx - 50} y1={H - 155} x2={cx - 5} y2={H - 148} stroke={GREEN} strokeWidth={2} strokeDasharray="4,2" strokeLinecap="round" />
        {/* Weight on lead foot */}
        <ellipse cx={cx - 50} cy={H - 72} rx={22} ry={7} fill={GREEN} opacity={0.5} />
        <text x={cx - 50} y={H - 70} fill={GREEN} fontSize={9} textAnchor="middle" fontWeight="700" fontFamily="DM Sans, system-ui">70%</text>
        {/* Arrow showing weight direction */}
        <path d={`M${cx + 30} ${H - 130} Q${cx} ${H - 110} ${cx - 30} ${H - 120}`}
          fill="none" stroke={GREEN} strokeWidth={2} strokeDasharray="4,2" />
        <polygon points={`${cx - 30},${H - 120} ${cx - 20},${H - 112} ${cx - 18},${H - 126}`} fill={GREEN} />
        <Label x={cx - 65} y={H - 305} text="Shift through" color={GREEN} />
      </g>

      <Legend W={W} H={H} />
    </svg>
  );
}

/* ─── HAND PATH ─── */
function HandPathOverlay({ W, H }: { W: number; H: number }) {
  const cx = W / 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ display: 'block' }}>
      <rect width={W} height={H} fill="#0a1008" rx={12} />
      <line x1={20} y1={H - 60} x2={W - 20} y2={H - 60} stroke="rgba(255,255,255,0.15)" strokeWidth={2} />

      {/* Body */}
      <circle cx={cx} cy={H - 290} r={18} fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth={1.5} />
      <line x1={cx} y1={H - 272} x2={cx} y2={H - 130} stroke="rgba(255,255,255,0.2)" strokeWidth={2} />

      {/* RED — hands looping out/away from body */}
      <g opacity={0.9}>
        <path d={`M${cx + 20} ${H - 255} C${cx + 80} ${H - 200} ${cx + 90} ${H - 130} ${cx + 30} ${H - 90}`}
          fill="none" stroke={RED} strokeWidth={3.5} strokeLinecap="round" />
        <polygon points={`${cx + 30},${H - 90} ${cx + 42},${H - 103} ${cx + 20},${H - 100}`} fill={RED} />
        <circle cx={cx + 20} cy={H - 255} r={6} fill={RED} />
        <Label x={cx + 100} y={H - 165} text="Away from body" color={RED} />
      </g>

      {/* GREEN — hands tight to body, drop into slot */}
      <g opacity={0.9}>
        <path d={`M${cx + 20} ${H - 255} C${cx + 15} ${H - 200} ${cx + 10} ${H - 140} ${cx - 5} ${H - 90}`}
          fill="none" stroke={GREEN} strokeWidth={3.5} strokeDasharray="7,4" strokeLinecap="round" />
        <polygon points={`${cx - 5},${H - 90} ${cx + 5},${H - 104} ${cx - 18},${H - 100}`} fill={GREEN} />
        <circle cx={cx + 20} cy={H - 255} r={6} fill={GREEN} opacity={0.7} />
        <Label x={cx - 55} y={H - 165} text="Close to body" color={GREEN} />
      </g>

      {/* Distance indicator */}
      <line x1={cx + 30} y1={H - 160} x2={cx + 82} y2={H - 160}
        stroke={RED} strokeWidth={1.5} strokeDasharray="3,2" opacity={0.6} />
      <text x={cx + 56} y={H - 148} fill="rgba(255,255,255,0.3)" fontSize={9} textAnchor="middle" fontFamily="DM Sans, system-ui">gap</text>

      <Legend W={W} H={H} />
    </svg>
  );
}

/* ─── STEEP BACKSWING PLANE ─── */
function SteepBackswingOverlay({ W, H }: { W: number; H: number }) {
  const cx = W / 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ display: 'block' }}>
      <rect width={W} height={H} fill="#0a1008" rx={12} />
      <line x1={20} y1={H - 60} x2={W - 20} y2={H - 60} stroke="rgba(255,255,255,0.15)" strokeWidth={2} />

      {/* Body */}
      <circle cx={cx} cy={H - 290} r={18} fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth={1.5} />
      <line x1={cx} y1={H - 272} x2={cx + 5} y2={H - 135} stroke="rgba(255,255,255,0.2)" strokeWidth={2} />
      <line x1={cx - 30} y1={H - 258} x2={cx + 30} y2={H - 250} stroke="rgba(255,255,255,0.2)" strokeWidth={1.5} />

      {/* Correct plane guide line */}
      <line x1={cx + 20} y1={H - 80} x2={cx - 60} y2={H - 320}
        stroke="rgba(255,255,255,0.1)" strokeWidth={1.5} strokeDasharray="6,4" />
      <text x={cx - 65} y={H - 328} fill="rgba(255,255,255,0.3)" fontSize={9} fontFamily="DM Sans, system-ui">Ideal plane</text>

      {/* RED — club goes too vertical / steep */}
      <g opacity={0.9}>
        <path d={`M${cx + 20} ${H - 90} L${cx - 10} ${H - 340}`}
          fill="none" stroke={RED} strokeWidth={3.5} strokeLinecap="round" />
        <circle cx={cx - 12} cy={H - 348} r={7} fill={RED} opacity={0.8} />
        {/* Too steep indicator */}
        <path d={`M${cx - 12} ${H - 340} Q${cx - 5} ${H - 310} ${cx + 5} ${H - 280}`}
          fill="none" stroke={RED} strokeWidth={2} strokeDasharray="4,2" />
        <Label x={cx - 40} y={H - 355} text="Too steep" color={RED} />
      </g>

      {/* GREEN — flatter backswing plane */}
      <g opacity={0.9}>
        <path d={`M${cx + 20} ${H - 90} L${cx - 70} ${H - 300}`}
          fill="none" stroke={GREEN} strokeWidth={3.5} strokeDasharray="8,4" strokeLinecap="round" />
        <circle cx={cx - 72} cy={H - 308} r={7} fill={GREEN} opacity={0.8} />
        <Label x={cx - 100} y={H - 318} text="Flatter" color={GREEN} />
      </g>

      {/* Angle arc showing difference */}
      <path d={`M${cx + 20} ${H - 160} A80 80 0 0 0 ${cx - 30} ${H - 205}`}
        fill="none" stroke="rgba(255,200,64,0.6)" strokeWidth={2} strokeDasharray="4,2" />
      <text x={cx + 30} y={H - 175} fill="rgba(255,200,64,0.8)" fontSize={10} fontFamily="DM Sans, system-ui">~15°</text>

      <Legend W={W} H={H} />
    </svg>
  );
}
