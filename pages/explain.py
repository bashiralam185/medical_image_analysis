"""pages/explain.py — Animated pipeline explanations"""
import streamlit as st
import streamlit.components.v1 as components
import utils

# ── Shared CSS ────────────────────────────────────────────────────────────────
BASE_CSS = """
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:transparent;font-family:'Segoe UI',system-ui,sans-serif;color:#c8d8e8;padding:10px}
canvas{display:block}
h2{font-family:monospace;font-size:.78rem;letter-spacing:2px;color:#00c8ff;margin-bottom:10px}
.row{display:flex;gap:8px;margin-bottom:8px;flex-wrap:wrap}
.card{background:#0e1520;border:1px solid #1e2d42;border-radius:6px;padding:12px;flex:1;min-width:160px}
.label{font-size:.65rem;font-family:monospace;letter-spacing:1px;color:#5a7080;margin-bottom:5px}
.val{font-family:monospace;font-size:1.1rem;color:#00c8ff;font-weight:700}
.sub{font-size:.7rem;color:#5a7080;margin-top:2px}
.btn-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
button{background:transparent;border:1px solid #00c8ff;color:#00c8ff;
       border-radius:4px;padding:5px 14px;cursor:pointer;font-size:.73rem;font-family:monospace}
button:hover{background:#00c8ff;color:#0a0e14}
button.sec{border-color:#1e2d42;color:#5a7080}
button.sec:hover{background:#1e2d42;color:#c8d8e8}
.desc{font-size:.78rem;color:#8aaccc;line-height:1.7;margin-top:6px}
input[type=range]{accent-color:#00c8ff;width:100%}
.ctrl-row{display:flex;align-items:center;gap:10px;font-size:.75rem;color:#5a7080;margin-top:6px}
.ctrl-row span.val2{min-width:44px;color:#00c8ff;font-family:monospace}
.formula{font-family:monospace;font-size:.88rem;color:#ffb547;text-align:center;
          background:#080c10;border:1px solid #1e2d42;border-radius:4px;padding:8px;margin:6px 0}
.pill{display:inline-block;font-size:.68rem;padding:2px 8px;border-radius:10px;
      font-family:monospace;margin:2px}
</style>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — COREGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════
COREG_HTML = BASE_CSS + """
<h2>3D RIGID COREGISTRATION — HOW OUR ALGORITHM WORKS</h2>

<div class="row">
  <!-- Left: animated main canvas -->
  <div style="flex:1.8;min-width:280px">
    <div class="card" style="padding:8px">
      <canvas id="mainCv" width="400" height="300" style="border-radius:4px;background:#080c10;width:100%"></canvas>
    </div>
    <div class="btn-row">
      <button onclick="showScene(0)">① Initialise</button>
      <button onclick="showScene(1)">② Pyramid</button>
      <button onclick="showScene(2)">③ Optimise</button>
      <button onclick="showScene(3)">④ Result</button>
      <button class="sec" onclick="autoPlay()">▶ Auto-play all</button>
    </div>
    <div id="sceneDesc" class="desc" style="margin-top:8px;min-height:44px"></div>
  </div>

  <!-- Right: live metrics panel -->
  <div style="flex:1;min-width:180px;display:flex;flex-direction:column;gap:6px">

    <div class="card">
      <div class="label">TRANSFORM — 6 DOF</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:4px">
        <div id="d_rx" style="background:#080c10;border:1px solid #a084e822;border-radius:3px;padding:5px;text-align:center">
          <div style="font-size:.62rem;color:#a084e8">θ<sub>x</sub> rotation</div>
          <div id="v_rx" style="font-family:monospace;font-size:.9rem;color:#a084e8">0.00°</div>
        </div>
        <div style="background:#080c10;border:1px solid #a084e822;border-radius:3px;padding:5px;text-align:center">
          <div style="font-size:.62rem;color:#a084e8">θ<sub>y</sub> rotation</div>
          <div id="v_ry" style="font-family:monospace;font-size:.9rem;color:#a084e8">0.00°</div>
        </div>
        <div style="background:#080c10;border:1px solid #a084e822;border-radius:3px;padding:5px;text-align:center">
          <div style="font-size:.62rem;color:#a084e8">θ<sub>z</sub> rotation</div>
          <div id="v_rz" style="font-family:monospace;font-size:.9rem;color:#a084e8">0.00°</div>
        </div>
        <div style="background:#080c10;border:1px solid #378ADD22;border-radius:3px;padding:5px;text-align:center">
          <div style="font-size:.62rem;color:#378ADD">t<sub>x</sub> translation</div>
          <div id="v_tx" style="font-family:monospace;font-size:.9rem;color:#378ADD">0.0mm</div>
        </div>
        <div style="background:#080c10;border:1px solid #378ADD22;border-radius:3px;padding:5px;text-align:center">
          <div style="font-size:.62rem;color:#378ADD">t<sub>y</sub> translation</div>
          <div id="v_ty" style="font-family:monospace;font-size:.9rem;color:#378ADD">0.0mm</div>
        </div>
        <div style="background:#080c10;border:1px solid #378ADD22;border-radius:3px;padding:5px;text-align:center">
          <div style="font-size:.62rem;color:#378ADD">t<sub>z</sub> translation</div>
          <div id="v_tz" style="font-family:monospace;font-size:.9rem;color:#378ADD">0.0mm</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="label">MATTES MI — LIVE</div>
      <canvas id="miCv" width="220" height="90" style="background:#080c10;border-radius:4px;width:100%;margin-top:4px"></canvas>
      <div style="display:flex;justify-content:space-between;margin-top:4px">
        <span style="font-size:.65rem;color:#5a7080">MI: <span id="miVal" style="color:#00c8ff;font-family:monospace">—</span></span>
        <span style="font-size:.65rem;color:#5a7080">NMI: <span id="nmiVal" style="color:#00e5a0;font-family:monospace">—</span></span>
        <span style="font-size:.65rem;color:#5a7080">iter: <span id="iterVal" style="color:#ffb547;font-family:monospace">—</span></span>
      </div>
    </div>

    <div class="card">
      <div class="label">PYRAMID LEVEL</div>
      <div id="pyrVis" style="margin-top:6px"></div>
    </div>

  </div>
</div>

<script>
const CV = document.getElementById('mainCv');
const ctx = CV.getContext('2d');
const W = CV.width, H = CV.height;

// ── MI convergence canvas ──────────────────────────────────────────────────
const mCv = document.getElementById('miCv');
const mCtx = mCv.getContext('2d');
const miHistory = [];

function drawMIcurve(current){
  const w = mCv.width, h = mCv.height;
  mCtx.clearRect(0,0,w,h);
  mCtx.fillStyle='#080c10'; mCtx.fillRect(0,0,w,h);
  if(miHistory.length < 2) return;
  const maxV = Math.max(...miHistory);
  const minV = Math.min(...miHistory);
  const pad = {l:24,r:6,t:8,b:18};
  const iw = w-pad.l-pad.r, ih = h-pad.t-pad.b;
  function px(i){ return pad.l + (i/(miHistory.length-1))*iw; }
  function py(v){ return pad.t + (1-(v-minV)/(maxV-minV+.001))*ih; }
  // grid
  mCtx.strokeStyle='#1e2d42'; mCtx.lineWidth=.4;
  [.25,.5,.75,1].forEach(t=>{
    mCtx.beginPath(); mCtx.moveTo(pad.l, pad.t+t*ih); mCtx.lineTo(pad.l+iw, pad.t+t*ih); mCtx.stroke();
  });
  // curve
  mCtx.beginPath();
  miHistory.forEach((v,i)=>{ if(i===0)mCtx.moveTo(px(i),py(v)); else mCtx.lineTo(px(i),py(v)); });
  mCtx.strokeStyle='#00c8ff'; mCtx.lineWidth=1.5; mCtx.stroke();
  // dot at end
  if(miHistory.length>0){
    const last = miHistory.length-1;
    mCtx.beginPath(); mCtx.arc(px(last),py(miHistory[last]),3,0,Math.PI*2);
    mCtx.fillStyle='#00c8ff'; mCtx.fill();
  }
  // axes labels
  mCtx.fillStyle='#5a7080'; mCtx.font='7px sans-serif'; mCtx.textAlign='center';
  mCtx.fillText('iterations →', pad.l+iw/2, h-2);
  mCtx.save(); mCtx.translate(8, pad.t+ih/2); mCtx.rotate(-Math.PI/2);
  mCtx.fillText('MI', 0, 0); mCtx.restore();
}

// ── Pyramid visualiser ─────────────────────────────────────────────────────
function drawPyramid(activeLevel){
  const el = document.getElementById('pyrVis');
  const levels = [
    {label:'Level 1 — 4× shrink, σ=2mm', col:'#D85A30', w:'40%'},
    {label:'Level 2 — 2× shrink, σ=1mm', col:'#EF9F27', w:'65%'},
    {label:'Level 3 — full res, σ=0mm',  col:'#1D9E75', w:'95%'},
  ];
  el.innerHTML = levels.map((l,i)=>`
    <div style="width:${l.w};margin:0 auto 4px auto;padding:5px 8px;border-radius:4px;
      background:${l.col}${i===activeLevel?'22':'0a'};
      border:1px solid ${l.col}${i===activeLevel?'':'44'};
      text-align:center;font-family:monospace;font-size:.68rem;
      color:${i===activeLevel?l.col:'#5a7080'};
      font-weight:${i===activeLevel?'700':'400'};
      transition:all .3s">
      ${l.label}
    </div>`).join('');
}
drawPyramid(-1);

// ── Helper draw functions ──────────────────────────────────────────────────
function clear(){ ctx.clearRect(0,0,W,H); ctx.fillStyle='#080c10'; ctx.fillRect(0,0,W,H); }

function brain(cx,cy,rx,ry,col,offset,blur){
  ctx.save();
  if(blur>0){ ctx.filter='blur('+blur+'px)'; }
  ctx.translate(cx+offset,cy);
  // outline
  ctx.beginPath(); ctx.ellipse(0,0,rx,ry,0,0,Math.PI*2);
  ctx.fillStyle=col+'18'; ctx.fill();
  ctx.strokeStyle=col; ctx.lineWidth=1.8; ctx.stroke();
  // hemispheres
  ctx.beginPath(); ctx.ellipse(-rx*.22,-ry*.15,rx*.38,ry*.28,-.18,0,Math.PI*2);
  ctx.fillStyle=col+'28'; ctx.fill();
  ctx.beginPath(); ctx.ellipse(rx*.22,-ry*.15,rx*.35,ry*.26,.18,0,Math.PI*2);
  ctx.fillStyle=col+'28'; ctx.fill();
  // brain stem
  ctx.beginPath(); ctx.ellipse(0,ry*.6,rx*.1,ry*.18,0,0,Math.PI*2);
  ctx.fillStyle=col+'44'; ctx.fill();
  ctx.restore();
}

function crosshair(cx,cy,col,size,alpha){
  ctx.save(); ctx.globalAlpha=alpha;
  ctx.strokeStyle=col; ctx.lineWidth=1; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(cx-size,cy); ctx.lineTo(cx+size,cy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx,cy-size); ctx.lineTo(cx,cy+size); ctx.stroke();
  ctx.setLineDash([]); ctx.restore();
}

function label(x,y,text,col,size){
  ctx.fillStyle=col; ctx.font=(size||10)+'px monospace';
  ctx.textAlign='center'; ctx.fillText(text,x,y);
}

// ── Scene 0: Initialisation ────────────────────────────────────────────────
let initAnim=null, initT=0;
function scene0(){
  stopAll();
  document.getElementById('sceneDesc').innerHTML=
    '<b style="color:#00c8ff">Step 1 — Geometry-based centre alignment.</b> '
    +'Before any optimisation, we compute the geometric centres of both images and '
    +'translate the PET so they overlap. This puts the optimiser well within the capture range '
    +'of the correct solution. Without this step the 50–100mm scanner offset would derail gradient descent.';
  drawPyramid(-1);
  ['rx','ry','rz'].forEach(k=>document.getElementById('v_'+k).textContent='0.00°');
  ['tx','ty','tz'].forEach(k=>document.getElementById('v_'+k).textContent='0.0mm');
  document.getElementById('miVal').textContent='—';
  document.getElementById('nmiVal').textContent='—';
  document.getElementById('iterVal').textContent='—';

  let petOffset = 90; // starts far right
  const target = 0;

  function frame(){
    clear();
    // MRI — fixed
    brain(160, 150, 80, 56, '#1D9E75', 0, 0);
    crosshair(160, 150, '#1D9E75', 20, .6);
    label(160, 222, 'MRI (fixed)', '#1D9E75', 9);

    // PET — moving toward centre
    brain(240, 150, 70, 50, '#D85A30', petOffset, 0);
    crosshair(240+petOffset, 150, '#D85A30', 18, .6);
    label(240+petOffset, 222, 'PET (moving)', '#D85A30', 9);

    // offset arrow
    if(Math.abs(petOffset)>2){
      ctx.strokeStyle='#ffb547'; ctx.lineWidth=1; ctx.setLineDash([3,2]);
      ctx.beginPath(); ctx.moveTo(160,150); ctx.lineTo(240+petOffset,150); ctx.stroke();
      ctx.setLineDash([]);
      label(200+petOffset/2, 140, 'offset: '+Math.round(petOffset)+'mm', '#ffb547', 8);
    }

    // update param display
    document.getElementById('v_tx').textContent = Math.round(petOffset)+'.0mm';

    if(petOffset>target+.5){
      petOffset -= 1.2;
      initAnim = requestAnimationFrame(frame);
    } else {
      petOffset=0;
      // draw final aligned state
      clear();
      brain(200, 150, 80, 56, '#1D9E75', 0, 0);
      brain(200, 150, 72, 51, '#D85A30', 0, 0);
      crosshair(200, 150, '#00e5a0', 24, .9);
      label(200, 222, 'centres aligned  ✓', '#00e5a0', 10);
      document.getElementById('v_tx').textContent='0.0mm';
    }
  }
  frame();
}

// ── Scene 1: Pyramid ────────────────────────────────────────────────────────
let pyrAnim=null, pyrStep=0;
function scene1(){
  stopAll(); pyrStep=0;
  document.getElementById('sceneDesc').innerHTML=
    '<b style="color:#EF9F27">Step 2 — Multi-resolution pyramid.</b> '
    +'We run registration 3 times, coarse→fine. At the coarsest level (4× blur) '
    +'only the whole-brain shape is visible — one clear MI peak, impossible to miss. '
    +'Each finer level starts from the previous result and adds precision. '
    +'Without this, full-resolution images have many local MI maxima that trap the optimiser.';

  const blurs = [12, 4, 0];
  const offsets = [35, 12, 0];
  const colors = ['#D85A30','#EF9F27','#1D9E75'];
  const labels2 = ['Level 1 — coarsest (4×)', 'Level 2 — medium (2×)', 'Level 3 — full res'];
  const miVals = [.24, .48, .72];

  function drawLevel(lv, progress){
    clear();
    const blur = blurs[lv] * (1-progress);
    const off  = offsets[lv] * (1-progress);
    const col  = colors[lv];
    drawPyramid(lv);

    // Show blurred PET sliding into place
    brain(160, 150, 78, 54, '#1D9E75', 0, blurs[lv]*(1-progress));
    brain(240, 150, 70, 50, col, off, blurs[lv]*(1-progress));

    // Level label
    ctx.fillStyle=col; ctx.font='bold 11px monospace'; ctx.textAlign='center';
    ctx.fillText(labels2[lv], W/2, 42);

    // sigma and shrink annotations
    ctx.fillStyle='#5a7080'; ctx.font='9px sans-serif';
    ctx.fillText(['4× shrink  σ=2mm  — only large structures visible',
                  '2× shrink  σ=1mm  — medium anatomy visible',
                  'Full res   σ=0mm  — pixel-level precision'][lv], W/2, 58);

    // MI progress bar
    const barW = 280, barX = (W-barW)/2, barY = H-52;
    ctx.fillStyle='#1e2d42'; ctx.fillRect(barX,barY,barW,10);
    const filled = barW * Math.min(miVals[lv]*progress + (lv>0?miVals[lv-1]:0), miVals[lv]);
    const grad = ctx.createLinearGradient(barX,0,barX+barW,0);
    grad.addColorStop(0,'#D85A30'); grad.addColorStop(.5,'#EF9F27'); grad.addColorStop(1,'#1D9E75');
    ctx.fillStyle=grad;
    ctx.fillRect(barX, barY, Math.max(0,filled), 10);
    ctx.fillStyle='#5a7080'; ctx.font='8px sans-serif'; ctx.textAlign='left';
    ctx.fillText('MI progress', barX, barY-4);

    // MI value
    const curMI = (lv>0?miVals[lv-1]:0) + progress*miVals[lv];
    document.getElementById('miVal').textContent=curMI.toFixed(3);
    document.getElementById('nmiVal').textContent=(1+curMI*.24).toFixed(3);
    document.getElementById('iterVal').textContent='L'+(lv+1)+':'+Math.round(progress*66);
    miHistory.push(curMI);
    drawMIcurve(curMI);
  }

  let progress=0;
  function frame(){
    drawLevel(pyrStep, progress);
    progress += .016;
    if(progress>=1){
      progress=0;
      pyrStep++;
      if(pyrStep>2){ pyrStep=2; return; }
    }
    pyrAnim=requestAnimationFrame(frame);
  }
  miHistory.length=0;
  frame();
}

// ── Scene 2: Optimiser ─────────────────────────────────────────────────────
let optAnim=null, optT=0;
function scene2(){
  stopAll(); optT=0;
  document.getElementById('sceneDesc').innerHTML=
    '<b style="color:#a084e8">Step 3 — Gradient descent optimiser.</b> '
    +'At each iteration: sample 15% of voxels randomly → compute Mattes MI → '
    +'estimate gradient across all 6 DOF → take a line-searched step. '
    +'Stops when MI improvement over last 10 iterations &lt; 1e-6 or after 200 iterations. '
    +'Parameter scaling ensures rotations (radians) and translations (mm) get balanced step sizes.';
  drawPyramid(2);
  miHistory.length=0;

  // MI landscape path (hand-crafted for visual clarity)
  const path = [];
  for(let i=0;i<=120;i++){
    const t=i/120;
    const mi = .72*(1-Math.exp(-6*t)) * (1-.08*Math.cos(18*t*Math.exp(-4*t)));
    path.push(mi);
    // 6 params converge at different rates
  }

  function frame(){
    if(optT>=path.length){optT=path.length-1;}

    clear();

    // Draw MI landscape
    const pad={l:52,r:20,t:24,b:36};
    const iw=W-pad.l-pad.r, ih=H-pad.t-pad.b;
    function px(t){ return pad.l+t*iw; }
    function py(v){ return pad.t+(1-v/0.75)*ih; }

    // fill under curve
    ctx.beginPath();
    for(let i=0;i<=120;i++){
      const mi=path[Math.min(i,path.length-1)];
      if(i===0)ctx.moveTo(px(i/120),py(mi)); else ctx.lineTo(px(i/120),py(mi));
    }
    ctx.lineTo(px(1),pad.t+ih); ctx.lineTo(px(0),pad.t+ih); ctx.closePath();
    ctx.fillStyle='#00c8ff08'; ctx.fill();

    // curve
    ctx.beginPath();
    path.forEach((v,i)=>{ if(i===0)ctx.moveTo(px(i/120),py(v)); else ctx.lineTo(px(i/120),py(v)); });
    ctx.strokeStyle='#00c8ff33'; ctx.lineWidth=2; ctx.stroke();

    // axes
    ctx.strokeStyle='#1e2d42'; ctx.lineWidth=.5;
    ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ih); ctx.lineTo(pad.l+iw,pad.t+ih); ctx.stroke();
    ctx.fillStyle='#5a7080'; ctx.font='8.5px sans-serif'; ctx.textAlign='center';
    ctx.fillText('Iteration →', pad.l+iw/2, H-4);
    ctx.save(); ctx.translate(14,pad.t+ih/2); ctx.rotate(-Math.PI/2);
    ctx.fillText('Mattes MI  (maximise)', 0, 0); ctx.restore();

    // trail
    const trail = path.slice(0, optT+1);
    if(trail.length>1){
      ctx.beginPath();
      trail.forEach((v,i)=>{ if(i===0)ctx.moveTo(px(i/120),py(v)); else ctx.lineTo(px(i/120),py(v)); });
      ctx.strokeStyle='#ffb547'; ctx.lineWidth=2; ctx.setLineDash([3,2]); ctx.stroke(); ctx.setLineDash([]);
    }

    // ball
    const bx=px(optT/120), by=py(path[optT]);
    ctx.beginPath(); ctx.arc(bx,by,7,0,Math.PI*2);
    ctx.fillStyle='#00c8ff'; ctx.fill();
    ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke();

    // gradient arrow
    if(optT<path.length-2){
      const nx=px((optT+3)/120), ny=py(path[Math.min(optT+3,path.length-1)]);
      ctx.strokeStyle='#ffb547'; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.moveTo(bx,by); ctx.lineTo(nx,ny); ctx.stroke();
    }

    // convergence indicator
    if(optT>100){
      ctx.fillStyle='#00e5a0'; ctx.font='bold 9px monospace'; ctx.textAlign='left';
      ctx.fillText('CONVERGED  ΔMI < 1e-6', pad.l+4, pad.t+14);
    }

    // update panels
    const t2=optT/120;
    const curMI=path[optT];
    const rx=(-14+14*t2).toFixed(2), ry=(8-8*t2).toFixed(2), rz=(-6+6*t2).toFixed(2);
    const tx=(32-32*t2).toFixed(1), ty=(-18+18*t2).toFixed(1), tz=(12-12*t2).toFixed(1);
    document.getElementById('v_rx').textContent=rx+'°';
    document.getElementById('v_ry').textContent=ry+'°';
    document.getElementById('v_rz').textContent=rz+'°';
    document.getElementById('v_tx').textContent=tx+'mm';
    document.getElementById('v_ty').textContent=ty+'mm';
    document.getElementById('v_tz').textContent=tz+'mm';
    document.getElementById('miVal').textContent=curMI.toFixed(4);
    document.getElementById('nmiVal').textContent=(1+curMI*.24).toFixed(4);
    document.getElementById('iterVal').textContent=Math.round(t2*200);
    miHistory.push(curMI);
    drawMIcurve(curMI);

    if(optT<path.length-1){
      optT++;
      optAnim=requestAnimationFrame(frame);
    }
  }
  frame();
}

// ── Scene 3: Result ─────────────────────────────────────────────────────────
function scene3(){
  stopAll();
  document.getElementById('sceneDesc').innerHTML=
    '<b style="color:#00e5a0">Step 4 — Resample & result.</b> '
    +'The optimal Euler3D transform is applied to the PET volume. '
    +'Trilinear interpolation computes PET intensity at every MRI voxel grid point. '
    +'Every voxel (z,y,x) in both images now refers to the same physical location in the brain. '
    +'Our MI improved from ~0.09 to ~0.75 — a 590% gain confirming successful alignment.';
  drawPyramid(2);
  document.getElementById('miVal').textContent='0.755';
  document.getElementById('nmiVal').textContent='1.171';
  document.getElementById('iterVal').textContent='done';
  ['rx','ry','rz'].forEach(k=>document.getElementById('v_'+k).textContent='0.00°');
  ['tx','ty','tz'].forEach(k=>document.getElementById('v_'+k).textContent='0.0mm');

  let alpha=0;
  function frame(){
    clear();
    // MRI
    brain(200,150,82,58,'#1D9E75',0,0);
    // PET fading in on top
    ctx.globalAlpha=Math.min(alpha,.7);
    brain(200,150,74,52,'#D85A30',0,0);
    ctx.globalAlpha=1;
    // Crosshair
    crosshair(200,150,'#00e5a0',26,Math.min(alpha,1));
    // Labels
    if(alpha>.5){
      ctx.fillStyle='#00e5a0'; ctx.font='bold 11px monospace'; ctx.textAlign='center';
      ctx.fillText('ALIGNED  ✓', 200, 42);
      ctx.fillStyle='#5a7080'; ctx.font='9px sans-serif';
      ctx.fillText('PET and MRI now share the same voxel coordinate system', 200, 60);
      // NMI improvement badge
      ctx.fillStyle='#00e5a022'; ctx.strokeStyle='#00e5a066'; ctx.lineWidth=1;
      ctx.beginPath(); ctx.roundRect(W/2-90,H-68,180,36,5); ctx.fill(); ctx.stroke();
      ctx.fillStyle='#00e5a0'; ctx.font='bold 9.5px monospace'; ctx.textAlign='center';
      ctx.fillText('NMI:  1.000  →  1.171  (+590%)', W/2, H-54);
      ctx.fillStyle='#5a7080'; ctx.font='8px sans-serif';
      ctx.fillText('before registration  →  after registration', W/2, H-38);
    }
    if(alpha<1){ alpha+=.015; optAnim=requestAnimationFrame(frame); }
  }
  frame();
}

// ── Auto-play ────────────────────────────────────────────────────────────────
let autoTimer=null;
function autoPlay(){
  stopAll();
  showScene(0);
  autoTimer=setTimeout(()=>{ showScene(1);
    autoTimer=setTimeout(()=>{ showScene(2);
      autoTimer=setTimeout(()=>showScene(3), 7000);
    },5000);
  },3500);
}

function stopAll(){
  if(initAnim){cancelAnimationFrame(initAnim);initAnim=null;}
  if(pyrAnim){cancelAnimationFrame(pyrAnim);pyrAnim=null;}
  if(optAnim){cancelAnimationFrame(optAnim);optAnim=null;}
  if(autoTimer){clearTimeout(autoTimer);autoTimer=null;}
}

const scenes=[scene0,scene1,scene2,scene3];
function showScene(i){stopAll();scenes[i]();}

// start on load
scene0();
</script>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
SEG_HTML = BASE_CSS + """
<h2>MONAI AI SEGMENTATION — HOW OUR ALGORITHM WORKS</h2>

<div class="row">
  <div style="flex:1.8;min-width:280px">
    <div class="card" style="padding:8px">
      <canvas id="sCv" width="400" height="290" style="border-radius:4px;background:#080c10;width:100%"></canvas>
    </div>
    <div class="btn-row">
      <button onclick="sScene(0)">① Normalise</button>
      <button onclick="sScene(1)">② Geodesic</button>
      <button onclick="sScene(2)">③ Otsu</button>
      <button onclick="sScene(3)">④ Rand Walker</button>
      <button onclick="sScene(4)">⑤ Final mask</button>
      <button class="sec" onclick="sAuto()">▶ Auto-play</button>
    </div>
    <div id="sDesc" class="desc" style="margin-top:8px;min-height:44px"></div>
  </div>

  <div style="flex:1;min-width:180px;display:flex;flex-direction:column;gap:6px">
    <div class="card">
      <div class="label">PIPELINE PROGRESS</div>
      <div id="sPipeline" style="margin-top:6px"></div>
    </div>
    <div class="card">
      <div class="label">CURRENT STEP OUTPUT</div>
      <canvas id="sSmall" width="220" height="110" style="background:#080c10;border-radius:4px;width:100%;margin-top:4px"></canvas>
      <div id="sMetrics" style="margin-top:5px;font-size:.72rem;color:#5a7080;line-height:1.7"></div>
    </div>
    <div class="card">
      <div class="label">SEGMENTATION METRICS</div>
      <div id="sStats" style="font-size:.73rem;color:#5a7080;line-height:1.9;margin-top:2px">
        <div>Voxels: <span style="color:#c8d8e8;font-family:monospace" id="mvox">—</span></div>
        <div>Volume: <span style="color:#c8d8e8;font-family:monospace" id="mvol">—</span></div>
        <div>Centroid shift: <span style="color:#c8d8e8;font-family:monospace" id="mshift">—</span></div>
        <div>Method: <span style="color:#00c8ff;font-family:monospace" id="mmethod">—</span></div>
      </div>
    </div>
  </div>
</div>

<script>
const sCv = document.getElementById('sCv');
const sCtx = sCv.getContext('2d');
const SW=sCv.width, SH=sCv.height;
const sSmall = document.getElementById('sSmall');
const ssCtx = sSmall.getContext('2d');
let sAnimId=null, sAutoTimer=null;

const STEPS=[
  {col:'#378ADD',label:'Normalise'},
  {col:'#a084e8',label:'Geodesic map'},
  {col:'#EF9F27',label:'Otsu threshold'},
  {col:'#1D9E75',label:'Random Walker'},
  {col:'#00e5a0',label:'Final mask'},
];

function drawPipeline(active){
  const el=document.getElementById('sPipeline');
  el.innerHTML=STEPS.map((s,i)=>`
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
      <div style="width:8px;height:8px;border-radius:50%;background:${i<=active?s.col:'#1e2d42'};flex-shrink:0;transition:all .3s"></div>
      <div style="font-size:.72rem;font-family:monospace;color:${i===active?s.col:i<active?'#5a7080':'#2a3a4a'};transition:all .3s">${s.label}</div>
      ${i<active?'<span style="font-size:.65rem;color:#00e5a044">✓</span>':''}
    </div>`).join('');
}

function sClr(){ sCtx.clearRect(0,0,SW,SH); sCtx.fillStyle='#080c10'; sCtx.fillRect(0,0,SW,SH); }
function ssClr(){ ssCtx.clearRect(0,0,220,110); ssCtx.fillStyle='#080c10'; ssCtx.fillRect(0,0,220,110); }

// Draw a brain slice (simplified, returns pixel grid for further use)
function drawBrainSlice(ctx,ox,oy,w,h,opts){
  const {tumorCol,wmCol,gmCol,csfCol,seedX,seedY,showSeed,geodRings,otsuLine,rwBoundary,mask}=opts;
  ctx.fillStyle='#1a2535'; ctx.beginPath(); ctx.ellipse(ox+w/2,oy+h/2,w*.44,h*.46,0,0,Math.PI*2); ctx.fill();
  // WM
  ctx.fillStyle=wmCol||'#1a3048';
  [[ox+w*.38,oy+h*.38,w*.22,h*.22],[ox+w*.60,oy+h*.42,w*.18,h*.18],[ox+w*.5,oy+h*.28,w*.28,h*.14]].forEach(([cx,cy,rx,ry])=>{ctx.beginPath();ctx.ellipse(cx,cy,rx,ry,0,0,Math.PI*2);ctx.fill();});
  // GM
  ctx.fillStyle=gmCol||'#243048';
  [[ox+w*.33,oy+h*.6,w*.12,h*.1],[ox+w*.65,oy+h*.58,w*.1,h*.1]].forEach(([cx,cy,rx,ry])=>{ctx.beginPath();ctx.ellipse(cx,cy,rx,ry,0,0,Math.PI*2);ctx.fill();});
  // Tumor
  const tx=ox+w*.68,ty=oy+h*.4,tr=w*.1,ttry=h*.09;
  ctx.fillStyle=tumorCol||'#4a1a0a';
  ctx.beginPath(); ctx.ellipse(tx,ty,tr,ttry,0,0,Math.PI*2); ctx.fill();
  if(otsuLine||rwBoundary){
    ctx.beginPath(); ctx.ellipse(tx,ty,tr+2,ttry+2,.05,0,Math.PI*2);
    ctx.strokeStyle=rwBoundary?'#00e5a0':'#EF9F27'; ctx.lineWidth=2; ctx.stroke();
  }
  if(mask){
    ctx.fillStyle='rgba(0,229,160,.2)';
    ctx.beginPath(); ctx.ellipse(tx,ty,tr,ttry,0,0,Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.ellipse(tx,ty,tr+2,ttry+2,.05,0,Math.PI*2);
    ctx.strokeStyle='#00e5a0'; ctx.lineWidth=2.5; ctx.stroke();
  }
  if(geodRings){
    [.4,.7,1,1.3].forEach((r,j)=>{
      ctx.beginPath(); ctx.ellipse(tx,ty,tr*r+2,ttry*r+2,0,0,Math.PI*2);
      ctx.strokeStyle=`rgba(160,132,232,${.7-j*.15})`; ctx.lineWidth=1; ctx.stroke();
    });
  }
  if(showSeed){
    ctx.beginPath(); ctx.arc(tx,ty,4,0,Math.PI*2);
    ctx.fillStyle='#00c8ff'; ctx.fill();
    ctx.fillStyle='#00c8ff'; ctx.font='8px monospace'; ctx.textAlign='center';
    ctx.fillText('seed',tx,ty-10);
  }
  return {tx,ty,tr,ttry};
}

// ── Scene 0: Normalise ────────────────────────────────────────────────────
let normAnim=null, normT=0;
function sScene0(){
  document.getElementById('sDesc').innerHTML=
    '<b style="color:#378ADD">Step 1 — MONAI ScaleIntensityRange + GaussianSmooth.</b> '
    +'Clips MRI intensities to [p<sub>1</sub>, p<sub>99</sub>] percentile window then maps to [0,1]. '
    +'Bright MRI artifacts are suppressed without losing tissue contrast. '
    +'GaussianSmooth (σ=0.8mm) then reduces noise before the geodesic computation begins.';
  drawPipeline(0);
  document.getElementById('mmethod').textContent='MONAI transforms';
  document.getElementById('mvox').textContent='—';
  document.getElementById('mvol').textContent='—';
  document.getElementById('mshift').textContent='—';

  // animated histogram equalisation
  const raw=[.04,.09,.18,.32,.56,.78,.86,.75,.50,.26,.14,.06,.03,.01];
  let anim=0;
  function frame(){
    sClr();
    const W2=SW/2-20;
    // raw histogram
    const bw=10,step=14,hs=SH*.62;
    raw.forEach((v,j)=>{
      const h2=v*hs;
      sCtx.fillStyle='#D85A3088'; sCtx.fillRect(18+j*step,SH-48-h2,bw,h2);
      sCtx.strokeStyle='#D85A30'; sCtx.lineWidth=.6; sCtx.strokeRect(18+j*step,SH-48-h2,bw,h2);
    });
    sCtx.strokeStyle='#D85A30'; sCtx.lineWidth=.8;
    sCtx.strokeRect(14,SH-48-hs-6,raw.length*step+4,hs+8);
    sCtx.fillStyle='#D85A30'; sCtx.font='9.5px monospace'; sCtx.textAlign='center';
    sCtx.fillText('raw MRI  (0 – 4000+)', 14+raw.length*step/2, SH-32);

    // arrow
    const arrowX=SW/2, arrowT=Math.min(anim*2,1);
    sCtx.globalAlpha=arrowT;
    sCtx.strokeStyle='#00c8ff'; sCtx.lineWidth=2;
    sCtx.beginPath(); sCtx.moveTo(arrowX-30,SH/2-10); sCtx.lineTo(arrowX+30,SH/2-10); sCtx.stroke();
    sCtx.beginPath(); sCtx.moveTo(arrowX+24,SH/2-18); sCtx.lineTo(arrowX+32,SH/2-10); sCtx.lineTo(arrowX+24,SH/2-2); sCtx.fillStyle='#00c8ff'; sCtx.fill();
    sCtx.fillStyle='#00c8ff'; sCtx.font='8px monospace'; sCtx.textAlign='center';
    sCtx.fillText('ScaleIntensityRange',arrowX,SH/2+6);
    sCtx.fillText('[p1,p99] → [0,1]',arrowX,SH/2+18);
    sCtx.globalAlpha=1;

    // normalised histogram
    const normProgress=Math.max(0,anim*2-1);
    raw.forEach((v,j)=>{
      const h2=v*hs*normProgress;
      const normV=v; // same shape, different axis
      sCtx.fillStyle='#378ADD88'; sCtx.fillRect(SW-18-raw.length*step+j*step,SH-48-h2,bw,h2);
      sCtx.strokeStyle='#378ADD'; sCtx.lineWidth=.6; sCtx.strokeRect(SW-18-raw.length*step+j*step,SH-48-h2,bw,h2);
    });
    if(normProgress>0){
      sCtx.strokeStyle='#378ADD'; sCtx.lineWidth=.8;
      sCtx.strokeRect(SW-22-raw.length*step, SH-48-hs-6, raw.length*step+4, hs+8);
      sCtx.fillStyle='#378ADD'; sCtx.font='9.5px monospace'; sCtx.textAlign='center';
      sCtx.fillText('normalised  [0 – 1]', SW-22-raw.length*step/2, SH-32);
      sCtx.fillStyle='#5a7080'; sCtx.font='7.5px sans-serif';
      sCtx.fillText('p1', SW-22-raw.length*step+4, SH-18);
      sCtx.fillText('p99', SW-22, SH-18);
    }

    anim=Math.min(anim+.015,1);
    if(anim<1) normAnim=requestAnimationFrame(frame);
    else {
      // draw small preview
      ssClr();
      const g=ssCtx.createLinearGradient(10,0,210,0);
      g.addColorStop(0,'#0d1f33'); g.addColorStop(.4,'#1a3048'); g.addColorStop(.8,'#4a1a0a'); g.addColorStop(1,'#2a3a4a');
      ssCtx.fillStyle=g; ssCtx.fillRect(10,20,200,70);
      ssCtx.fillStyle='#378ADD44'; ssCtx.fillRect(10,20,200,70);
      ssCtx.strokeStyle='#378ADD'; ssCtx.lineWidth=.8; ssCtx.strokeRect(10,20,200,70);
      ssCtx.fillStyle='#378ADD'; ssCtx.font='8px monospace'; ssCtx.textAlign='center';
      ssCtx.fillText('MRI — normalised to [0, 1]', 110, 106);
      document.getElementById('sMetrics').innerHTML='p<sub>1</sub>=clip low | p<sub>99</sub>=clip high | GaussianSmooth σ=0.8mm';
    }
  }
  normAnim=requestAnimationFrame(frame);
}

// ── Scene 1: Geodesic ─────────────────────────────────────────────────────
let geoAnim=null, geoT=0;
function sScene1(){
  document.getElementById('sDesc').innerHTML=
    '<b style="color:#a084e8">Step 2 — Geodesic distance map.</b> '
    +'From the user centroid seed we compute geodesic distance for every voxel. '
    +'Formula: <code style="color:#ffb547">geodesic = 0.5 × euclidean_norm + 0.5 × |intensity − seed_intensity|</code>. '
    +'Unlike Euclidean, it is blocked by intensity barriers — rings stay inside the tumour and stop at tissue edges.';
  drawPipeline(1);
  document.getElementById('mmethod').textContent='Geodesic EDT';
  geoT=0;
  function frame(){
    sClr();
    const {tx,ty,tr,ttry}=drawBrainSlice(sCtx,30,40,340,220,{
      tumorCol:'#4a1a0a',showSeed:geoT>.2,geodRings:geoT>.4
    });
    // animate expanding rings
    const maxR=geoT*120;
    if(geoT>.1){
      [.6,.8,1,1.2,1.5,1.9,2.4].forEach((r,j)=>{
        if(r*40>maxR) return;
        const alpha=Math.max(0,.7-j*.1);
        sCtx.beginPath(); sCtx.ellipse(tx,ty,r*tr,r*ttry,.08,0,Math.PI*2);
        sCtx.strokeStyle=`rgba(160,132,232,${alpha})`; sCtx.lineWidth=1.2; sCtx.stroke();
      });
      // euclidean rings for comparison (dashed, cross boundary)
      [55,85,115].forEach((r,j)=>{
        if(r>maxR) return;
        sCtx.beginPath(); sCtx.arc(tx,ty,r,0,Math.PI*2);
        sCtx.strokeStyle=`rgba(255,107,53,${.28-j*.07})`; sCtx.lineWidth=1; sCtx.setLineDash([4,3]); sCtx.stroke(); sCtx.setLineDash([]);
      });
    }
    // legend
    sCtx.fillStyle='#a084e8'; sCtx.font='8.5px sans-serif'; sCtx.textAlign='left';
    sCtx.fillText('— geodesic: stays in tumour', 36, SH-28);
    sCtx.fillStyle='#ff6b35';
    sCtx.fillText('- - euclidean: crosses boundaries', 36, SH-14);
    // title
    sCtx.fillStyle='#a084e8'; sCtx.font='bold 10px monospace'; sCtx.textAlign='center';
    sCtx.fillText('Geodesic distance map expanding from seed', SW/2, 28);
    geoT=Math.min(geoT+.012,1);
    if(geoT<1) geoAnim=requestAnimationFrame(frame);
    else{
      ssClr();
      // heatmap of geodesic
      const ig=ssCtx.createRadialGradient(110,55,5,110,55,90);
      ig.addColorStop(0,'#a084e8'); ig.addColorStop(.4,'#a084e844'); ig.addColorStop(1,'#0e1520');
      ssCtx.fillStyle=ig; ssCtx.fillRect(10,10,200,88);
      ssCtx.strokeStyle='#a084e8'; ssCtx.lineWidth=.8; ssCtx.strokeRect(10,10,200,88);
      ssCtx.fillStyle='#a084e8'; ssCtx.font='8px monospace'; ssCtx.textAlign='center';
      ssCtx.fillText('geodesic distance map (low=tumour)', 110, 105);
      document.getElementById('sMetrics').innerHTML='low geodesic = near seed + similar intensity';
    }
  }
  frame();
}

// ── Scene 2: Otsu ─────────────────────────────────────────────────────────
let otsuAnim=null, otsuT=0;
function sScene2(){
  document.getElementById('sDesc').innerHTML=
    '<b style="color:#EF9F27">Step 3 — Otsu automatic threshold.</b> '
    +'Applied to the geodesic map. Finds the split that maximises variance between two classes. '
    +'No manual threshold needed. Voxels below the threshold (low geodesic = close to seed) '
    +'are labelled tumour. This gives us the initial binary mask.';
  drawPipeline(2);
  document.getElementById('mmethod').textContent='Otsu threshold';
  otsuT=0;
  const bars=[.03,.06,.11,.19,.27,.23,.15,.09,.06,.04,.02,.01];
  const thr=5;
  function frame(){
    sClr();
    const ox=30,oy=20,bw2=SW-60,bh=SH-70;
    sCtx.strokeStyle='#1e2d42'; sCtx.lineWidth=.5; sCtx.strokeRect(ox,oy,bw2,bh);
    bars.forEach((v,j)=>{
      const bWidth=(bw2-4)/bars.length;
      const bHeight=v*bh*.88;
      const filled=Math.min(otsuT*bars.length*1.5-j*1.5,1);
      if(filled<=0) return;
      sCtx.fillStyle=j<thr?'#a084e833':'#1e2d4444';
      sCtx.fillRect(ox+2+j*bWidth,oy+bh-bHeight*filled,bWidth-1,bHeight*filled);
      sCtx.strokeStyle=j<thr?'#a084e8':'#378ADD';
      sCtx.lineWidth=.7; sCtx.strokeRect(ox+2+j*bWidth,oy+bh-bHeight*filled,bWidth-1,bHeight*filled);
    });
    // Otsu line
    const lineProgress=Math.max(0,otsuT*3-1);
    const tx2=ox+2+thr*((bw2-4)/bars.length);
    if(lineProgress>0){
      sCtx.globalAlpha=lineProgress;
      sCtx.strokeStyle='#EF9F27'; sCtx.lineWidth=2.5; sCtx.setLineDash([5,3]);
      sCtx.beginPath(); sCtx.moveTo(tx2,oy-10); sCtx.lineTo(tx2,oy+bh+10); sCtx.stroke();
      sCtx.setLineDash([]);
      sCtx.fillStyle='#EF9F27'; sCtx.font='bold 10px monospace'; sCtx.textAlign='left';
      sCtx.fillText('Otsu threshold  (automatic)', tx2+4, oy+16);
      sCtx.globalAlpha=1;
    }
    sCtx.fillStyle='#a084e8'; sCtx.font='9px sans-serif'; sCtx.textAlign='center';
    sCtx.fillText('tumour  (low geodesic)', ox+thr*((bw2-4)/bars.length)/2, oy+bh+20);
    sCtx.fillStyle='#5a7080';
    sCtx.fillText('background  (high geodesic)', ox+thr*((bw2-4)/bars.length)+(bars.length-thr)*((bw2-4)/bars.length)/2, oy+bh+20);
    sCtx.fillStyle='#5a7080'; sCtx.font='9px sans-serif';
    sCtx.fillText('geodesic distance  →', ox+bw2/2, oy+bh+36);
    sCtx.fillStyle='#EF9F27'; sCtx.font='bold 10px monospace'; sCtx.textAlign='center';
    sCtx.fillText('Otsu — Automatic Threshold on Geodesic Map', SW/2, 14);
    otsuT=Math.min(otsuT+.012,1);
    if(otsuT<1) otsuAnim=requestAnimationFrame(frame);
    else{
      ssClr();
      drawBrainSlice(ssCtx,10,10,200,88,{tumorCol:'#EF9F2766',otsuLine:true});
      ssCtx.fillStyle='#EF9F27'; ssCtx.font='8px monospace'; ssCtx.textAlign='center';
      ssCtx.fillText('initial binary mask', 110, 105);
      document.getElementById('sMetrics').innerHTML='threshold maximises inter-class variance';
    }
  }
  frame();
}

// ── Scene 3: Random Walker ────────────────────────────────────────────────
let rwAnim=null, rwT=0;
function sScene3(){
  document.getElementById('sDesc').innerHTML=
    '<b style="color:#1D9E75">Step 4 — Random Walker boundary refinement.</b> '
    +'Each voxel is a node, intensity differences are resistances, seeds are voltage sources. '
    +'Random walks from each seed find the minimum-energy boundary — '
    +'they naturally flow through similar-intensity tissue and stop at sharp intensity edges. '
    +'This gives us pixel-precise tumour boundaries.';
  drawPipeline(3);
  document.getElementById('mmethod').textContent='Random Walker';
  rwT=0;
  // Random walk path cache
  const walks=[];
  for(let i=0;i<20;i++){
    const startX=200+((Math.random()-.5)*60);
    const startY=130+((Math.random()-.5)*40);
    const pts=[[startX,startY]];
    let x=startX,y=startY;
    for(let j=0;j<30;j++){ x+=(Math.random()-.5)*16; y+=(Math.random()-.5)*13; pts.push([x,y]); }
    walks.push({pts,col:i<12?'#00c8ff':'#D85A30'});
  }
  function frame(){
    sClr();
    const {tx,ty,tr,ttry}=drawBrainSlice(sCtx,30,40,340,220,{tumorCol:'#4a1a0a'});
    // animate walks
    const steps=Math.floor(rwT*32);
    walks.forEach(w=>{
      sCtx.beginPath();
      for(let k=0;k<=Math.min(steps,w.pts.length-1);k++){
        if(k===0)sCtx.moveTo(w.pts[k][0]+30,w.pts[k][1]+40); else sCtx.lineTo(w.pts[k][0]+30,w.pts[k][1]+40);
      }
      sCtx.strokeStyle=w.col+'22'; sCtx.lineWidth=1; sCtx.stroke();
    });
    // boundary emerges
    if(rwT>.6){
      const alpha=(rwT-.6)/.4;
      sCtx.globalAlpha=alpha;
      sCtx.beginPath(); sCtx.ellipse(tx,ty,tr+2,ttry+2,.05,0,Math.PI*2);
      sCtx.strokeStyle='#00e5a0'; sCtx.lineWidth=2.5; sCtx.stroke();
      sCtx.globalAlpha=1;
    }
    // seed dots
    [[tx,ty,'#D85A30'],[tx-20,ty+15,'#D85A30'],[tx+20,ty-5,'#D85A30'],
     [tx-60,ty-40,'#1e2d42'],[tx+90,ty+50,'#1e2d42'],[tx-50,ty+70,'#1e2d42']].forEach(([x,y,c])=>{
      sCtx.beginPath(); sCtx.arc(x,y,3.5,0,Math.PI*2); sCtx.fillStyle=c; sCtx.fill();
    });
    sCtx.fillStyle='#D85A30'; sCtx.font='8px sans-serif'; sCtx.textAlign='left';
    sCtx.fillText('● tumour seeds', 36, SH-38);
    sCtx.fillStyle='#1e2d42';
    sCtx.fillText('■ background seeds', 36, SH-24);
    sCtx.fillStyle='#00e5a0';
    sCtx.fillText('— refined boundary', 36, SH-10);
    sCtx.fillStyle='#1D9E75'; sCtx.font='bold 10px monospace'; sCtx.textAlign='center';
    sCtx.fillText('Random Walker — graph-based boundary refinement', SW/2, 26);
    rwT=Math.min(rwT+.01,1);
    if(rwT<1) rwAnim=requestAnimationFrame(frame);
    else{
      ssClr();
      drawBrainSlice(ssCtx,10,10,200,88,{tumorCol:'#1D9E7544',rwBoundary:true});
      ssCtx.fillStyle='#1D9E75'; ssCtx.font='8px monospace'; ssCtx.textAlign='center';
      ssCtx.fillText('refined tumour boundary', 110, 105);
      document.getElementById('sMetrics').innerHTML='high gradient = boundary stop | β=10 (graph weight)';
    }
  }
  frame();
}

// ── Scene 4: Final mask ───────────────────────────────────────────────────
let maskAnim=null, maskT=0;
function sScene4(){
  document.getElementById('sDesc').innerHTML=
    '<b style="color:#00e5a0">Step 5 — 3D morphological cleaning + final mask.</b> '
    +'Binary closing (3 iterations) fills small gaps. '
    +'Hole filling makes the interior completely solid. '
    +'Connected component labelling keeps only the blob containing the user centroid, '
    +'discarding all spurious noise regions. Output: 3D boolean mask (Z,Y,X).';
  drawPipeline(4);
  document.getElementById('mmethod').textContent='Morphological + CC';
  maskT=0;
  function frame(){
    sClr();
    // show 4 stages side by side
    const stages=[
      {label:'noisy mask',col:'#D85A30',gap:.08},
      {label:'after closing',col:'#EF9F27',gap:.04},
      {label:'holes filled',col:'#1D9E75',gap:.02},
      {label:'final clean',col:'#00e5a0',gap:0},
    ];
    const stW=SW/4-10;
    const progress=Math.min(maskT*4,1);
    stages.forEach((s,i)=>{
      if(i/4>maskT*.8) return;
      const ox=i*(stW+10)+8,oy=30,sh=SH-70;
      const alpha=Math.min((maskT*4-i*0.8),1);
      sCtx.globalAlpha=Math.max(0,alpha);
      drawBrainSlice(sCtx,ox,oy,stW,sh-10,{
        tumorCol:s.col+'66',rwBoundary:i>=2,mask:i===3
      });
      sCtx.fillStyle=s.col; sCtx.font='7.5px monospace'; sCtx.textAlign='center';
      sCtx.fillText(s.label,ox+stW/2,oy+sh+8);
      if(i<3){
        sCtx.fillStyle='#00c8ff'; sCtx.font='12px monospace';
        sCtx.fillText('→',ox+stW+5,oy+sh/2);
      }
      sCtx.globalAlpha=1;
    });
    if(maskT>.85){
      const alpha=(maskT-.85)/.15;
      sCtx.globalAlpha=alpha;
      sCtx.fillStyle='#00e5a0'; sCtx.font='bold 9.5px monospace'; sCtx.textAlign='center';
      sCtx.fillText('3D mask ready  ✓  (Z,Y,X)  bool array', SW/2, SH-12);
      sCtx.globalAlpha=1;
      // update metrics
      document.getElementById('mvox').textContent='~18,400';
      document.getElementById('mvol').textContent='~18.4 cm³';
      document.getElementById('mshift').textContent='2.3 vox';
      document.getElementById('mmethod').textContent='MONAI + RW ✓';
    }
    sCtx.fillStyle='#00e5a0'; sCtx.font='bold 10px monospace'; sCtx.textAlign='center';
    sCtx.fillText('Morphological cleaning pipeline', SW/2, 20);
    maskT=Math.min(maskT+.01,1);
    if(maskT<1) maskAnim=requestAnimationFrame(frame);
    else{
      ssClr();
      drawBrainSlice(ssCtx,10,10,200,88,{tumorCol:'#00e5a022',mask:true});
      ssCtx.fillStyle='#00e5a0'; ssCtx.font='8px monospace'; ssCtx.textAlign='center';
      ssCtx.fillText('final 3D tumour mask', 110, 105);
      document.getElementById('sMetrics').innerHTML='closing=3 iter | hole-fill | largest CC kept';
    }
  }
  frame();
}

function sStopAll(){
  [normAnim,geoAnim,otsuAnim,rwAnim,maskAnim].forEach(id=>{if(id)cancelAnimationFrame(id);});
  normAnim=geoAnim=otsuAnim=rwAnim=maskAnim=null;
  if(sAutoTimer){clearTimeout(sAutoTimer);sAutoTimer=null;}
}

const sScenes=[sScene0,sScene1,sScene2,sScene3,sScene4];
function sScene(i){sStopAll();sScenes[i]();}

function sAuto(){
  sStopAll();sScene(0);
  sAutoTimer=setTimeout(()=>{sScene(1);
    sAutoTimer=setTimeout(()=>{sScene(2);
      sAutoTimer=setTimeout(()=>{sScene(3);
        sAutoTimer=setTimeout(()=>sScene(4),5000);
      },5000);
    },5000);
  },4000);
}

sScene(0);
// Fix unclosed string in mmethod
document.getElementById('mmethod').textContent='MONAI transforms';
</script>
"""


def render():
    utils.try_restore_from_cache()
    utils.sidebar_status()

    st.title("💡 Explanation!")

    tab1, tab2 = st.tabs([
        "🔗 Coregistration pipeline",
        "🎯 Segmentation pipeline",
    ])

    with tab1:
        components.html(COREG_HTML, height=820, scrolling=False)

    with tab2:
        components.html(SEG_HTML, height=820, scrolling=False)