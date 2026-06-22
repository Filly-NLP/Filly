/* ─────────────────────────────────────────────
   FILLY – Unified Application Logic
───────────────────────────────────────────── */

// ══════════════════════════════════════════════
// DATA
// ══════════════════════════════════════════════
const NORM = {
  'aq':{to:'ako',type:'norm',label:'Abbreviation'},'nsa':{to:'nasa',type:'norm',label:'Abbreviation'},
  'bah8':{to:'bahay',type:'norm',label:'Leet speak'},'kse':{to:'kasi',type:'norm',label:'Abbreviation'},
  'kc':{to:'kasi',type:'norm',label:'Abbreviation'},'knina':{to:'kanina',type:'norm',label:'Abbreviation'},
  'tpos':{to:'pagkatapos',type:'norm',label:'Abbreviation'},'w/':{to:'kasama ng',type:'norm',label:'Symbol'},
  'dba':{to:'diba',type:'norm',label:'Abbreviation'},'nlang':{to:'na lang',type:'norm',label:'Contraction'},
  'nlng':{to:'na lang',type:'norm',label:'Contraction'},'ngaun':{to:'ngayon',type:'norm',label:'Abbreviation'},
  'ung':{to:'yung',type:'norm',label:'Abbreviation'},'cno':{to:'sino',type:'norm',label:'Abbreviation'},
  'cnu':{to:'sino',type:'norm',label:'Abbreviation'},'xa':{to:'siya',type:'norm',label:'Abbreviation'},
  'cya':{to:'siya',type:'norm',label:'Abbreviation'},'sya':{to:'siya',type:'norm',label:'Abbreviation'},
  'poh':{to:'po',type:'norm',label:'Colloquial'},'nman':{to:'naman',type:'norm',label:'Abbreviation'},
  'nmn':{to:'naman',type:'norm',label:'Abbreviation'},'dn':{to:'rin',type:'norm',label:'Abbreviation'},
  'pra':{to:'para',type:'norm',label:'Abbreviation'},'pro':{to:'pero',type:'norm',label:'Abbreviation'},
  'pru':{to:'pero',type:'norm',label:'Abbreviation'},'dpat':{to:'dapat',type:'norm',label:'Abbreviation'},
  'lhat':{to:'lahat',type:'norm',label:'Abbreviation'},'hnd':{to:'hindi',type:'norm',label:'Abbreviation'},
  'hndi':{to:'hindi',type:'norm',label:'Abbreviation'},'wla':{to:'wala',type:'norm',label:'Abbreviation'},
  'bkit':{to:'bakit',type:'norm',label:'Abbreviation'},'bkt':{to:'bakit',type:'norm',label:'Abbreviation'},
  'gsto':{to:'gusto',type:'norm',label:'Abbreviation'},'mgkita':{to:'magkita',type:'norm',label:'Abbreviation'},
  'tyo':{to:'tayo',type:'norm',label:'Abbreviation'},'tyu':{to:'tayo',type:'norm',label:'Abbreviation'},
  'yun':{to:'iyon',type:'gram',label:'Grammar'},'ganun':{to:'ganoon',type:'gram',label:'Grammar'},
  'ndi':{to:'hindi',type:'norm',label:'Abbreviation'},'na2':{to:'na rin',type:'norm',label:'Number sub'},
  'din':{to:'rin',type:'gram',label:'Grammar'},'sna':{to:'sana',type:'norm',label:'Abbreviation'},
  'cge':{to:'sige',type:'norm',label:'Abbreviation'},'cgeh':{to:'sige',type:'norm',label:'Abbreviation'},
  'tska':{to:'tsaka',type:'norm',label:'Abbreviation'},'tlga':{to:'talaga',type:'norm',label:'Abbreviation'},
  'talga':{to:'talaga',type:'norm',label:'Abbreviation'},'nyo':{to:'ninyo',type:'gram',label:'Grammar'},
  'ntin':{to:'natin',type:'norm',label:'Abbreviation'},'ksma':{to:'kasama',type:'norm',label:'Abbreviation'},
  'mhal':{to:'mahal',type:'norm',label:'Abbreviation'},'lbas':{to:'labas',type:'norm',label:'Abbreviation'},
};

const METRICS = {
  precision:82.45, recall:76.12, f05:80.91, err:71.33,
  bars:{pN:85,pG:79,rN:72,rG:80,fN:81,fG:79,eN:68,eG:74},
  donut:[
    {label:'Abbreviations',value:42,color:'#FFD23F'},
    {label:'Grammar Errors',value:28,color:'#3B5FE6'},
    {label:'Spelling / Leet',value:18,color:'#FFED99'},
    {label:'Punctuation',value:12,color:'#8BA4FB'},
  ]
};

// ══════════════════════════════════════════════
// LANDING PAGE
// ══════════════════════════════════════════════
function initLanding() {
  initParticles();
  window.addEventListener('scroll', () => {
    const n = document.getElementById('landingNav');
    if (n) n.classList.toggle('scrolled', window.scrollY > 50);
  });
  ['heroGetStarted','navGetStarted','teamGetStarted'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', e => { e.preventDefault(); goToApp(); });
  });
  document.querySelectorAll('.lnav-links a, .btn-hero-secondary').forEach(a => {
    a.addEventListener('click', e => {
      const h = a.getAttribute('href');
      if (h && h.startsWith('#')) { e.preventDefault(); document.querySelector(h)?.scrollIntoView({behavior:'smooth'}); }
    });
  });
  initPreviewCycle();
}

function goToApp() {
  const l = document.getElementById('landingPage'), a = document.getElementById('appWrapper');
  l.classList.add('exit');
  setTimeout(() => { l.style.display='none'; a.style.display='flex'; a.classList.add('enter'); window.scrollTo(0,0); }, 500);
}
function goToLanding() {
  const l = document.getElementById('landingPage'), a = document.getElementById('appWrapper');
  a.style.display='none'; a.classList.remove('enter');
  l.classList.remove('exit'); l.style.display='';
  window.scrollTo(0,0);
}

function initParticles() {
  const c = document.getElementById('particleCanvas');
  if (!c) return;
  const ctx = c.getContext('2d');
  const resize = () => { c.width = window.innerWidth; c.height = window.innerHeight; };
  resize(); window.addEventListener('resize', resize);
  const w = 'aq nsa bah8 kse tpos nlang dba cya hnd wla sya F I L L Y'.split(' ');
  const ps = Array.from({length:35}, () => ({
    x:Math.random()*window.innerWidth, y:Math.random()*window.innerHeight,
    vx:(Math.random()-0.5)*0.35, vy:(Math.random()-0.5)*0.35,
    a:Math.random()*0.3+0.05, t:w[Math.floor(Math.random()*w.length)],
    s:Math.random()*5+10, color:Math.random()>0.5?'#FFD23F':'#5B7EF7',
  }));
  (function animate() {
    ctx.clearRect(0,0,c.width,c.height);
    ps.forEach(p => {
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<-60)p.x=c.width+60; if(p.x>c.width+60)p.x=-60;
      if(p.y<-30)p.y=c.height+30; if(p.y>c.height+30)p.y=-30;
      ctx.save(); ctx.globalAlpha=p.a; ctx.font=`${p.s}px 'JetBrains Mono',monospace`; ctx.fillStyle=p.color; ctx.fillText(p.t,p.x,p.y); ctx.restore();
    });
    requestAnimationFrame(animate);
  })();
}

function initPreviewCycle() {
  const exs = [
    {i:'kumusta ka na? nsa bah8 aq ngayon kse wala akong pasok',o:'Kumusta ka na? Nasa bahay ako ngayon kasi wala akong pasok.'},
    {i:'nag-aral aq knina tpos kumain w/ barkada ko sa cafeteria',o:'Nag-aral ako kanina pagkatapos kumain kasama ng barkada ko sa cafeteria.'},
    {i:'grabe ang init ngaun dba? nlang mainit tuwing tag-araw',o:'Grabe ang init ngayon, diba? Na lang mainit tuwing tag-araw.'},
  ];
  const iE = document.getElementById('previewInput'), oE = document.getElementById('previewOutput');
  if (!iE||!oE) return;
  let idx = 0;
  function type(el,txt,spd=22){return new Promise(r=>{el.textContent='';let i=0;const iv=setInterval(()=>{el.textContent+=txt[i];i++;if(i>=txt.length){clearInterval(iv);r();}},spd);})}
  async function cycle(){const e=exs[idx%exs.length];await type(iE,e.i,24);await new Promise(r=>setTimeout(r,600));await type(oE,e.o,18);await new Promise(r=>setTimeout(r,3000));idx++;cycle();}
  setTimeout(cycle,1200);
}

// ══════════════════════════════════════════════
// DASHBOARD NAV
// ══════════════════════════════════════════════
function initNav() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      const v = item.dataset.view;
      document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
      item.classList.add('active');
      document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));
      const t = document.getElementById('view'+v.charAt(0).toUpperCase()+v.slice(1));
      if(t) t.classList.add('active');
      document.getElementById('sidebar').classList.remove('open');
      document.getElementById('sidebarOverlay').classList.remove('active');
      if(v==='analytics') setTimeout(animateMetrics,200);
    });
  });
  const tog=document.getElementById('menuToggle'),sb=document.getElementById('sidebar'),ov=document.getElementById('sidebarOverlay');
  if(tog)tog.addEventListener('click',()=>{sb.classList.toggle('open');ov.classList.toggle('active');});
  if(ov)ov.addEventListener('click',()=>{sb.classList.remove('open');ov.classList.remove('active');});
  const back=document.getElementById('backToLanding');
  if(back)back.addEventListener('click',goToLanding);
}

// ══════════════════════════════════════════════
// WRITE VIEW
// ══════════════════════════════════════════════
const WORD_LIMIT = 250;

function initWrite() {
  const ta = document.getElementById('editorTextarea');
  ta.addEventListener('input', updateStats);
  document.getElementById('checkBtn').addEventListener('click', processText);
  document.getElementById('copyBtn').addEventListener('click', () => {
    const el = document.getElementById('outputDisplay').querySelector('.output-text');
    if(el){ navigator.clipboard.writeText(el.textContent).then(()=>{
      const b=document.getElementById('copyBtn');
      b.innerHTML='<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Copied!';
      setTimeout(()=>{b.innerHTML='<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy';},2000);
    });}
  });
  document.getElementById('saveBtn').addEventListener('click',downloadCorrectedOutput);
  document.getElementById('uploadBtn').addEventListener('click',()=>{
    const inp=document.createElement('input');inp.type='file';inp.accept='.txt';
    inp.onchange=e=>{const f=e.target.files[0];if(!f)return;const r=new FileReader();r.onload=ev=>{ta.value=limitWords(ev.target.result);updateStats();};r.readAsText(f);};
    inp.click();
  });
  updateStats();
}

function updateStats(){
  const ta=document.getElementById('editorTextarea');
  const limited=limitWords(ta.value);
  if(limited!==ta.value){
    const pos=ta.selectionStart;
    ta.value=limited;
    ta.setSelectionRange(Math.min(pos,limited.length),Math.min(pos,limited.length));
    showToast(`Maximum ${WORD_LIMIT} words allowed.`);
  }
  const t=ta.value;
  const w=t.trim()?t.trim().split(/\s+/).length:0;
  document.getElementById('inputStats').textContent=`${w}/${WORD_LIMIT} word${w!==1?'s':''}`;
  document.getElementById('inputChars').textContent=`${t.length} characters`;
}

function limitWords(text){
  const matches=String(text).match(/\S+\s*/g);
  if(!matches || matches.length<=WORD_LIMIT) return String(text);
  return matches.slice(0,WORD_LIMIT).join('').trimEnd();
}

function downloadCorrectedOutput(){
  const output=document.getElementById('outputDisplay').querySelector('.output-text');
  const text=output?.textContent.trim();
  if(!text){showToast('Check text first before saving.');return;}
  const rawTitle=document.getElementById('docTitle').value.trim() || 'Untitled';
  const filename=rawTitle.replace(/[<>:"/\\|?*\x00-\x1F]/g,'').trim() || 'Untitled';
  const blob=new Blob([text],{type:'text/plain;charset=utf-8'});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');
  a.href=url;
  a.download=`${filename}.txt`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  showToast('Corrected output downloaded.');
}

function processText(){
  const ta=document.getElementById('editorTextarea'), text=ta.value.trim();
  if(!text){ta.focus();return;}
  const out=document.getElementById('outputDisplay');
  out.innerHTML='<div class="processing-state"><div class="spinner"></div><span style="color:var(--text-3);font-size:0.9rem;">Processing…</span></div>';
  document.getElementById('suggestionsBar') && (document.getElementById('suggestionsBar').style.display='none');
  // Clear recs
  document.getElementById('recsBody').innerHTML='<div class="recs-empty"><div class="spinner"></div><p style="margin-top:12px;font-size:0.85rem;color:var(--text-3);">Analyzing…</p></div>';
  document.getElementById('recsCount').textContent='Suggestions (…)';

  setTimeout(()=>{
    const {corrected, suggestions} = analyze(text);
    // Output
    let html = corrected;
    suggestions.forEach(s=>{
      const re=new RegExp(`(${escRx(s.to)})`,'gi');
      html=html.replace(re,'<mark>$1</mark>');
    });
    out.innerHTML=`<div class="output-text">${html}</div>`;
    document.getElementById('outputChars').textContent=`${corrected.length} characters`;

    // Recommendations
    document.getElementById('recsCount').textContent=`Suggestions (${suggestions.length})`;
    if(suggestions.length===0){
      document.getElementById('recsBody').innerHTML=`<div class="recs-empty"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#4ADE80" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polyline points="16 9 10.5 14.5 8 12"/></svg><p>No issues found!</p><span>Your text looks good.</span></div>`;
    } else {
      renderRecs(suggestions);
    }
  },1100);
}

function analyze(text){
  const tokens=text.split(/(\s+)/);
  const sugs=[], seen=new Set(), out=[];
  tokens.forEach(tok=>{
    if(/^\s+$/.test(tok)){out.push(tok);return;}
    const lo=tok.toLowerCase(), stripped=lo.replace(/[.,!?;:]+$/,''), punct=lo.slice(stripped.length);
    let match=null;
    if(NORM[lo]){match={key:lo,word:tok,entry:NORM[lo],punct:''};}
    else if(stripped!==lo && NORM[stripped]){match={key:stripped,word:tok,entry:NORM[stripped],punct};}
    if(match){
      const repl=match.entry.to+match.punct;
      if(!seen.has(match.key)){seen.add(match.key);sugs.push({from:match.word,to:repl,type:match.entry.type,label:match.entry.label});}
      out.push(repl);
    } else { out.push(tok); }
  });
  let corrected=out.join('');
  if(corrected.length>0) corrected=corrected.charAt(0).toUpperCase()+corrected.slice(1);
  if(corrected.length>0 && !/[.!?]$/.test(corrected.trim())) corrected=corrected.trimEnd()+'.';
  return {corrected,suggestions:sugs};
}

function renderRecs(sugs){
  const body=document.getElementById('recsBody');
  body.innerHTML='';
  sugs.forEach((s,i)=>{
    const card=document.createElement('div');
    card.className='sug-card';
    card.style.animationDelay=`${i*0.06}s`;
    card.innerHTML=`
      <div class="sug-header">
        <span class="sug-tag ${s.type==='norm'?'tag-norm':'tag-gram'}">${s.type==='norm'?'Normalization':'Grammar'}</span>
        <span class="sug-label">${escH(s.label)}</span>
      </div>
      <div class="sug-from">${escH(s.from)}</div>
      <div class="sug-to">→ ${escH(s.to)}</div>
      <div class="sug-actions">
        <button class="sug-btn sug-accept" data-f="${escA(s.from)}" data-t="${escA(s.to)}">Accept</button>
        <button class="sug-btn sug-ignore">Ignore</button>
      </div>`;
    body.appendChild(card);
  });
  body.querySelectorAll('.sug-accept').forEach(b=>{
    b.addEventListener('click',()=>{applyFix(b.dataset.f,b.dataset.t);dismiss(b);});
  });
  body.querySelectorAll('.sug-ignore').forEach(b=>{
    b.addEventListener('click',()=>dismiss(b));
  });
}

function applyFix(from,to){
  const ta=document.getElementById('editorTextarea');
  ta.value=ta.value.replace(new RegExp(`\\b${escRx(from)}\\b`,'gi'),to);
  updateStats();
}

function dismiss(btn){
  const c=btn.closest('.sug-card');
  c.style.opacity='0';c.style.transform='translateX(16px)';c.style.transition='all 0.25s ease';
  setTimeout(()=>{
    c.remove();
    const rem=document.querySelectorAll('.sug-card').length;
    document.getElementById('recsCount').textContent=`Suggestions (${rem})`;
    if(rem===0){
      document.getElementById('recsBody').innerHTML=`<div class="recs-empty"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#4ADE80" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polyline points="16 9 10.5 14.5 8 12"/></svg><p>All resolved!</p><span>Your text has been improved.</span></div>`;
    }
  },250);
}

// ══════════════════════════════════════════════
// ANALYTICS
// ══════════════════════════════════════════════
let metricsReady=false;
function animateMetrics(){
  if(metricsReady)return;metricsReady=true;
  const d=METRICS;
  animNum(document.getElementById('metricPrecision'),d.precision);
  animNum(document.getElementById('metricRecall'),d.recall);
  animNum(document.getElementById('metricF05'),d.f05);
  animNum(document.getElementById('metricERR'),d.err);
  setTimeout(()=>{
    document.getElementById('precisionBar').style.width=d.precision+'%';
    document.getElementById('recallBar').style.width=d.recall+'%';
    document.getElementById('f05Bar').style.width=d.f05+'%';
    document.getElementById('errBar').style.width=d.err+'%';
  },100);
  setTimeout(()=>{
    setBar('bar-p-n',d.bars.pN);setBar('bar-p-g',d.bars.pG);
    setBar('bar-r-n',d.bars.rN);setBar('bar-r-g',d.bars.rG);
    setBar('bar-f-n',d.bars.fN);setBar('bar-f-g',d.bars.fG);
    setBar('bar-e-n',d.bars.eN);setBar('bar-e-g',d.bars.eG);
  },300);
  setTimeout(animDonut,400);
}
function animNum(el,target){if(!el)return;const dur=1200,start=performance.now();(function tick(now){const p=Math.min((now-start)/dur,1);el.textContent=(p*(1-Math.pow(1-p,2))*target/(p||1)).toFixed(2);if(p>=1){el.textContent=target.toFixed(2);return;}requestAnimationFrame(tick);})(performance.now());}
function setBar(id,v){const el=document.getElementById(id);if(!el)return;el.style.height=v+'%';const s=el.querySelector('.bar-v');if(s)s.textContent=v+'%';}
function animDonut(){
  const data=METRICS.donut,total=data.reduce((s,d)=>s+d.value,0),circ=2*Math.PI*80;
  let off=0;
  data.forEach((seg,i)=>{const el=document.getElementById(`ds${i+1}`);if(!el)return;const dash=(seg.value/total)*circ;el.style.stroke=seg.color;el.setAttribute('stroke-dasharray',`${dash} ${circ-dash}`);el.setAttribute('stroke-dashoffset',-off);off+=dash;});
  const tEl=document.getElementById('donutTotal');
  if(tEl){animNum(tEl,total);setTimeout(()=>{tEl.textContent=total;},1300);}
  const leg=document.getElementById('donutLegend');
  if(leg)leg.innerHTML=data.map(d=>`<div class="dl-item"><span class="dl-dot" style="background:${d.color}"></span><span>${d.label}</span><span class="dl-val">${d.value}%</span></div>`).join('');
}

// ══════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════
function showToast(msg){let t=document.getElementById('toast');if(!t){t=document.createElement('div');t.id='toast';t.style.cssText='position:fixed;bottom:24px;right:24px;z-index:9999;background:var(--yellow-400);color:var(--blue-950);padding:12px 22px;border-radius:12px;font-size:0.88rem;font-weight:700;font-family:var(--font-h);box-shadow:var(--shadow-glow-y);transform:translateY(20px);opacity:0;transition:all 0.3s ease;';document.body.appendChild(t);}t.textContent=msg;requestAnimationFrame(()=>{t.style.transform='translateY(0)';t.style.opacity='1';});setTimeout(()=>{t.style.transform='translateY(20px)';t.style.opacity='0';},2500);}
function escH(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function escA(s){return String(s).replace(/"/g,'&quot;').replace(/'/g,'&#39;');}
function escRx(s){return s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');}

// ══════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════


export function initFillyApp() {
  metricsReady = false;
  initLanding();
  initNav();
  initWrite();
}
