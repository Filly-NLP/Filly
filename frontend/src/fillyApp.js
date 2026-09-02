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

// Analytics state
const ANALYTICS = {
  unnormalizedWords: 0,
  grammarFound: 0,
  grammarFixed: 0,
  grammarIgnored: 0
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
      if(v==='analytics') setTimeout(updateAnalytics,200);
    });
  });
  const tog=document.getElementById('menuToggle'),sb=document.getElementById('sidebar'),ov=document.getElementById('sidebarOverlay');
  if(tog)tog.addEventListener('click',()=>{sb.classList.toggle('open');ov.classList.toggle('active');});
  if(ov)ov.addEventListener('click',()=>{sb.classList.remove('open');ov.classList.remove('active');});
  const back=document.getElementById('backToLanding');
  if(back)back.addEventListener('click',goToLanding);

  // Theme toggle (Sidebar & Landing Page)
  const toggleBtnSidebar = document.getElementById('themeToggle');
  const toggleBtnLanding = document.getElementById('landingThemeToggle');
  const btns = [toggleBtnSidebar, toggleBtnLanding].filter(Boolean);
  
  // Light mode is default — only go dark if explicitly saved
  const saved = localStorage.getItem('filly-theme');
  if(saved !== 'dark'){
    document.body.classList.add('light-mode');
    btns.forEach(btn => btn.querySelector('.theme-label') && (btn.querySelector('.theme-label').textContent = 'Dark Mode'));
  }
  
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      document.body.classList.toggle('light-mode');
      const isLight = document.body.classList.contains('light-mode');
      localStorage.setItem('filly-theme', isLight ? 'light' : 'dark');
      btns.forEach(b => {
        const label = b.querySelector('.theme-label');
        if(label) label.textContent = isLight ? 'Dark Mode' : 'Light Mode';
      });
    });
  });
}

// ══════════════════════════════════════════════
// WRITE VIEW
// ══════════════════════════════════════════════
const WORD_LIMIT = 250;

// State variables
window.segments = [];
window.suggestions = {};
window.isProgrammaticEdit = false;
let activeTooltip = null;

function initWrite() {
  const ta = document.getElementById('editorTextarea');
  
  // Set up backward compatibility properties/methods for contenteditable div
  Object.defineProperty(ta, 'value', {
    get() {
      return this.innerText;
    },
    set(val) {
      this.innerText = val;
    },
    configurable: true
  });
  
  ta.selectionStart = 0;
  ta.setSelectionRange = function(start, end) {
    try {
      const range = document.createRange();
      const sel = window.getSelection();
      if (this.childNodes.length > 0) {
        const node = this.childNodes[0];
        const length = node.length || 0;
        range.setStart(node, Math.min(start, length));
        range.collapse(true);
        sel.removeAllRanges();
        sel.addRange(range);
      }
    } catch (e) {
      console.warn("setSelectionRange fallback:", e);
    }
  };

  ta.addEventListener('input', () => {
    // If the edit is not programmatic (i.e. user typed manually)
    if (!window.isProgrammaticEdit) {
      // Clear current recommendations
      window.segments = [];
      window.suggestions = {};
      removeActiveTooltip();
      
      // Reset recommendations UI
      const body = document.getElementById('recsBody');
      if (body) {
        body.innerHTML = `<div class="recs-empty"><svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg><p>No suggestions yet.</p><span>Start typing or paste text to see recommendations.</span></div>`;
      }
      const recsCount = document.getElementById('recsCount');
      if (recsCount) recsCount.textContent = 'Suggestions (0)';
      
      const bulkActions = document.querySelector('.recs-bulk-actions');
      if (bulkActions) bulkActions.style.display = 'none';

      // Clear output display
      const out = document.getElementById('outputDisplay');
      if (out) {
        out.innerHTML = `<div class="output-placeholder"><svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg><p>Corrected text will appear here</p><span>Click "Check Text" to process</span></div>`;
      }
      const outChars = document.getElementById('outputChars');
      if (outChars) outChars.innerHTML = '&nbsp;';
    }
    updateStats();
  });

  document.getElementById('checkBtn').addEventListener('click', processText);
  document.getElementById('copyBtn').addEventListener('click', () => {
    const text = getOutputPlainText();
    if(text){ navigator.clipboard.writeText(text).then(()=>{
      const b=document.getElementById('copyBtn');
      b.innerHTML='<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Copied!';
      setTimeout(()=>{b.innerHTML='<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy';},2000);
    });}
  });
  
  document.getElementById('saveBtn').addEventListener('click', downloadCorrectedOutput);
  document.getElementById('uploadBtn').addEventListener('click', () => {
    const inp = document.createElement('input'); inp.type = 'file'; inp.accept = '.txt';
    inp.onchange = e => {
      const f = e.target.files[0]; if (!f) return;
      const r = new FileReader(); r.onload = ev => {
        window.isProgrammaticEdit = true;
        ta.value = limitWords(ev.target.result);
        window.isProgrammaticEdit = false;
        updateStats();
      }; r.readAsText(f);
    };
    inp.click();
  });

  // Wire up bulk actions
  const bulkAccept = document.getElementById('bulkAcceptBtn');
  if (bulkAccept) bulkAccept.addEventListener('click', acceptAllSuggestions);
  
  const bulkIgnore = document.getElementById('bulkIgnoreBtn');
  if (bulkIgnore) bulkIgnore.addEventListener('click', ignoreAllSuggestions);

  // Global click listener to dismiss tooltip
  document.addEventListener('click', (e) => {
    if (activeTooltip && !activeTooltip.contains(e.target)) {
      removeActiveTooltip();
    }
  });

  updateStats();
}

function updateStats(){
  const ta=document.getElementById('editorTextarea');
  const limited=limitWords(ta.value);
  if(limited!==ta.value){
    const pos=ta.selectionStart;
    window.isProgrammaticEdit = true;
    ta.value=limited;
    window.isProgrammaticEdit = false;
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
  const text=getOutputPlainText();
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

function segmentText(text, suggestionsList) {
  // Sort suggestions ascending by start to partition
  const sorted = [];
  suggestionsList.forEach(s => {
    if (!sorted.some(x => (s.start < x.end && s.end > x.start))) {
      sorted.push(s);
    }
  });
  sorted.sort((a, b) => a.start - b.start);

  const segments = [];
  let lastIdx = 0;
  sorted.forEach(s => {
    if (s.start > lastIdx) {
      segments.push({
        text: text.slice(lastIdx, s.start),
        sugId: null
      });
    }
    segments.push({
      text: text.slice(s.start, s.end),
      sugId: s.id
    });
    lastIdx = s.end;
  });
  if (lastIdx < text.length) {
    segments.push({
      text: text.slice(lastIdx),
      sugId: null
    });
  }
  return segments;
}

function processApiResponse(text, data) {
  const normalizations = data.normalizations || [];
  const grammarCorrections = data.grammar_corrections || [];
  
  const suggestions = {};
  let sugIdx = 0;

  normalizations.forEach(n => {
    const id = `sug-${sugIdx++}`;
    suggestions[id] = {
      id,
      start: n.start,
      end: n.end,
      from: n.word,
      to: n.suggestion,
      type: 'norm',
      label: '', // when you are recommending a changes for normalization, do not add abbreviations, slangs, spelling variations next to the normalization tag
      status: 'pending'
    };
  });

  grammarCorrections.forEach(g => {
    const id = `sug-${sugIdx++}`;
    suggestions[id] = {
      id,
      start: g.start,
      end: g.end,
      from: g.original,
      to: g.correction,
      type: 'gram',
      label: g.rule ? g.rule.charAt(0).toUpperCase() + g.rule.slice(1) : 'Grammar',
      status: 'pending'
    };
  });

  const list = Object.values(suggestions);
  const segments = segmentText(text, list);

  return { segments, suggestions };
}

function processText(){
  const ta=document.getElementById('editorTextarea'), text=ta.value.trim();
  if(!text){ta.focus();return;}
  const out=document.getElementById('outputDisplay');
  out.innerHTML='<div class="processing-state"><div class="spinner"></div><span style="color:var(--text-3);font-size:0.9rem;">Processing…</span></div>';
  document.getElementById('suggestionsBar') && (document.getElementById('suggestionsBar').style.display='none');
  
  // Clear recs and hide bulk container
  document.getElementById('recsBody').innerHTML='<div class="recs-empty"><div class="spinner"></div><p style="margin-top:12px;font-size:0.85rem;color:var(--text-3);">Analyzing…</p></div>';
  document.getElementById('recsCount').textContent='Suggestions (…)';
  const bulkActions = document.querySelector('.recs-bulk-actions');
  if (bulkActions) bulkActions.style.display = 'none';

  fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  })
  .then(res => {
    if (!res.ok) throw new Error('API request failed');
    return res.json();
  })
  .then(data => {
    const { segments, suggestions } = processApiResponse(text, data);
    window.segments = segments;
    window.suggestions = suggestions;

    // Track analytics
    trackAnalytics(Object.values(suggestions));

    // Render everything
    window.isProgrammaticEdit = true;
    reRenderAll();
    window.isProgrammaticEdit = false;
  })
  .catch(err => {
    console.error(err);
    out.innerHTML='<div style="color:red;font-size:0.9rem;padding:20px;text-align:center;">Failed to connect to backend server.</div>';
    document.getElementById('recsBody').innerHTML='<div class="recs-empty"><p style="color:red;">Error loading suggestions.</p></div>';
  });
}

function reRenderAll() {
  renderEditor();
  renderOutputDisplay();
  renderRecs();
}

function renderEditor() {
  const ta = document.getElementById('editorTextarea');
  if (!window.segments || window.segments.length === 0) return;

  let htmlParts = [];
  window.segments.forEach(seg => {
    if (!seg.sugId) {
      htmlParts.push(escH(seg.text));
    } else {
      const sug = window.suggestions[seg.sugId];
      if (sug.status === 'pending') {
        const className = sug.type === 'norm' ? 'underline-norm' : 'underline-gram';
        htmlParts.push(`<span class="original-highlight ${className}" data-sug-id="${sug.id}">${escH(seg.text)}</span>`);
      } else {
        htmlParts.push(escH(sug.from));
      }
    }
  });

  ta.innerHTML = htmlParts.join('');
  attachEditorListeners();
}

function renderOutputDisplay() {
  const out = document.getElementById('outputDisplay');
  if (!window.segments || window.segments.length === 0) return;

  let htmlParts = [];
  window.segments.forEach(seg => {
    if (!seg.sugId) {
      htmlParts.push(escH(seg.text));
    } else {
      const sug = window.suggestions[seg.sugId];
      if (sug.status === 'pending') {
        htmlParts.push(`<mark class="output-highlight mark-${sug.type}" data-sug-id="${sug.id}">${escH(sug.to)}</mark>`);
      } else if (sug.status === 'accepted') {
        htmlParts.push(escH(sug.to));
      } else {
        htmlParts.push(escH(sug.from));
      }
    }
  });

  let html = htmlParts.join('');
  html = capitalizeHtmlFirstLetter(html);

  const plainText = getOutputPlainText();
  if (plainText.length > 0 && !/[.!?]$/.test(plainText.trim())) {
    html = html.trimEnd() + '.';
  }

  out.innerHTML = `<div class="output-text">${html}</div>`;
  document.getElementById('outputChars').textContent = `${plainText.length} characters`;
}

function getOutputPlainText() {
  let parts = [];
  window.segments.forEach(seg => {
    if (!seg.sugId) {
      parts.push(seg.text);
    } else {
      const sug = window.suggestions[seg.sugId];
      if (sug.status === 'pending' || sug.status === 'accepted') {
        parts.push(sug.to);
      } else {
        parts.push(sug.from);
      }
    }
  });
  let text = parts.join('');
  if (text.length > 0) text = text.charAt(0).toUpperCase() + text.slice(1);
  if (text.length > 0 && !/[.!?]$/.test(text.trim())) text = text.trimEnd() + '.';
  return text;
}

function capitalizeHtmlFirstLetter(html) {
  let inTag = false;
  for (let i = 0; i < html.length; i++) {
    if (html[i] === '<') {
      inTag = true;
    } else if (html[i] === '>') {
      inTag = false;
    } else if (!inTag) {
      return html.slice(0, i) + html[i].toUpperCase() + html.slice(i + 1);
    }
  }
  return html;
}

function renderRecs() {
  const body = document.getElementById('recsBody');
  const sugs = Object.values(window.suggestions).filter(s => s.status === 'pending');

  const bulkActions = document.querySelector('.recs-bulk-actions');
  if (bulkActions) {
    bulkActions.style.display = sugs.length > 0 ? 'flex' : 'none';
  }

  document.getElementById('recsCount').textContent = `Suggestions (${sugs.length})`;

  if (sugs.length === 0) {
    body.innerHTML = `<div class="recs-empty"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#4ADE80" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polyline points="16 9 10.5 14.5 8 12"/></svg><p>All resolved!</p><span>Your text has been improved.</span></div>`;
    return;
  }

  body.innerHTML = '';
  sugs.forEach((s, i) => {
    const card = document.createElement('div');
    card.className = 'sug-card';
    card.style.animationDelay = `${i * 0.06}s`;
    card.innerHTML = `
      <div class="sug-header">
        <span class="sug-tag ${s.type === 'norm' ? 'tag-norm' : 'tag-gram'}">${s.type === 'norm' ? 'Normalization' : 'Grammar'}</span>
        ${s.label ? `<span class="sug-label">${escH(s.label)}</span>` : ''}
      </div>
      <div class="sug-from">${escH(s.from)}</div>
      <div class="sug-to">→ ${escH(s.to)}</div>
      <div class="sug-actions">
        <button class="sug-btn sug-accept" data-id="${s.id}">Accept</button>
        <button class="sug-btn sug-ignore" data-id="${s.id}">Ignore</button>
      </div>`;
    body.appendChild(card);
  });

  body.querySelectorAll('.sug-accept').forEach(b => {
    b.addEventListener('click', () => {
      acceptSuggestion(b.dataset.id);
    });
  });
  body.querySelectorAll('.sug-ignore').forEach(b => {
    b.addEventListener('click', () => {
      ignoreSuggestion(b.dataset.id);
    });
  });
}

function acceptSuggestion(id) {
  const sug = window.suggestions[id];
  if (!sug) return;
  sug.status = 'accepted';
  ANALYTICS.grammarFixed++;
  updateAnalytics();
  
  window.isProgrammaticEdit = true;
  reRenderAll();
  window.isProgrammaticEdit = false;
}

function ignoreSuggestion(id) {
  const sug = window.suggestions[id];
  if (!sug) return;
  sug.status = 'ignored';
  ANALYTICS.grammarIgnored++;
  updateAnalytics();
  
  window.isProgrammaticEdit = true;
  reRenderAll();
  window.isProgrammaticEdit = false;
}

function acceptAllSuggestions() {
  const sugs = Object.values(window.suggestions).filter(s => s.status === 'pending');
  sugs.forEach(s => {
    s.status = 'accepted';
    ANALYTICS.grammarFixed++;
  });
  updateAnalytics();
  
  window.isProgrammaticEdit = true;
  reRenderAll();
  window.isProgrammaticEdit = false;
}

function ignoreAllSuggestions() {
  const sugs = Object.values(window.suggestions).filter(s => s.status === 'pending');
  sugs.forEach(s => {
    s.status = 'ignored';
    ANALYTICS.grammarIgnored++;
  });
  updateAnalytics();
  
  window.isProgrammaticEdit = true;
  reRenderAll();
  window.isProgrammaticEdit = false;
}

function showTooltipForSpan(span, sug) {
  removeActiveTooltip();

  const tooltip = document.createElement('div');
  tooltip.className = 'editor-tooltip';
  tooltip.innerHTML = `
    <div class="tooltip-title">${sug.type === 'norm' ? 'Normalization' : 'Grammar'}</div>
    <div class="tooltip-body">
      <span class="tooltip-from">${escH(sug.from)}</span>
      <span style="color:var(--text-4)">&rarr;</span>
      <span class="tooltip-to">${escH(sug.to)}</span>
    </div>
    <div class="tooltip-footer">
      <button class="tooltip-btn btn-accept" data-id="${sug.id}">Accept</button>
      <button class="tooltip-btn btn-ignore" data-id="${sug.id}">Ignore</button>
    </div>
  `;

  document.body.appendChild(tooltip);
  activeTooltip = tooltip;

  const rect = span.getBoundingClientRect();
  const left = rect.left + window.scrollX + (rect.width / 2) - (tooltip.offsetWidth / 2);
  const top = rect.top + window.scrollY - tooltip.offsetHeight - 8;
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;

  tooltip.querySelector('.btn-accept').addEventListener('click', (e) => {
    e.stopPropagation();
    acceptSuggestion(sug.id);
    removeActiveTooltip();
  });
  tooltip.querySelector('.btn-ignore').addEventListener('click', (e) => {
    e.stopPropagation();
    ignoreSuggestion(sug.id);
    removeActiveTooltip();
  });
}

function removeActiveTooltip() {
  if (activeTooltip) {
    activeTooltip.remove();
    activeTooltip = null;
  }
}

function attachEditorListeners() {
  const ta = document.getElementById('editorTextarea');
  ta.querySelectorAll('.original-highlight').forEach(span => {
    const sugId = span.dataset.sugId;
    const sug = window.suggestions[sugId];
    if (!sug) return;

    span.addEventListener('click', (e) => {
      e.stopPropagation();
      showTooltipForSpan(span, sug);
    });
  });
}


// ══════════════════════════════════════════════
// ANALYTICS
// ══════════════════════════════════════════════
function updateAnalytics() {
  const a = ANALYTICS;
  // Update stat values
  const setEl = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  setEl('statUnnormalized', a.unnormalizedWords);
  setEl('statFound', a.grammarFound);
  setEl('statFixed', a.grammarFixed);
  setEl('statIgnored', a.grammarIgnored);

  // Quality score: 100 minus total issues (minimum 0)
  const totalIssues = a.unnormalizedWords + a.grammarFound;
  const score = Math.max(0, 100 - totalIssues * 5);
  setEl('qualityScore', score);

  // Update ring
  const ring = document.getElementById('qualityRing');
  if (ring) {
    const circ = 2 * Math.PI * 52;
    const offset = circ - (score / 100) * circ;
    ring.style.strokeDasharray = circ;
    ring.style.strokeDashoffset = offset;
  }

  // Update message
  const msgEl = document.getElementById('qualityMessage');
  if (msgEl) {
    if (score >= 90) msgEl.textContent = 'Excellent! Your Filipino writing is well-formed.';
    else if (score >= 70) msgEl.textContent = 'Good writing quality. A few issues were found.';
    else if (score >= 50) msgEl.textContent = 'Fair quality. Consider reviewing the suggestions.';
    else msgEl.textContent = 'Needs improvement. Review and accept the suggestions.';
  }
}

function trackAnalytics(suggestions) {
  // Reset counts
  ANALYTICS.unnormalizedWords = 0;
  ANALYTICS.grammarFound = 0;
  ANALYTICS.grammarFixed = 0;
  ANALYTICS.grammarIgnored = 0;

  suggestions.forEach(s => {
    if (s.type === 'norm') {
      ANALYTICS.unnormalizedWords++;
    } else {
      ANALYTICS.grammarFound++;
    }
  });
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
  ANALYTICS.unnormalizedWords = 0;
  ANALYTICS.grammarFound = 0;
  ANALYTICS.grammarFixed = 0;
  ANALYTICS.grammarIgnored = 0;
  initLanding();
  initNav();
  initWrite();
}
