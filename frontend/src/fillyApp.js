/* ─────────────────────────────────────────────
   FILLY – Unified Application Logic
───────────────────────────────────────────── */

import { analyzeText } from './api/fillyApi.js'
import {
  applyAcceptedSuggestionsSafely,
  normalizeSuggestions,
  segmentText,
} from './suggestions.js'

// ══════════════════════════════════════════════
// DATA
// ══════════════════════════════════════════════
// Analytics state
const ANALYTICS = {
  unnormalizedWords: 0,
  grammarFound: 0,
  acceptedSuggestions: 0,
  ignoredSuggestions: 0
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
window.analysisResult = null;
let activeTooltip = null;
let currentAnalysisText = null;
let analysisRequestId = 0;
let activeSuggestionId = null;
let reanalysisTimer = null;

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
    if (!window.isProgrammaticEdit) {
      const checkButton = document.getElementById('checkBtn');
      const reanalyze = currentAnalysisText !== null || reanalysisTimer !== null || checkButton.disabled;
      invalidateAnalysis();
      if (reanalyze) {
        reanalysisTimer = setTimeout(() => {
          reanalysisTimer = null;
          if (ta.value.trim()) processText();
        }, 650);
      }
    }
    updateStats();
  });
  document.getElementById('checkBtn').addEventListener('click', processText);
document.getElementById('copyBtn').addEventListener('click', () => {
    const text = getOutputPlainText();
    if(!text){showToast('There is no analyzed text to copy.');return;}
    if(text){ navigator.clipboard.writeText(text).then(()=>{      const b=document.getElementById('copyBtn');
      b.textContent='Copied!';
      setTimeout(()=>{b.textContent='Copy accepted text';},2000);
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
        invalidateAnalysis();
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
if(!text){showToast('There is no analyzed text to save.');return;}  const rawTitle=document.getElementById('docTitle').value.trim() || 'Untitled';
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
  showToast('Accepted text downloaded.');
}

function invalidateAnalysis() {
  analysisRequestId += 1;
  clearTimeout(reanalysisTimer);
  reanalysisTimer = null;
  currentAnalysisText = null;
  activeSuggestionId = null;
  window.segments = [];
  window.suggestions = {};
  window.analysisResult = null;
  removeActiveTooltip();
  setAnalyzeLoading(false);
  clearEditorHighlights();
  renderOutputPlaceholder();

  const body = document.getElementById('recsBody');
  if (body) body.innerHTML = '<div class="recs-empty"><p>No current suggestions.</p><span>Analyze the edited text to check it again.</span></div>';
  const count = document.getElementById('recsCount');
  if (count) count.textContent = 'Suggestions (0)';
  const bulkActions = document.querySelector('.recs-bulk-actions');
  if (bulkActions) bulkActions.style.display = 'none';
}

function clearEditorHighlights() {
  const editor = document.getElementById('editorTextarea');
  editor.querySelectorAll('.original-highlight').forEach((highlight) => {
    const parent = highlight.parentNode;
    while (highlight.firstChild) parent.insertBefore(highlight.firstChild, highlight);
    parent.removeChild(highlight);
  });
}

function setAnalyzeLoading(loading) {
  const button = document.getElementById('checkBtn');
  if (!button) return;
  if (!button.dataset.defaultMarkup) button.dataset.defaultMarkup = button.innerHTML;
  button.disabled = loading;
  button.setAttribute('aria-busy', String(loading));
  button.innerHTML = loading
    ? '<span class="spinner spinner-small" aria-hidden="true"></span> Analyzing…'
    : button.dataset.defaultMarkup;
}

function renderOutputPlaceholder() {
  const out = document.getElementById('outputDisplay');
  if (!out) return;
  out.innerHTML = '<div class="output-placeholder"><svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg><p>Corrected text will appear here</p><span>Click "Check Text" to process</span></div>';
  const chars = document.getElementById('outputChars');
  if (chars) chars.innerHTML = '&nbsp;';
}

function showAnalysisError(error) {
  const message = error && error.message ? error.message : 'Could not connect to the FILLY service.';
  document.getElementById('outputDisplay').innerHTML =
    '<div class="api-error" role="alert"><strong>Analysis failed</strong><p>' + escH(message) + '</p></div>';
  document.getElementById('recsBody').innerHTML =
    '<div class="recs-empty api-error" role="alert"><p>' + escH(message) + '</p></div>';
  document.getElementById('recsCount').textContent = 'Suggestions (error)';
  document.getElementById('outputChars').innerHTML = '&nbsp;';
  const bulkActions = document.querySelector('.recs-bulk-actions');
  if (bulkActions) bulkActions.style.display = 'none';
}

function processApiResponse(text, data) {
  if (!data || data.original_text !== text) {
    throw new Error('The analysis result does not match the current editor text. Please analyze again.');
  }
  if (typeof data.normalized_text !== 'string' || typeof data.corrected_text !== 'string') {
    throw new Error('The analysis response is missing normalized or corrected text.');
  }

  const list = normalizeSuggestions(text, data.suggestions);
  const suggestions = Object.fromEntries(list.map((suggestion) => [suggestion.id, suggestion]));
  return {
    segments: segmentText(text, list),
    suggestions,
    result: {
      original_text: data.original_text,
      normalized_text: data.normalized_text,
      corrected_text: data.corrected_text,
      normalization: data.normalization || { changes: [] },
      gec: data.gec || { iterations: 0, changes: [], iteration_outputs: [] },
    },
  };
}

async function processText() {
  const editor = document.getElementById('editorTextarea');
  const text = editor.value;
  clearTimeout(reanalysisTimer);
  reanalysisTimer = null;
  if (!text.trim()) {
    showToast('Please enter some text to check.');
    editor.focus();
    return;
  }

  const requestId = ++analysisRequestId;
  currentAnalysisText = null;
  window.segments = [];
  window.suggestions = {};
  window.analysisResult = null;
  activeSuggestionId = null;
  removeActiveTooltip();
  clearEditorHighlights();
  setAnalyzeLoading(true);
  document.getElementById('outputDisplay').innerHTML =
    '<div class="processing-state"><div class="spinner"></div><span>Analyzing your Filipino text…</span></div>';
  document.getElementById('recsBody').innerHTML =
    '<div class="recs-empty"><div class="spinner"></div><p>Analyzing…</p></div>';
  document.getElementById('recsCount').textContent = 'Suggestions (…)';
  const bulkActions = document.querySelector('.recs-bulk-actions');
  if (bulkActions) bulkActions.style.display = 'none';

  try {
    const data = await analyzeText(text);
    if (requestId !== analysisRequestId || editor.value !== text) return;

    const analysis = processApiResponse(text, data);
    currentAnalysisText = text;
    window.segments = analysis.segments;
    window.suggestions = analysis.suggestions;
    window.analysisResult = analysis.result;
    trackAnalytics(Object.values(analysis.suggestions));
    reRenderAll();
  } catch (error) {
    if (requestId !== analysisRequestId || editor.value !== text) return;
    showAnalysisError(error);
  } finally {
    if (requestId === analysisRequestId) setAnalyzeLoading(false);
  }
}
function reRenderAll() {
  renderEditor();
  renderOutputDisplay();
  renderRecs();
}

function suggestionKind(suggestion) {
  if (suggestion.type === 'norm') return 'Normalization';
  if (suggestion.type === 'combined') return 'Combined correction';
  return 'Grammar';
}

function suggestionClass(suggestion) {
  if (suggestion.type === 'norm') return 'norm';
  if (suggestion.type === 'combined') return 'combined';
  return 'gram';
}

function renderEditor() {
  const editor = document.getElementById('editorTextarea');
  if (currentAnalysisText === null) {
    const plainText = editor.value;
    editor.textContent = plainText;
    return;
  }

  const htmlParts = [];
  window.segments.forEach((segment) => {
    if (!segment.suggestionId) {
      htmlParts.push(escH(segment.text));
      return;
    }

    const suggestion = window.suggestions[segment.suggestionId];
    if (!suggestion || suggestion.status !== 'pending') {
      htmlParts.push(escH(segment.text));
      return;
    }

    const className = 'underline-' + suggestionClass(suggestion);
    if (segment.insertion) {
      htmlParts.push('<span class="original-highlight insertion-marker ' + className
        + '" data-sug-id="' + escH(suggestion.id) + '" aria-label="Suggested insertion"></span>');
    } else {
      htmlParts.push('<span class="original-highlight ' + className
        + '" data-sug-id="' + escH(suggestion.id) + '">' + escH(segment.text) + '</span>');
    }
  });

  editor.innerHTML = htmlParts.join('');
  attachEditorListeners();
}
function renderOutputDisplay() {
  const out = document.getElementById('outputDisplay');
  const result = window.analysisResult;
  if (!result) {
    renderOutputPlaceholder();
    return;
  }

  const normalized = escH(result.normalized_text);
  const corrected = escH(result.corrected_text);
  out.innerHTML = '<section class="result-stage normalization-stage">'
    + '<h4>After normalization</h4><div class="output-text">' + normalized + '</div></section>'
    + '<section class="result-stage grammar-stage">'
    + '<h4>Correction preview (includes pending suggestions)</h4><div class="output-text">' + corrected + '</div></section>';
  document.getElementById('outputChars').textContent =
    Array.from(result.corrected_text).length + ' characters';
}

function getOutputPlainText() {
  const editor = document.getElementById('editorTextarea');
  return window.analysisResult && currentAnalysisText === editor.value ? editor.value : '';
}
function renderRecs() {
  const body = document.getElementById('recsBody');
  const suggestions = Object.values(window.suggestions);
  const pending = suggestions.filter((suggestion) => suggestion.status === 'pending');
  const bulkActions = document.querySelector('.recs-bulk-actions');
  if (bulkActions) bulkActions.style.display = pending.length > 0 ? 'flex' : 'none';
  document.getElementById('recsCount').textContent = 'Suggestions (' + pending.length + ')';

  if (pending.length === 0) {
    body.innerHTML = '<div class="recs-empty"><p>No pending suggestions.</p>'
      + '<span>The backend returned ' + suggestions.length + ' correction(s).</span></div>';
    return;
  }

  body.innerHTML = '';
  pending.forEach((suggestion, index) => {
    const card = document.createElement('article');
    card.className = 'sug-card' + (suggestion.id === activeSuggestionId ? ' selected' : '');
    card.dataset.sugId = suggestion.id;
    card.setAttribute('role', 'group');
    card.setAttribute('aria-label', suggestionKind(suggestion)
      + ' suggestion: ' + suggestion.original + ' to ' + suggestion.replacement);
    card.style.animationDelay = (index * 0.06) + 's';
    const kind = suggestionKind(suggestion);
    card.innerHTML = '<div class="sug-header"><button class="sug-select sug-tag '
      + 'tag-' + suggestionClass(suggestion) + '" type="button">' + kind + '</button>'
      + (suggestion.tag ? '<span class="sug-label">' + escH(suggestion.tag) + '</span>' : '')
      + '</div><div class="sug-from">' + escH(suggestion.original) + '</div>'
      + '<div class="sug-to">&rarr; ' + escH(suggestion.replacement) + '</div>'
      + '<div class="sug-actions"><button class="sug-btn sug-accept" type="button">Accept</button>'
      + '<button class="sug-btn sug-ignore" type="button">Ignore</button></div>';
    body.appendChild(card);

    card.addEventListener('click', (event) => {
      if (event.target.closest('button')) return;
      selectSuggestion(suggestion.id);
    });
    card.querySelector('.sug-select').addEventListener('click', (event) => {
      event.stopPropagation();
      selectSuggestion(suggestion.id);
    });
    card.querySelector('.sug-accept').addEventListener('click', (event) => {
      event.stopPropagation();
      acceptSuggestion(suggestion.id);
    });
    card.querySelector('.sug-ignore').addEventListener('click', (event) => {
      event.stopPropagation();
      ignoreSuggestion(suggestion.id);
    });
  });
}

function acceptSuggestion(id) {
  const suggestion = window.suggestions[id];
  if (!suggestion || suggestion.status !== 'pending') return;
  const editor = document.getElementById('editorTextarea');
  const suggestions = Object.values(window.suggestions);
  const current = currentAnalysisText === editor.value;
  suggestion.status = 'accepted';
  const applied = current
    ? applyAcceptedSuggestionsSafely(currentAnalysisText, suggestions)
    : { ok: false, reason: 'The editor text changed after analysis.' };

  if (!applied.ok) {
    invalidateAnalysis();
    showToast('The suggestion was stale. Analyze the current text again.');
    return;
  }
  editor.value = applied.text;
  ANALYTICS.acceptedSuggestions += 1;
  updateAnalytics();
  updateStats();
  invalidateAnalysis();
  if (applied.text.trim()) processText();
}

function ignoreSuggestion(id) {
  const suggestion = window.suggestions[id];
  if (!suggestion || suggestion.status !== 'pending') return;
  suggestion.status = 'ignored';
  activeSuggestionId = null;
  ANALYTICS.ignoredSuggestions++;
  updateAnalytics();
  reRenderAll();
}

function acceptAllSuggestions() {
  const editor = document.getElementById('editorTextarea');
  const pending = Object.values(window.suggestions).filter((suggestion) => suggestion.status === 'pending');
  if (pending.length === 0) return;
  pending.forEach((suggestion) => { suggestion.status = 'accepted'; });
  const applied = currentAnalysisText === editor.value
    ? applyAcceptedSuggestionsSafely(currentAnalysisText, Object.values(window.suggestions))
    : { ok: false, reason: 'The editor text changed after analysis.' };

  if (!applied.ok) {
    invalidateAnalysis();
    showToast('Suggestions were stale. Analyze the current text again.');
    return;
  }
  editor.value = applied.text;
  ANALYTICS.acceptedSuggestions += pending.length;
  updateAnalytics();
  updateStats();
  invalidateAnalysis();
  if (applied.text.trim()) processText();
}

function ignoreAllSuggestions() {
  const pending = Object.values(window.suggestions).filter((suggestion) => suggestion.status === 'pending');
  pending.forEach((suggestion) => { suggestion.status = 'ignored'; });
  activeSuggestionId = null;
  ANALYTICS.ignoredSuggestions += pending.length;
  updateAnalytics();
  reRenderAll();
}

function selectSuggestion(id) {
  const suggestion = window.suggestions[id];
  if (!suggestion || suggestion.status !== 'pending') return;
  if (currentAnalysisText !== document.getElementById('editorTextarea').value) {
    invalidateAnalysis();
    showToast('The editor text changed. Analyze it again to refresh suggestions.');
    return;
  }

  activeSuggestionId = id;
  selectTextRange(suggestion.start, suggestion.end);
  document.querySelectorAll('.sug-card').forEach((card) => {
    const active = card.dataset.sugId === id;
    card.classList.toggle('selected', active);
    if (active) card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  });
}

function selectTextRange(start, end) {
  const editor = document.getElementById('editorTextarea');
  const walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  let node;
  while ((node = walker.nextNode())) textNodes.push(node);

  const locate = (offset) => {
    let passed = 0;
    for (const textNode of textNodes) {
      const length = textNode.nodeValue.length;
      if (offset <= passed + length) return { node: textNode, offset: offset - passed };
      passed += length;
    }
    if (textNodes.length > 0) {
      const last = textNodes[textNodes.length - 1];
      return { node: last, offset: last.nodeValue.length };
    }
    return { node: editor, offset: editor.childNodes.length };
  };

  const startPoint = locate(start);
  const endPoint = locate(end);
  const range = document.createRange();
  range.setStart(startPoint.node, startPoint.offset);
  range.setEnd(endPoint.node, endPoint.offset);
  const selection = window.getSelection();
  selection.removeAllRanges();
  selection.addRange(range);
  editor.focus({ preventScroll: true });
}
function showTooltipForSpan(span, sug) {
  removeActiveTooltip();

  const tooltip = document.createElement('div');
  tooltip.className = 'editor-tooltip';
  tooltip.innerHTML = `
    <div class="tooltip-title">${suggestionKind(sug)}</div>
    <div class="tooltip-body">
      <span class="tooltip-from">${escH(sug.original)}</span>
      <span style="color:var(--text-4)">&rarr;</span>
      <span class="tooltip-to">${escH(sug.replacement)}</span>
    </div>
    <div class="tooltip-footer">
      <button class="tooltip-btn btn-accept">Accept</button>
      <button class="tooltip-btn btn-ignore">Ignore</button>
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
      selectSuggestion(sugId);
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
  setEl('statAccepted', a.acceptedSuggestions);
  setEl('statIgnored', a.ignoredSuggestions);
}

function trackAnalytics(suggestions) {
  // Refresh current-check suggestions; accept/ignore action totals last for this page session.
  ANALYTICS.unnormalizedWords = 0;
  ANALYTICS.grammarFound = 0;

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
  ANALYTICS.acceptedSuggestions = 0;
  ANALYTICS.ignoredSuggestions = 0;
  initLanding();
  initNav();
  initWrite();
}
