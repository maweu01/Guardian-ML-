/* ═══════════════════════════════════════════════════════════════
   GUARDIAN ML — Frontend Application Controller
   Manages: pipeline state, API calls, Plotly rendering, terminal
═══════════════════════════════════════════════════════════════ */

'use strict';

// ─── State ────────────────────────────────────────────────────────────────────
const STATE = {
  jobId:      null,
  apiBase:    window.location.origin,
  activePanel:'upload',
  stages:     { upload: false, process: false, train: false, predict: false, visualize: false },
  schema:     null,
  report:     null,
  predictions:null,
  charts:     {},
};

// ─── Initialization ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  startClock();
  log('INFO', 'GUARDIAN ML system initialized.');
  log('INFO', 'Awaiting data ingest. Upload a dataset to begin.');
  setStatus('SYSTEM READY', 'ok');
});

// ─── Clock ────────────────────────────────────────────────────────────────────
function startClock() {
  const el = document.getElementById('clockEl');
  const tick = () => {
    const now = new Date();
    el.textContent = now.toUTCString().slice(17, 25) + ' UTC';
  };
  tick();
  setInterval(tick, 1000);
}

// ─── Status Bar ───────────────────────────────────────────────────────────────
function setStatus(label, type = 'ok') {
  const dot = document.getElementById('statusDot');
  const lbl = document.getElementById('statusLabel');
  dot.className  = 'status-dot' + (type !== 'ok' ? ` ${type}` : '');
  lbl.textContent = label;
  lbl.style.color = type === 'error' ? 'var(--red)' : type === 'warning' ? 'var(--amber)' : 'var(--green)';
}

// ─── Terminal Logger ──────────────────────────────────────────────────────────
function log(level, message) {
  const body  = document.getElementById('termBody');
  const ts    = new Date().toISOString().slice(11, 23);
  const line  = document.createElement('div');
  line.className = 'log-line';
  const lvlClass = { INFO:'info', SUCCESS:'success', WARN:'warn', ERROR:'error' }[level] || 'info';
  line.innerHTML =
    `<span class="log-ts">${ts}</span>` +
    `<span class="log-level-${lvlClass}">[${level}]</span>` +
    `<span class="log-msg ${level !== 'INFO' ? 'hi' : ''}">${escHtml(message)}</span>`;
  body.appendChild(line);
  body.scrollTop = body.scrollHeight;
}

function clearLog() {
  document.getElementById('termBody').innerHTML = '';
}

// ─── Panel Navigation ─────────────────────────────────────────────────────────
function showPanel(name) {
  document.querySelectorAll('.panel').forEach(p => {
    p.classList.remove('active');
    p.classList.add('hidden');
  });
  document.querySelectorAll('.pipeline-step').forEach(b => b.classList.remove('active'));

  const panel = document.getElementById(`panel-${name}`);
  panel.classList.remove('hidden');
  panel.classList.add('active');

  const btn = document.querySelector(`[data-step="${name}"]`);
  if (btn) btn.classList.add('active');

  STATE.activePanel = name;
  log('INFO', `Panel: ${name.toUpperCase()}`);

  if (name === 'visualize' && STATE.stages.predict) {
    loadAllCharts();
  }
}

function markStage(stage, done = true) {
  STATE.stages[stage] = done;
  const el = document.getElementById(`st-${stage}`);
  if (el) {
    el.textContent  = done ? '●' : '○';
    el.className    = 'step-status ' + (done ? 'done' : '');
  }
}

// ─── Toast Notifications ──────────────────────────────────────────────────────
let toastTimer = null;
function showToast(msg, type = 'info', duration = 4000) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className   = `toast ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = 'toast hidden'; }, duration);
}

// ─── API Helper ───────────────────────────────────────────────────────────────
async function apiCall(method, path, body = null, isFormData = false) {
  const opts = { method, headers: {} };

  if (body && !isFormData) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  } else if (body) {
    opts.body = body; // FormData
  }

  const res = await fetch(`${STATE.apiBase}${path}`, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || data.message || 'API error');
  return data;
}

// ─── SESSION ──────────────────────────────────────────────────────────────────
function updateSession(patch) {
  if (patch.jobId)      { STATE.jobId = patch.jobId; document.getElementById('sJobId').textContent = patch.jobId.slice(-12); }
  if (patch.dataset)    document.getElementById('sDataset').textContent   = patch.dataset;
  if (patch.features)   document.getElementById('sFeatures').textContent  = patch.features;
  if (patch.bestModel)  document.getElementById('sBestModel').textContent = patch.bestModel;
  if (patch.highRisk !== undefined) {
    const el = document.getElementById('sHighRisk');
    el.textContent = patch.highRisk;
    el.className   = `sv ${Number(patch.highRisk) > 0 ? 'risk-high' : ''}`;
  }
}

function resetSession() {
  if (!confirm('Reset session? All trained models will be cleared from memory.')) return;
  Object.assign(STATE, {
    jobId: null, schema: null, report: null, predictions: null, charts: {},
    stages: { upload:false, process:false, train:false, predict:false, visualize:false }
  });
  ['sJobId','sDataset','sFeatures','sBestModel','sHighRisk'].forEach(id => {
    document.getElementById(id).textContent = '—';
  });
  ['upload','process','train','predict','visualize'].forEach(markStage.bind(null, undefined, false));
  document.getElementById('schemaPreview').classList.add('hidden');
  document.getElementById('processResult').classList.add('hidden');
  document.getElementById('trainResult').classList.add('hidden');
  document.getElementById('predictResult').classList.add('hidden');
  document.getElementById('riskDashboard').classList.add('hidden');
  document.getElementById('cfgModelSelect').innerHTML = '<option value="">Best Model (Auto)</option>';
  showPanel('upload');
  setStatus('SESSION RESET', 'warning');
  log('WARN', 'Session reset. All in-memory artifacts cleared.');
  showToast('Session cleared.', 'info');
}

// ─── 01 UPLOAD ────────────────────────────────────────────────────────────────
function onDragOver(e)  { e.preventDefault(); document.getElementById('dropZone').classList.add('drag-over'); }
function onDragLeave()  { document.getElementById('dropZone').classList.remove('drag-over'); }
function onDrop(e)      { e.preventDefault(); onDragLeave(); handleFiles(e.dataTransfer.files); }
function handleFileSelect(e) { handleFiles(e.target.files); }

function handleFiles(files) {
  if (!files || !files.length) return;
  const file = files[0];
  const ext  = file.name.split('.').pop().toLowerCase();
  const allowed = ['csv','json','geojson','xlsx'];
  if (!allowed.includes(ext)) {
    showToast(`Unsupported format: .${ext}`, 'error');
    return;
  }
  uploadFile(file);
}

async function uploadFile(file) {
  setStatus('UPLOADING…', 'warning');
  log('INFO', `Uploading: ${file.name} (${(file.size/1024/1024).toFixed(2)} MB)`);

  const fd = new FormData();
  fd.append('file', file);

  try {
    const res = await apiCall('POST', '/upload/', fd, true);
    const d   = res.data;
    STATE.jobId  = d.job_id;
    STATE.schema = d.schema;

    updateSession({ jobId: d.job_id, dataset: file.name });

    // Build schema preview
    const shape = d.schema.shape;
    document.getElementById('prevShape').textContent = `${shape[0]} rows × ${shape[1]} cols`;

    // Info chips
    const missingPcts = Object.values(d.schema.missing_pct);
    const avgMissing  = (missingPcts.reduce((a,b)=>a+b,0)/missingPcts.length*100).toFixed(1);
    document.getElementById('infoGrid').innerHTML = [
      { l:'ROWS',     v: shape[0].toLocaleString() },
      { l:'COLUMNS',  v: shape[1] },
      { l:'MISSING%', v: avgMissing + '%' },
      { l:'FORMAT',   v: '.' + file.name.split('.').pop().toUpperCase() },
    ].map(c=>`<div class="info-chip"><span class="ic-label">${c.l}</span><span class="ic-value">${c.v}</span></div>`).join('');

    // Sample table
    renderTable('sampleTable', d.schema.sample, d.schema.columns);

    // Inferred metadata
    document.getElementById('inferTarget').textContent = d.inferred_target || 'Not detected';
    document.getElementById('inferLat').textContent    = d.inferred_lat    || 'Not detected';
    document.getElementById('inferLon').textContent    = d.inferred_lon    || 'Not detected';

    // Set target default for process panel
    if (d.inferred_target) document.getElementById('cfgTarget').value = d.inferred_target;

    // Warnings
    const warnEl = document.getElementById('uploadWarnings');
    warnEl.innerHTML = (d.warnings || [])
      .map(w=>`<div class="warning-item">⚠ ${escHtml(w)}</div>`).join('');

    document.getElementById('schemaPreview').classList.remove('hidden');
    markStage('upload');
    setStatus('FILE LOADED', 'ok');
    log('SUCCESS', `File loaded: ${file.name} — ${shape[0]}R × ${shape[1]}C`);
    showToast('Dataset loaded successfully.', 'success');

  } catch(err) {
    setStatus('UPLOAD ERROR', 'error');
    log('ERROR', err.message);
    showToast(err.message, 'error');
  }
}

function proceedToProcess() {
  if (!STATE.jobId) { showToast('Upload a file first.', 'error'); return; }
  showPanel('process');
}

// ─── 02 PROCESS ───────────────────────────────────────────────────────────────
async function runProcess() {
  if (!STATE.jobId) { showToast('Upload a file first.', 'error'); return; }
  const target = document.getElementById('cfgTarget').value.trim() || null;
  const scaling= document.getElementById('cfgScaler').value;

  const btn    = document.getElementById('btnProcess');
  btn.disabled = true;
  btn.innerHTML= '<span class="loader"></span> PROCESSING…';
  setStatus('PREPROCESSING…', 'warning');
  log('INFO', `Starting preprocessing — target: ${target || 'auto'}, scaler: ${scaling}`);

  try {
    const res = await apiCall('POST', '/process/', { job_id: STATE.jobId, target_col: target, scaling });
    const d   = res.data;

    updateSession({ features: d.feature_names.length });

    // Result display
    const resultEl = document.getElementById('processResult');
    resultEl.className = 'result-block success';
    resultEl.innerHTML =
      `<div class="section-label" style="margin-bottom:.5rem">PREPROCESSING REPORT</div>` +
      `<div class="result-grid">` +
      mkStat('TRAIN ROWS',   d.splits.train) +
      mkStat('VAL ROWS',     d.splits.val) +
      mkStat('TEST ROWS',    d.splits.test) +
      mkStat('FEATURES',     d.feature_names.length) +
      mkStat('MISSING%',     (d.stats.missing_pct).toFixed(2) + '%') +
      (d.stats.class_distribution ? mkStat('CLASSES', Object.keys(d.stats.class_distribution).join(' / ')) : '') +
      `</div>` +
      (d.warnings.length ? `<div class="warnings-block mt-1">${d.warnings.map(w=>`<div class="warning-item">⚠ ${escHtml(w)}</div>`).join('')}</div>` : '') +
      `<button class="btn-primary mt-2" onclick="showPanel('train')">PROCEED TO TRAINING →</button>`;
    resultEl.classList.remove('hidden');

    markStage('process');
    setStatus('PREPROCESSING DONE', 'ok');
    log('SUCCESS', `Preprocessing complete. ${d.feature_names.length} features extracted.`);
    showToast('Preprocessing complete.', 'success');

  } catch(err) {
    const resultEl = document.getElementById('processResult');
    resultEl.className = 'result-block error';
    resultEl.innerHTML = `<span class="text-red">ERROR: ${escHtml(err.message)}</span>`;
    resultEl.classList.remove('hidden');
    setStatus('PROCESS ERROR', 'error');
    log('ERROR', err.message);
    showToast(err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML= '<span class="btn-icon">▶</span> RUN PREPROCESSING';
  }
}

// ─── 03 TRAIN ─────────────────────────────────────────────────────────────────
async function runTrain() {
  if (!STATE.jobId) { showToast('Complete preprocessing first.', 'error'); return; }

  const selected = [...document.querySelectorAll('.model-check input:checked')].map(c=>c.value);
  if (!selected.length) { showToast('Select at least one model.', 'error'); return; }

  const btn = document.getElementById('btnTrain');
  btn.disabled = true;
  btn.innerHTML= '<span class="loader"></span> TRAINING…';

  const progress = document.getElementById('trainProgress');
  const fill     = document.getElementById('progressFill');
  const label    = document.getElementById('progressLabel');
  progress.classList.remove('hidden');

  setStatus('TRAINING MODELS…', 'warning');
  log('INFO', `Training: [${selected.join(', ')}]`);

  // Animated progress (approximation — real ML is async blocking on server)
  const stages = [
    [15,  'Initializing training pipeline…'],
    [30,  'Fitting Random Forest…'],
    [55,  'Fitting XGBoost…'],
    [72,  'Fitting Logistic Regression…'],
    [85,  'Running cross-validation…'],
    [94,  'Fitting anomaly detector…'],
    [98,  'Persisting model artifacts…'],
  ];
  let si = 0;
  const progInterval = setInterval(() => {
    if (si < stages.length) {
      const [pct, msg] = stages[si++];
      fill.style.width  = pct + '%';
      label.textContent = msg;
      log('INFO', msg);
    }
  }, 900);

  try {
    const res = await apiCall('POST', '/train/', { job_id: STATE.jobId, models: selected });
    clearInterval(progInterval);
    fill.style.width  = '100%';
    label.textContent = 'Training complete.';

    STATE.report = res.data;
    const best   = res.data.best_model;
    updateSession({ bestModel: best || 'N/A' });

    // Populate model select for predict
    const sel = document.getElementById('cfgModelSelect');
    sel.innerHTML = '<option value="">Best Model (Auto)</option>';
    res.data.models_trained.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
      sel.appendChild(opt);
    });

    // Build metrics table
    const tableRows = Object.entries(res.data.results)
      .filter(([,r]) => r.validation)
      .map(([name, r]) => {
        const v   = r.validation;
        const isBest = name === best;
        const scoreClass = v.f1 >= 0.8 ? 'score-cell' : v.f1 >= 0.6 ? 'score-cell medium' : 'score-cell low';
        return `<tr class="${isBest ? 'best-row' : ''}">
          <td>${name.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}</td>
          <td class="${scoreClass}">${(v.accuracy*100).toFixed(1)}%</td>
          <td class="${scoreClass}">${(v.f1*100).toFixed(1)}%</td>
          <td>${v.roc_auc ? (v.roc_auc*100).toFixed(1)+'%' : '—'}</td>
          <td>${(r.validation.precision*100).toFixed(1)}%</td>
          <td>${(r.validation.recall*100).toFixed(1)}%</td>
          <td>${r.train_time_s}s</td>
        </tr>`;
      }).join('');

    const resultEl = document.getElementById('trainResult');
    resultEl.className = 'result-block success';
    resultEl.innerHTML =
      `<div class="section-label" style="margin-bottom:.5rem">MODEL PERFORMANCE (VALIDATION SET)</div>` +
      `<table class="metrics-table">` +
        `<thead><tr><th>MODEL</th><th>ACCURACY</th><th>F1</th><th>ROC-AUC</th><th>PRECISION</th><th>RECALL</th><th>TIME</th></tr></thead>` +
        `<tbody>${tableRows}</tbody>` +
      `</table>` +
      `<p style="margin-top:.75rem;font-family:var(--font-mono);font-size:.7rem;color:var(--cyan-dim)">★ = BEST MODEL</p>` +
      `<button class="btn-primary mt-2" onclick="showPanel('predict')">PROCEED TO PREDICTION →</button>`;
    resultEl.classList.remove('hidden');

    markStage('train');
    setStatus('TRAINING COMPLETE', 'ok');
    log('SUCCESS', `Training complete. Best model: ${best}`);
    showToast(`Training complete. Best: ${best}`, 'success');

  } catch(err) {
    clearInterval(progInterval);
    const resultEl = document.getElementById('trainResult');
    resultEl.className = 'result-block error';
    resultEl.innerHTML = `<span class="text-red">ERROR: ${escHtml(err.message)}</span>`;
    resultEl.classList.remove('hidden');
    setStatus('TRAINING ERROR', 'error');
    log('ERROR', err.message);
    showToast(err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML= '<span class="btn-icon">▶</span> BEGIN TRAINING';
  }
}

// ─── 04 PREDICT ───────────────────────────────────────────────────────────────
async function runPredict() {
  if (!STATE.jobId) { showToast('Complete training first.', 'error'); return; }

  const modelName = document.getElementById('cfgModelSelect').value || null;
  const btn = document.getElementById('btnPredict');
  btn.disabled = true;
  btn.innerHTML= '<span class="loader"></span> SCORING…';
  setStatus('COMPUTING RISK…', 'warning');
  log('INFO', `Running prediction — model: ${modelName || 'best'}`);

  try {
    const res = await apiCall('POST', '/predict/', { job_id: STATE.jobId, model_name: modelName });
    const d   = res.data;
    STATE.predictions = d;

    const s = d.risk_summary;
    updateSession({ highRisk: s.high_risk_count });

    const resultEl = document.getElementById('predictResult');
    resultEl.className = 'result-block success';
    resultEl.innerHTML =
      `<div class="section-label" style="margin-bottom:.5rem">PREDICTION SUMMARY</div>` +
      `<div class="result-grid">` +
      mkStat('SAMPLES',    d.n_samples) +
      mkStat('MODEL',      d.model_used.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())) +
      mkStat('MEAN RISK',  s.mean_risk.toFixed(3)) +
      mkStat('MAX RISK',   s.max_risk.toFixed(3)) +
      `</div>`;
    resultEl.classList.remove('hidden');

    // Risk dashboard cards
    const cards = document.getElementById('riskCards');
    cards.innerHTML =
      riskCard('HIGH RISK',   s.level_pct?.high  ?? 0,  s.level_counts?.high   ?? 0, 'high-card',  '%') +
      riskCard('MEDIUM RISK', s.level_pct?.medium ?? 0, s.level_counts?.medium  ?? 0, 'med-card',   '%') +
      riskCard('LOW RISK',    s.level_pct?.low    ?? 0, s.level_counts?.low     ?? 0, 'low-card',   '%') +
      riskCard('ANOMALIES',   d.anomaly_flags.filter(Boolean).length, '', '',  'flags') +
      riskCard('MEAN SCORE',  s.mean_risk.toFixed(3),   '', '', '') +
      riskCard('STD SCORE',   s.std_risk.toFixed(3),    '', '', '');
    document.getElementById('riskDashboard').classList.remove('hidden');

    markStage('predict');
    setStatus('PREDICTION COMPLETE', 'ok');
    log('SUCCESS', `Prediction done. High-risk: ${s.high_risk_count} (${(s.level_pct?.high ?? 0).toFixed(1)}%)`);
    showToast('Risk prediction complete.', 'success');

    // Auto-proceed
    setTimeout(() => showPanel('visualize'), 600);

  } catch(err) {
    const resultEl = document.getElementById('predictResult');
    resultEl.className = 'result-block error';
    resultEl.innerHTML = `<span class="text-red">ERROR: ${escHtml(err.message)}</span>`;
    resultEl.classList.remove('hidden');
    setStatus('PREDICT ERROR', 'error');
    log('ERROR', err.message);
    showToast(err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML= '<span class="btn-icon">▶</span> RUN PREDICTION';
  }
}

// ─── 05 VISUALIZE ─────────────────────────────────────────────────────────────
const VIZ_MAP = {
  'risk-dist':  { endpoint: '/visualize/risk-distribution', container: 'chart-risk-dist' },
  'model-cmp':  { endpoint: '/visualize/model-comparison',  container: 'chart-model-cmp' },
  'feat-imp':   { endpoint: '/visualize/feature-importance', container: 'chart-feat-imp' },
  'conf-mat':   { endpoint: '/visualize/confusion-matrix',   container: 'chart-conf-mat' },
  'anom-tl':    { endpoint: '/visualize/anomaly-timeline',   container: 'chart-anom-tl' },
  'geo-map':    { endpoint: '/visualize/geospatial',         container: 'chart-geo-map' },
};

function switchVizTab(btn, tabId) {
  document.querySelectorAll('.viz-tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.chart-container').forEach(c => c.classList.remove('active-chart'));
  btn.classList.add('active');
  document.getElementById(VIZ_MAP[tabId].container).classList.add('active-chart');

  if (!STATE.charts[tabId] && STATE.stages.predict) {
    loadChart(tabId);
  }
}

async function loadChart(tabId) {
  if (!STATE.jobId) return;
  const { endpoint, container } = VIZ_MAP[tabId];
  const el = document.getElementById(container);
  el.innerHTML = `<div style="padding:2rem;text-align:center;font-family:var(--font-mono);font-size:.75rem;color:var(--cyan-dim)"><span class="loader"></span> Loading chart…</div>`;

  try {
    const res = await apiCall('GET', `${endpoint}?job_id=${STATE.jobId}`);
    const fig  = res.data?.figure;
    if (!fig || !fig.data) throw new Error('No figure data returned.');

    el.innerHTML = '';
    Plotly.newPlot(container, fig.data, fig.layout, {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['select2d','lasso2d'],
      toImageButtonOptions: { format:'png', filename:`guardian_${tabId}`, width:1400, height:700, scale:2 },
    });
    STATE.charts[tabId] = true;
    log('SUCCESS', `Chart rendered: ${tabId}`);

  } catch(err) {
    el.innerHTML = `<div style="padding:2rem;font-family:var(--font-mono);font-size:.75rem;color:var(--red)">⚠ ${escHtml(err.message)}</div>`;
    log('WARN', `Chart load failed (${tabId}): ${err.message}`);
  }
}

async function loadAllCharts() {
  if (!STATE.jobId || !STATE.stages.predict) {
    showToast('Complete prediction pipeline first.', 'error'); return;
  }
  markStage('visualize');
  log('INFO', 'Loading all visualization charts…');
  setStatus('RENDERING CHARTS…', 'warning');

  for (const tabId of Object.keys(VIZ_MAP)) {
    await loadChart(tabId);
  }

  setStatus('DASHBOARD READY', 'ok');
  showToast('All charts loaded.', 'success');
}

// ─── Render Helpers ───────────────────────────────────────────────────────────
function renderTable(containerId, rows, cols) {
  if (!rows || !rows.length) return;
  const thead = `<tr>${cols.map(c=>`<th>${escHtml(c)}</th>`).join('')}</tr>`;
  const tbody = rows.map(r =>
    `<tr>${cols.map(c=>`<td>${escHtml(String(r[c] ?? ''))}</td>`).join('')}</tr>`
  ).join('');
  document.getElementById(containerId).innerHTML =
    `<table class="schema-table"><thead>${thead}</thead><tbody>${tbody}</tbody></table>`;
}

function mkStat(label, value) {
  return `<div class="result-stat">
    <span class="rs-label">${label}</span>
    <span class="rs-value">${value}</span>
  </div>`;
}

function riskCard(label, value, sub, cls, unit) {
  return `<div class="risk-card ${cls}">
    <span class="rc-label">${label}</span>
    <span class="rc-value">${value}${unit==='%' ? '%' : ''}</span>
    <span class="rc-sub">${sub !== '' && sub !== undefined ? sub + ' samples' : ''}</span>
  </div>`;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}
