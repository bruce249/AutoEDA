/* ═══════════════════════════════════════════════════════════════════════
   app.js — Upload handling, API call, dashboard population
   ═══════════════════════════════════════════════════════════════════════ */

// ── DOM refs ────────────────────────────────────────────────────────────
const $landing    = document.getElementById('landing');
const $loading    = document.getElementById('loading');
const $dashboard  = document.getElementById('dashboard');
const $uploadZone = document.getElementById('upload-zone');
const $fileInput  = document.getElementById('file-input');

// ── View management ─────────────────────────────────────────────────────
function showLanding() {
  $landing.classList.remove('hidden');
  $loading.classList.add('hidden');
  $dashboard.classList.add('hidden');
  document.getElementById('nav-home').classList.add('active');
  document.getElementById('nav-dash').classList.remove('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
function showLoading() {
  $landing.classList.add('hidden');
  $loading.classList.remove('hidden');
  $dashboard.classList.add('hidden');
}
function showDashboard() {
  $landing.classList.add('hidden');
  $loading.classList.add('hidden');
  $dashboard.classList.remove('hidden');
  document.getElementById('nav-home').classList.remove('active');
  document.getElementById('nav-dash').classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Loader steps ────────────────────────────────────────────────────────
function setLoaderStep(stepNum, text) {
  const ids = ['step-ingest', 'step-eda', 'step-model', 'step-viz'];
  ids.forEach((id, i) => {
    const el = document.getElementById(id);
    el.classList.remove('active', 'done');
    if (i < stepNum) el.classList.add('done');
    else if (i === stepNum) el.classList.add('active');
  });
  document.getElementById('loader-status').textContent = text;
}

// ── Upload handling ─────────────────────────────────────────────────────

// Drag & drop
$uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  $uploadZone.classList.add('drag-over');
});
$uploadZone.addEventListener('dragleave', () => {
  $uploadZone.classList.remove('drag-over');
});
$uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  $uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
$uploadZone.addEventListener('click', () => $fileInput.click());
$fileInput.addEventListener('change', () => {
  if ($fileInput.files.length) handleFile($fileInput.files[0]);
});

async function handleFile(file) {
  showLoading();
  setLoaderStep(0, 'Running Ingestion Agent…');

  const formData = new FormData();
  formData.append('file', file);

  try {
    // Simulate step progression
    setTimeout(() => setLoaderStep(1, 'Running EDA Agent…'), 800);
    setTimeout(() => setLoaderStep(2, 'Running Modeling Agent…'), 2200);
    setTimeout(() => setLoaderStep(3, 'Building visualisations…'), 3800);

    const resp = await fetch('/api/analyze', { method: 'POST', body: formData });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || 'Analysis failed');
    }

    const data = await resp.json();
    populateDashboard(data);
    showDashboard();

  } catch (err) {
    alert('Error: ' + err.message);
    showLanding();
  }
}

// ── Populate dashboard ──────────────────────────────────────────────────

function populateDashboard(data) {
  const meta = data.metadata || {};
  const schema = data.schema || {};

  // Header
  document.getElementById('dash-title').textContent = data.filename || 'Dataset Analysis';
  document.getElementById('dash-sub').textContent =
    `${meta.row_count?.toLocaleString() ?? '?'} rows · ${meta.column_count ?? '?'} columns · source: ${meta.source_type ?? 'unknown'}`;

  // KPIs
  buildKPIs(meta, schema);

  // Summary
  document.getElementById('summary-text').textContent = data.eda_summary || '';

  // Insights
  buildInsights(data.insights || []);

  // Preview table
  buildPreviewTable(data.preview || { columns: [], rows: [] });

  // Segments
  buildSegments(data.segments || []);

  // Charts
  const grid    = document.getElementById('charts-grid');
  const filters = document.getElementById('chart-filters');
  buildChartGrid(grid, data.charts || []);
  buildChartFilters(filters, grid, data.charts || []);

  // Modeling results
  buildModelingSection(data.modeling || {}, data.modeling_charts || []);

  // Show chat FAB now that analysis is ready
  initChat();
}


// ── KPIs ────────────────────────────────────────────────────────────────
function buildKPIs(meta, schema) {
  const row = document.getElementById('kpi-row');
  row.innerHTML = '';

  const kpis = [
    {
      label: 'Rows',
      value: meta.row_count?.toLocaleString() ?? '—',
      extra: null,
    },
    {
      label: 'Columns',
      value: meta.column_count ?? '—',
      extra: `${schema.numeric_columns?.length ?? 0} numeric · ${schema.categorical_columns?.length ?? 0} categorical`,
    },
    {
      label: 'Missing Cells',
      value: meta.total_missing_cells?.toLocaleString() ?? '0',
      extra: meta.total_missing_cells
        ? `${((meta.total_missing_cells / (meta.row_count * meta.column_count)) * 100).toFixed(1)}% of dataset`
        : 'No missing data',
    },
    {
      label: 'Duplicate Rows',
      value: meta.duplicate_row_count?.toLocaleString() ?? '0',
      extra: null,
    },
    {
      label: 'Memory',
      value: `${meta.memory_usage_mb ?? '?'}`,
      unit: 'MB',
      extra: null,
    },
  ];

  kpis.forEach((k, i) => {
    const div = document.createElement('div');
    div.className = `kpi fade-up fade-up-d${i + 1}`;
    div.innerHTML = `
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value">${k.value}${k.unit ? `<span class="kpi-unit">${k.unit}</span>` : ''}</div>
      ${k.extra ? `<div class="kpi-extra">${k.extra}</div>` : ''}
    `;
    row.appendChild(div);
  });
}


// ── Insights ────────────────────────────────────────────────────────────
function buildInsights(insights) {
  const card = document.getElementById('insights-card');
  const list = document.getElementById('insights-list');
  list.innerHTML = '';

  if (!insights.length) {
    card.style.display = 'none';
    return;
  }
  card.style.display = '';

  insights.forEach(ins => {
    const div = document.createElement('div');
    div.className = 'insight';
    div.innerHTML = `
      <span class="insight-badge ${ins.category}">${ins.category.replace(/_/g, ' ')}</span>
      <span>${ins.detail}</span>
    `;
    list.appendChild(div);
  });
}


// ── Preview table ───────────────────────────────────────────────────────
function buildPreviewTable(preview) {
  const wrap = document.getElementById('preview-table');
  if (!preview.columns.length) { wrap.innerHTML = '<p>No preview available.</p>'; return; }

  let html = '<table class="data-table"><thead><tr>';
  preview.columns.forEach(c => { html += `<th>${c}</th>`; });
  html += '</tr></thead><tbody>';

  preview.rows.forEach(row => {
    html += '<tr>';
    preview.columns.forEach(c => {
      let v = row[c];
      if (v === null || v === undefined || v === '') v = '<span style="color:var(--text-muted)">null</span>';
      html += `<td>${v}</td>`;
    });
    html += '</tr>';
  });

  html += '</tbody></table>';
  wrap.innerHTML = html;
}


// ── Segments ────────────────────────────────────────────────────────────
function buildSegments(segments) {
  const card = document.getElementById('segments-card');
  const list = document.getElementById('segments-list');
  list.innerHTML = '';

  if (!segments.length) { card.style.display = 'none'; return; }
  card.style.display = '';

  segments.forEach(seg => {
    const type = seg.type || '';
    let iconClass = 'cat';
    let iconText = 'C';
    if (type === 'numeric_threshold') { iconClass = 'num'; iconText = 'N'; }
    if (type === 'outlier_segment')   { iconClass = 'out'; iconText = '!'; }

    const div = document.createElement('div');
    div.className = 'segment';
    div.innerHTML = `
      <div class="seg-icon ${iconClass}">${iconText}</div>
      <span>${seg.description || JSON.stringify(seg)}</span>
    `;
    list.appendChild(div);
  });
}


// ── Modeling Section ────────────────────────────────────────────────────

function buildModelingSection(modeling, modelingCharts) {
  const header = document.getElementById('modeling-header');
  const kpiRow = document.getElementById('model-kpi-row');
  const chartsGrid = document.getElementById('modeling-charts-grid');
  const reportCard = document.getElementById('model-report-card');
  const takeawaysCard = document.getElementById('model-takeaways-card');
  const recsCard = document.getElementById('model-recommendations-card');
  const pillsContainer = document.getElementById('model-pills');

  if (!modeling || !modeling.evaluation || !modeling.evaluation.best_model || modeling.evaluation.best_model === 'N/A') {
    [header, kpiRow, chartsGrid, reportCard, takeawaysCard, recsCard].forEach(el => { if(el) el.style.display = 'none'; });
    return;
  }

  header.style.display = '';

  // Problem type pills
  const problem = modeling.problem || {};
  pillsContainer.innerHTML = '';
  const typeLabelMap = {
    regression: { label: 'Regression', cls: 'pill-reg' },
    classification: { label: 'Classification', cls: 'pill-clf' },
    time_series: { label: 'Time Series', cls: 'pill-ts' },
    clustering: { label: 'Clustering', cls: 'pill-clust' },
  };
  const pt = typeLabelMap[problem.problem_type] || { label: problem.problem_type || '?', cls: '' };
  pillsContainer.innerHTML = `
    <span class="model-pill ${pt.cls}">${pt.label}</span>
    ${problem.target ? `<span class="model-pill pill-target">Target: ${problem.target}</span>` : ''}
  `;

  // Model KPIs
  const ev = modeling.evaluation || {};
  const bestMetrics = ev.best_metrics || {};
  const training = modeling.training || {};
  const fe = modeling.feature_engineering || {};

  kpiRow.style.display = '';
  kpiRow.innerHTML = '';

  const modelKpis = [
    { label: 'Best Model', value: ev.best_model || '—', extra: null },
    ...Object.entries(bestMetrics).map(([k, v]) => ({
      label: k,
      value: typeof v === 'number' ? v.toFixed(4) : v,
      extra: null,
    })),
    { label: 'Features', value: fe.n_features ?? '—', extra: `${fe.n_samples ?? '?'} samples` },
    { label: 'Train / Test', value: `${training.split_info?.train_rows ?? '?'} / ${training.split_info?.test_rows ?? '?'}`, extra: training.split_info?.method?.replace(/_/g, ' ') || null },
  ];

  modelKpis.forEach((k, i) => {
    const div = document.createElement('div');
    div.className = `kpi fade-up fade-up-d${Math.min(i + 1, 6)}`;
    div.innerHTML = `
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value">${k.value}</div>
      ${k.extra ? `<div class="kpi-extra">${k.extra}</div>` : ''}
    `;
    kpiRow.appendChild(div);
  });

  // Modeling Charts
  if (modelingCharts && modelingCharts.length) {
    chartsGrid.style.display = '';
    chartsGrid.innerHTML = '';
    modelingCharts.forEach((chart, idx) => {
      const card = document.createElement('div');
      card.className = 'chart-card fade-up';
      card.style.animationDelay = `${Math.min(idx * 0.06, 0.5)}s`;
      card.dataset.chartType = chart.type;
      if (['feature_importance', 'predictions', 'residuals'].includes(chart.type)) {
        card.classList.add('wide');
      }
      const hdr = document.createElement('div');
      hdr.className = 'chart-card-header';
      hdr.innerHTML = `
        <div class="chart-card-title">${chart.title}</div>
        ${chart.subtitle ? `<div class="chart-card-subtitle">${chart.subtitle}</div>` : ''}
      `;
      const body = document.createElement('div');
      body.className = 'chart-card-body';
      body.style.height = '360px';
      card.appendChild(hdr);
      card.appendChild(body);
      chartsGrid.appendChild(card);
      requestAnimationFrame(() => renderChart(body, chart));
    });
  } else {
    chartsGrid.style.display = 'none';
  }

  // Insight Report
  const insights = modeling.insights || {};
  if (insights.report_text) {
    reportCard.style.display = '';
    document.getElementById('model-report-text').innerHTML = insights.report_text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .split('\n\n').map(p => `<p>${p}</p>`).join('');
  } else { reportCard.style.display = 'none'; }

  // Key Takeaways
  const takeaways = insights.key_takeaways || [];
  if (takeaways.length) {
    takeawaysCard.style.display = '';
    const tList = document.getElementById('model-takeaways-list');
    tList.innerHTML = '';
    takeaways.forEach(t => {
      const div = document.createElement('div');
      div.className = 'takeaway';
      div.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg><span>${t}</span>`;
      tList.appendChild(div);
    });
  } else { takeawaysCard.style.display = 'none'; }

  // Recommendations
  const recs = insights.recommendations || [];
  if (recs.length) {
    recsCard.style.display = '';
    const rList = document.getElementById('model-recommendations-list');
    rList.innerHTML = '';
    recs.forEach(r => {
      const div = document.createElement('div');
      div.className = `recommendation priority-${r.priority || 'medium'}`;
      div.innerHTML = `
        <span class="rec-priority">${(r.priority || 'medium').toUpperCase()}</span>
        <div class="rec-body"><div class="rec-title">${r.title}</div><div class="rec-detail">${r.detail}</div></div>
      `;
      rList.appendChild(div);
    });
  } else { recsCard.style.display = 'none'; }
}


// ═══════════════════════════════════════════════════════════════════════
//  CHAT LOGIC
// ═══════════════════════════════════════════════════════════════════════

const $chatFab     = document.getElementById('chat-fab');
const $chatPanel   = document.getElementById('chat-panel');
const $chatMsgs    = document.getElementById('chat-messages');
const $chatForm    = document.getElementById('chat-form');
const $chatInput   = document.getElementById('chat-input');
const $chatSendBtn = document.getElementById('chat-send-btn');

let chatOpen = false;

/** Show the FAB after analysis completes */
function initChat() {
  $chatFab.classList.remove('hidden');
}

/** Toggle the chat panel open / closed */
function toggleChat() {
  chatOpen = !chatOpen;
  $chatPanel.classList.toggle('hidden', !chatOpen);
  $chatFab.classList.toggle('open', chatOpen);
  if (chatOpen) {
    $chatInput.focus();
    $chatMsgs.scrollTop = $chatMsgs.scrollHeight;
  }
}

/** Append a bubble to the chat panel */
function appendBubble(role, text) {
  const wrap = document.createElement('div');
  wrap.className = `chat-bubble ${role}`;
  wrap.innerHTML = `
    <div class="bubble-avatar">${role === 'user' ? 'You' : 'AI'}</div>
    <div class="bubble-body">${escapeHtml(text)}</div>
  `;
  $chatMsgs.appendChild(wrap);
  $chatMsgs.scrollTop = $chatMsgs.scrollHeight;
  return wrap;
}

/** Show / remove typing indicator */
function showTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'chat-bubble assistant';
  wrap.id = 'typing-indicator';
  wrap.innerHTML = `
    <div class="bubble-avatar">AI</div>
    <div class="typing-dots"><span></span><span></span><span></span></div>
  `;
  $chatMsgs.appendChild(wrap);
  $chatMsgs.scrollTop = $chatMsgs.scrollHeight;
}
function removeTyping() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

/** Send a message to /api/chat */
async function sendChat(e) {
  e.preventDefault();
  const text = $chatInput.value.trim();
  if (!text) return;

  appendBubble('user', text);
  $chatInput.value = '';
  $chatSendBtn.disabled = true;
  showTyping();

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    removeTyping();

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      appendBubble('assistant', '\u26a0\ufe0f ' + (err.detail || 'Something went wrong.'));
    } else {
      const data = await resp.json();
      appendBubble('assistant', data.answer || 'No response.');
    }
  } catch (err) {
    removeTyping();
    appendBubble('assistant', '\u26a0\ufe0f Network error \u2014 is the server running?');
  } finally {
    $chatSendBtn.disabled = false;
    $chatInput.focus();
  }
}

/** Basic HTML escape */
function escapeHtml(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}