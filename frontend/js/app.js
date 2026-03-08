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
  const ids = ['step-ingest', 'step-eda', 'step-viz'];
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
    setTimeout(() => setLoaderStep(2, 'Building visualisations…'), 2200);

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
