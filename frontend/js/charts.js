/* ═══════════════════════════════════════════════════════════════════════
   charts.js — Renders Plotly charts inside the dashboard grid.
   ═══════════════════════════════════════════════════════════════════════ */

const CHART_COLORS = {
  bg:       'rgba(0,0,0,0)',
  paper:    'rgba(0,0,0,0)',
  grid:     'rgba(255,255,255,0.05)',
  font:     '#94a3b8',
  accent:   'rgba(99,102,241,0.8)',
  accent2:  'rgba(168,85,247,0.8)',
  accent3:  'rgba(236,72,153,0.8)',
  cyan:     'rgba(34,211,238,0.8)',
  green:    'rgba(74,222,128,0.8)',
  amber:    'rgba(250,204,21,0.8)',
  red:      'rgba(244,63,94,0.8)',
};

const COLOR_CYCLE = [
  CHART_COLORS.accent, CHART_COLORS.accent2, CHART_COLORS.accent3,
  CHART_COLORS.cyan,   CHART_COLORS.green,   CHART_COLORS.amber,
  CHART_COLORS.red,    'rgba(148,163,184,0.7)',
];

/**
 * Build the shared Plotly layout used by every chart, merged with
 * per-chart overrides.
 */
function baseLayout(overrides = {}) {
  const base = {
    paper_bgcolor: CHART_COLORS.paper,
    plot_bgcolor:  CHART_COLORS.bg,
    font: { family: "'Inter', sans-serif", size: 12, color: CHART_COLORS.font },
    margin: { t: 12, r: 16, b: 44, l: 52 },
    xaxis: {
      gridcolor: CHART_COLORS.grid,
      zerolinecolor: CHART_COLORS.grid,
      linecolor: CHART_COLORS.grid,
    },
    yaxis: {
      gridcolor: CHART_COLORS.grid,
      zerolinecolor: CHART_COLORS.grid,
      linecolor: CHART_COLORS.grid,
    },
    coloraxis: { colorbar: { outlinewidth: 0 } },
    hoverlabel: {
      bgcolor: '#1e1b4b',
      bordercolor: 'rgba(99,102,241,0.4)',
      font: { family: "'Inter', sans-serif", size: 12, color: '#e2e8f0' },
    },
    showlegend: false,
    autosize: true,
  };

  // Deep merge overrides.xaxis / yaxis
  if (overrides.xaxis) {
    base.xaxis = { ...base.xaxis, ...overrides.xaxis };
    delete overrides.xaxis;
  }
  if (overrides.yaxis) {
    base.yaxis = { ...base.yaxis, ...overrides.yaxis };
    delete overrides.yaxis;
  }

  return { ...base, ...overrides };
}

const PLOTLY_CONFIG = {
  displayModeBar: false,
  responsive: true,
};

/**
 * Render a single chart object into a target container.
 * @param {HTMLElement} container  — the .chart-card-body element
 * @param {object} chart           — chart descriptor from the API
 */
function renderChart(container, chart) {
  const traces = chart.traces.map((t, i) => {
    const clone = { ...t };
    // Assign colour cycle to box / bar traces without explicit colour
    if ((clone.type === 'box' || clone.type === 'bar') && !clone.marker?.color) {
      clone.marker = { ...clone.marker, color: COLOR_CYCLE[i % COLOR_CYCLE.length] };
    }
    return clone;
  });

  const layout = baseLayout(chart.layout || {});

  // Heatmaps need square aspect & annotations
  if (chart.type === 'heatmap') {
    layout.margin = { t: 12, r: 12, b: 80, l: 80 };
  }

  // Show legend for multi-trace charts
  if (traces.length > 1 && chart.type !== 'heatmap') {
    layout.showlegend = true;
    layout.legend = {
      font: { size: 11, color: CHART_COLORS.font },
      bgcolor: 'rgba(0,0,0,0)',
      orientation: 'h',
      y: -0.2,
    };
  }

  // Pie special
  if (chart.type === 'pie') {
    layout.margin = { t: 0, r: 0, b: 0, l: 0 };
    layout.showlegend = true;
    layout.legend = {
      font: { size: 11, color: CHART_COLORS.font },
      bgcolor: 'rgba(0,0,0,0)',
    };
  }

  Plotly.newPlot(container, traces, layout, PLOTLY_CONFIG);
}


/**
 * Build the entire chart grid from the API response.
 * @param {HTMLElement} grid   — #charts-grid
 * @param {Array}       charts — array of chart descriptors
 */
function buildChartGrid(grid, charts) {
  grid.innerHTML = '';

  charts.forEach((chart, idx) => {
    const card = document.createElement('div');
    card.className = 'chart-card fade-up';
    card.style.animationDelay = `${Math.min(idx * 0.04, 0.6)}s`;
    card.dataset.chartType = chart.type;

    // Make heatmaps and grouped boxes full-width
    if (chart.type === 'heatmap' || chart.type === 'grouped_box') {
      card.classList.add('wide');
    }

    const header = document.createElement('div');
    header.className = 'chart-card-header';
    header.innerHTML = `
      <div class="chart-card-title">${chart.title}</div>
      ${chart.subtitle ? `<div class="chart-card-subtitle">${chart.subtitle}</div>` : ''}
    `;

    const body = document.createElement('div');
    body.className = 'chart-card-body';
    body.style.height = chart.type === 'pie' ? '320px' : '340px';

    card.appendChild(header);
    card.appendChild(body);
    grid.appendChild(card);

    // Render once the element is in the DOM
    requestAnimationFrame(() => renderChart(body, chart));
  });
}


/**
 * Populate the filter pills above the chart grid.
 */
function buildChartFilters(filterContainer, grid, charts) {
  const types = [...new Set(charts.map(c => c.type))];

  filterContainer.innerHTML = '';

  // "All" filter
  const allBtn = document.createElement('button');
  allBtn.className = 'chart-filter active';
  allBtn.textContent = 'All';
  allBtn.onclick = () => {
    filterContainer.querySelectorAll('.chart-filter').forEach(b => b.classList.remove('active'));
    allBtn.classList.add('active');
    grid.querySelectorAll('.chart-card').forEach(c => c.style.display = '');
  };
  filterContainer.appendChild(allBtn);

  const labels = {
    histogram: 'Histograms',
    box: 'Box Plots',
    heatmap: 'Heatmap',
    bar: 'Bar Charts',
    scatter: 'Scatter',
    grouped_box: 'Grouped Box',
    pie: 'Pie Charts',
    line: 'Time Series',
  };

  types.forEach(t => {
    const btn = document.createElement('button');
    btn.className = 'chart-filter';
    btn.textContent = labels[t] || t;
    btn.onclick = () => {
      filterContainer.querySelectorAll('.chart-filter').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      grid.querySelectorAll('.chart-card').forEach(c => {
        c.style.display = c.dataset.chartType === t ? '' : 'none';
      });
    };
    filterContainer.appendChild(btn);
  });
}
