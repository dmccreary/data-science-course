// Plotly Interactive Features Playground
// CANVAS_HEIGHT: 575
// Bloom L3 (Apply): students practice the interactive features that every Plotly chart has:
// hover, zoom, pan, box select, lasso select, reset, download, and legend toggling.
// This is a real Plotly.js chart (main.html loads Plotly, not p5.js). The sim listens to the
// chart's own events to tick off each feature the first time the student uses it, and checks
// four challenge tasks the same way. Nothing is simulated: what works here works on any
// chart made with plotly.express.

(function () {
  const FEATURES = [
    { id: 'hover', name: 'Hover', how: 'Move the mouse over a point to read its tooltip.' },
    { id: 'zoom', name: 'Zoom', how: 'Drag a box on the chart to zoom into that region.' },
    { id: 'pan', name: 'Pan', how: 'Hold Shift and drag, or pick Pan in the toolbar, to slide the view.' },
    { id: 'box', name: 'Box select', how: 'Pick Box Select in the toolbar, then drag a box around some points.' },
    { id: 'lasso', name: 'Lasso select', how: 'Pick Lasso Select in the toolbar, then draw a loop around some points.' },
    { id: 'reset', name: 'Reset', how: 'Double-click the chart to bring back the full view.' },
    { id: 'download', name: 'Download', how: 'Click the camera in the toolbar to save the chart as a PNG.' },
    { id: 'legend', name: 'Legend', how: 'Click a group in the legend to hide it. Click again to show it.' }
  ];
  const CHALLENGES = [
    { id: 'cluster', text: 'Zoom into the cluster in the upper right.' },
    { id: 'groupA', text: 'Select all of group A and nothing else.' },
    { id: 'outlier', text: 'Find the outlier with the highest y value.' },
    { id: 'png', text: 'Download the chart as a PNG.' }
  ];

  let used = {};            // features the student has used
  let solved = {};          // challenges the student has completed
  let lastSpan = null;      // axis spans before the latest change, to tell a pan from a zoom
  let chart, featureList, challengeList, progressLine, hintLine;
  let outlierIndex = 0;
  let groupASize = 0;

  // ---------- sample data: three groups, one cluster, one outlier ----------
  // A small seeded generator keeps the points the same on every load.
  let seed = 20260;
  function rand() {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  }
  function gauss(mean, sd) {
    return mean + sd * (rand() + rand() + rand() + rand() - 2) * 1.73;   // roughly normal
  }
  function makeGroup(name, n, cx, cy, sx, sy, color) {
    const x = [], y = [];
    for (let i = 0; i < n; i++) {
      x.push(Math.round(gauss(cx, sx) * 10) / 10);
      y.push(Math.round(gauss(cy, sy) * 10) / 10);
    }
    return { name, x, y, type: 'scatter', mode: 'markers', marker: { size: 10, color, line: { color: 'white', width: 1 } },
      hovertemplate: 'group=' + name + '<br>x=%{x}<br>y=%{y}<extra></extra>' };
  }
  function makeData() {
    seed = 20260;
    const a = makeGroup('A', 20, 22, 30, 6, 8, 'royalblue');
    const b = makeGroup('B', 20, 50, 52, 9, 9, 'darkorange');
    const c = makeGroup('C', 20, 82, 80, 4, 5, 'seagreen');
    // one point far above everything else
    b.x.push(40);
    b.y.push(118);
    outlierIndex = b.x.length - 1;
    groupASize = a.x.length;
    return [a, b, c];
  }

  // ---------- page ----------
  function build() {
    const main = document.querySelector('main');
    const style = document.createElement('style');
    style.textContent =
      'main { display: block; background: aliceblue; border: 1px solid silver; box-sizing: border-box; padding: 6px 10px 8px; }' +
      '.pf-title { text-align: center; font-size: 22px; margin: 2px 0 4px; }' +
      '.pf-chart { background: white; border: 1px solid silver; border-radius: 8px; }' +
      '.pf-panel { background: white; border: 1px solid silver; border-radius: 8px; margin-top: 6px; padding: 6px 10px; font-size: 15px; }' +
      '.pf-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 2px 10px; }' +
      '.pf-item { cursor: pointer; padding: 2px 4px; border-radius: 4px; white-space: nowrap; }' +
      '.pf-item:hover { background: lightyellow; }' +
      '.pf-done { color: seagreen; font-weight: bold; }' +
      '.pf-progress { font-weight: bold; margin-top: 4px; }' +
      '.pf-hint { color: dimgray; min-height: 19px; }' +
      '.pf-challenges { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1px 12px; margin-top: 4px; font-size: 14px; }' +
      '.pf-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; }' +
      '@media (max-width: 600px) { .pf-title { font-size: 17px; } .pf-panel { font-size: 12px; } ' +
      '.pf-grid { grid-template-columns: repeat(2, 1fr); } .pf-challenges { grid-template-columns: 1fr; font-size: 12px; } }';
    document.head.appendChild(style);

    const el = (tag, cls, text) => {
      const node = document.createElement(tag);
      if (cls) node.className = cls;
      if (text) node.textContent = text;
      return node;
    };
    main.appendChild(el('div', 'pf-title', 'Explore an Interactive Plotly Chart'));
    chart = el('div', 'pf-chart');
    main.appendChild(chart);

    const panel = el('div', 'pf-panel');
    featureList = el('div', 'pf-grid');
    panel.appendChild(featureList);
    const row = el('div', 'pf-row');
    progressLine = el('div', 'pf-progress');
    const restart = el('button', '', 'Restart');
    restart.addEventListener('click', () => { used = {}; solved = {}; drawChart(); render(); });
    row.appendChild(progressLine);
    row.appendChild(restart);
    panel.appendChild(row);
    hintLine = el('div', 'pf-hint');
    panel.appendChild(hintLine);
    challengeList = el('div', 'pf-challenges');
    panel.appendChild(challengeList);
    main.appendChild(panel);

    drawChart();
    render();
    window.addEventListener('resize', () => {
      chart.style.height = chartHeight() + 'px';
      Plotly.Plots.resize(chart);
    });
  }

  function chartHeight() {
    return window.innerWidth < 600 ? 290 : 372;
  }

  function drawChart() {
    chart.style.height = chartHeight() + 'px';
    const layout = {
      margin: { l: 50, r: 15, t: 34, b: 42 },
      dragmode: 'zoom',
      hovermode: 'closest',
      xaxis: { title: { text: 'x' }, zeroline: false },
      yaxis: { title: { text: 'y' }, zeroline: false },
      legend: { orientation: 'h', x: 0, y: 1.12, title: { text: 'group  ' } }
    };
    const config = {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      scrollZoom: false,          // never take over page scrolling inside the iframe
      modeBarButtonsToRemove: ['toImage'],
      // the camera button is replaced by one that also reports the download to the tracker
      modeBarButtonsToAdd: [{
        name: 'Download plot as a PNG',
        icon: Plotly.Icons.camera,
        click: gd => {
          mark('download');
          solve('png');
          Plotly.downloadImage(gd, { format: 'png', filename: 'plotly-interactive-features', width: 900, height: 500 });
        }
      }]
    };
    Plotly.newPlot(chart, makeData(), layout, config).then(() => {
      lastSpan = spans();
      listen();
    });
  }

  function spans() {
    const x = chart._fullLayout.xaxis.range, y = chart._fullLayout.yaxis.range;
    return { x: x[1] - x[0], y: y[1] - y[0], x0: x[0], y0: y[0] };
  }

  // ---------- Plotly events drive the tracker ----------
  function listen() {
    chart.removeAllListeners && chart.removeAllListeners();
    chart.on('plotly_hover', ev => {
      mark('hover');
      const p = ev.points[0];
      if (p.curveNumber === 1 && p.pointNumber === outlierIndex) solve('outlier');
    });
    chart.on('plotly_legendclick', () => { mark('legend'); });
    chart.on('plotly_doubleclick', () => { mark('reset'); });
    chart.on('plotly_selected', ev => {
      if (!ev || !ev.points) return;
      if (ev.lassoPoints) mark('lasso');
      else if (ev.range) mark('box');
      const onlyA = ev.points.length === groupASize && ev.points.every(p => p.curveNumber === 0);
      if (onlyA) solve('groupA');
    });
    chart.on('plotly_relayout', ev => {
      if (ev['xaxis.autorange'] || ev['yaxis.autorange']) {
        mark('reset');
      } else if ('xaxis.range[0]' in ev || 'yaxis.range[0]' in ev) {
        // a pan slides the view and keeps its size; a zoom changes the size
        const now = spans();
        const same = Math.abs(now.x - lastSpan.x) < 1e-6 * Math.abs(lastSpan.x) + 1e-9 &&
          Math.abs(now.y - lastSpan.y) < 1e-6 * Math.abs(lastSpan.y) + 1e-9;
        mark(same ? 'pan' : 'zoom');
        // group C sits near (82, 80): the view must show only the upper right corner
        if (!same && now.x0 >= 55 && now.y0 >= 55 && now.x < 60 && now.y < 70) solve('cluster');
      }
      lastSpan = spans();
    });
  }

  function mark(id) {
    if (used[id]) return;
    used[id] = true;
    render();
  }

  function solve(id) {
    if (solved[id]) return;
    solved[id] = true;
    render();
  }

  // ---------- tracker display ----------
  function render() {
    featureList.innerHTML = '';
    for (const f of FEATURES) {
      const item = document.createElement('div');
      item.className = 'pf-item' + (used[f.id] ? ' pf-done' : '');
      item.textContent = (used[f.id] ? '☑ ' : '☐ ') + f.name;
      item.title = f.how;
      item.addEventListener('click', () => { hintLine.textContent = f.name + ': ' + f.how; });
      featureList.appendChild(item);
    }
    const count = FEATURES.filter(f => used[f.id]).length;
    progressLine.textContent = count === FEATURES.length
      ? 'You have explored all 8 interactive features!'
      : 'You have explored ' + count + ' of 8 interactive features.';
    progressLine.style.color = count === FEATURES.length ? 'seagreen' : 'black';
    const next = FEATURES.find(f => !used[f.id]);
    hintLine.textContent = next ? 'Try next: ' + next.name + '. ' + next.how
      : 'Every chart made with plotly.express has these same features.';

    challengeList.innerHTML = '';
    for (const c of CHALLENGES) {
      const item = document.createElement('div');
      if (solved[c.id]) item.className = 'pf-done';
      item.textContent = (solved[c.id] ? '☑ ' : '☐ ') + 'Challenge: ' + c.text;
      challengeList.appendChild(item);
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
  else build();
})();
