// Mean, Median, and Mode Explorer
// CANVAS_HEIGHT: 540
// Bloom L2 (Understand): students drag data points and add outliers to a dot plot and explain
// why the mean, the median, and the mode respond differently to the same change.
//
// Model: values are whole multiples of 5 from 0 to 100, so equal values stack into a dot histogram.
//   mean    sum of the values divided by the number of values
//   median  middle value of the sorted data (the average of the two middle values when n is even)
//   mode    the value or values that occur most often (no mode when no value repeats)
// The four starting datasets come from a seeded generator, so they are the same on every visit.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 460;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const STEP = 5;             // spacing of the allowed values
const MAX_POINTS = 30;
const DIVERGE = 2;          // mean and median count as "apart" when they differ by this much
const DATASETS = ['Symmetric', 'Right-skewed', 'Left-skewed', 'Bimodal'];
const SEEDS = [5, 18, 18, 52];
const MEAN_COLOR = 'crimson', MEDIAN_COLOR = 'seagreen', MODE_COLOR = 'royalblue';

let datasetSelect, addButton, outlierButton, resetButton;
let start = [], values = [];
let dots = [];              // dots drawn in the last frame: { i, x, y }
let plot = { x0: 0, x1: 1 };
let dragIdx = -1, lastAdded = -1;

// Seeded data: a linear congruential generator, Box-Muller normals, and exponentials for the skewed sets
function makeData(k) {
  let seed = SEEDS[k];
  const uniform = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return (seed + 0.5) / 4294967296; };
  const normal = (m, sd) => m + sd * Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
  const expo = m => -m * Math.log(uniform());
  const make = [() => normal(50, 10), () => 20 + expo(15), () => 80 - expo(15), i => normal(i < 10 ? 30 : 70, 6)][k];
  return Array.from({ length: 20 }, (_, i) => snap(make(i)));
}
function snap(v) { return Math.min(100, Math.max(0, Math.round(v / STEP) * STEP)); }

function stats(v) {
  const s = v.slice().sort((a, b) => a - b), n = s.length;
  const mean = s.reduce((sum, x) => sum + x, 0) / n;
  const median = n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  const counts = new Map();
  for (const x of s) counts.set(x, (counts.get(x) || 0) + 1);
  const top = Math.max(...counts.values());
  const modes = top > 1 ? [...counts.keys()].filter(x => counts.get(x) === top) : [];
  return { n, mean, median, modes, top };
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  const row = createDiv();
  row.parent(mainElement);
  row.position(10, drawHeight + 8);
  row.style('font-size', '16px');
  createSpan('Starting data: ').parent(row);
  datasetSelect = createSelect();
  datasetSelect.parent(row);
  DATASETS.forEach(name => datasetSelect.option(name));
  datasetSelect.style('font-size', '15px');
  datasetSelect.changed(() => loadData(DATASETS.indexOf(datasetSelect.value())));

  addButton = makeButton('Add Point', 10, () => {
    if (values.length < MAX_POINTS) { values.push(snap(stats(values).median)); lastAdded = values.length - 1; }
  });
  // an outlier goes to the end of the scale that is farther from the median
  outlierButton = makeButton('Add Outlier', 105, () => {
    if (values.length < MAX_POINTS) { values.push(stats(values).median > 50 ? 0 : 100); lastAdded = values.length - 1; }
  });
  resetButton = makeButton('Reset', 212, () => loadData(DATASETS.indexOf(datasetSelect.value())));
  loadData(0);

  describe('A dot plot of about twenty values from 0 to 100 with a red line at the mean, a green line at the median, ' +
    'and a blue band at the mode. Dots can be dragged left and right, and buttons add a point or an outlier. ' +
    'A table compares the mean, median, and mode at the start with their values now.', LABEL);
}

function makeButton(label, x, action) {
  const b = createButton(label);
  b.parent(document.querySelector('main'));
  b.position(x, drawHeight + 45);
  b.mousePressed(action);
  return b;
}

function loadData(k) {
  start = makeData(k);
  values = start.slice();
  lastAdded = -1;
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600;
  const now = stats(values), was = stats(start);
  textWrap(WORD);

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Mean, Median, and Mode Explorer', canvasWidth / 2, 7);
  fill('dimgray');
  textSize(narrow ? 12 : 14);
  text('Each dot is one value. Drag a dot left or right to change it.', canvasWidth / 2, narrow ? 31 : 35);

  const w = canvasWidth - 2 * margin;
  if (narrow) {
    drawDotPlot(52, 264, now, 12);
    drawTable(margin, 270, w, 82, was, now, 12, 17);
    drawMessage(margin, 358, w, 94, now, 12);
  } else {
    drawDotPlot(56, 318, now, 15);
    drawTable(margin, 326, 340, 126, was, now, 15, 23);
    drawMessage(margin + 350, 326, w - 350, 126, now, 15);
  }
}

// Stacked dot plot with the three measures of center drawn over it
function drawDotPlot(top, bottom, now, ts) {
  const x0 = margin + 22, x1 = canvasWidth - margin - 22, axisY = bottom - 24;
  const gx = v => x0 + v / 100 * (x1 - x0);
  plot = { x0, x1 };
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(margin, top, canvasWidth - 2 * margin, bottom - top, 10);

  // axis
  stroke('gray');
  strokeWeight(1.5);
  line(x0 - 10, axisY, x1 + 10, axisY);
  textSize(12);
  for (let v = 0; v <= 100; v += STEP) {
    stroke('gray');
    strokeWeight(1);
    line(gx(v), axisY, gx(v), axisY + (v % 10 === 0 ? 6 : 3));
    if (v % 10 === 0) {
      noStroke();
      fill('dimgray');
      textAlign(CENTER, TOP);
      text(v, gx(v), axisY + 8);
    }
  }

  // dots: equal values stack upward from the axis
  const lineTop = top + 3 * (ts + 3) + 8;
  const d = min((x1 - x0) / (100 / STEP) - 2, 18, (axisY - 6 - lineTop) / now.top);
  const stack = {};
  dots = values.map((v, i) => {
    stack[v] = (stack[v] || 0) + 1;
    return { i, x: gx(v), y: axisY - 4 - d * (stack[v] - 0.5) };
  });
  const over = dragIdx >= 0 ? dragIdx : dotAt(mouseX, mouseY);
  cursor(over >= 0 ? HAND : ARROW);
  for (const p of dots) {
    stroke(p.i === lastAdded ? 'black' : 'steelblue');
    strokeWeight(p.i === lastAdded ? 2 : 1);
    fill(p.i === over ? 'gold' : 'lightsteelblue');
    circle(p.x, p.y, d - 1);
  }

  // mode: wide translucent band, median: solid line, mean: dashed line (so all three show when they coincide)
  const showModes = now.modes.length >= 1 && now.modes.length <= 3;
  if (showModes) {
    stroke(65, 105, 225, 110);
    strokeWeight(9);
    for (const m of now.modes) line(gx(m), lineTop, gx(m), axisY);
  }
  stroke(MEDIAN_COLOR);
  strokeWeight(3);
  line(gx(now.median), lineTop, gx(now.median), axisY);
  stroke(MEAN_COLOR);
  strokeWeight(2.5);
  drawingContext.setLineDash([7, 5]);
  line(gx(now.mean), lineTop, gx(now.mean), axisY);
  drawingContext.setLineDash([]);

  // labels in three rows above the lines
  noStroke();
  textStyle(BOLD);
  textSize(ts);
  textAlign(CENTER, TOP);
  const label = (str, v, rowIndex, col) => {
    const half = textWidth(str) / 2 + 4;
    fill(col);
    text(str, constrain(gx(v), margin + half, canvasWidth - margin - half), top + 6 + rowIndex * (ts + 3));
  };
  label('Mean = ' + nf(now.mean, 1, 1), now.mean, 0, MEAN_COLOR);
  label('Median = ' + now.median, now.median, 1, MEDIAN_COLOR);
  if (showModes) for (const m of now.modes) label('Mode = ' + m, m, 2, MODE_COLOR);
  else label(now.modes.length ? 'More than three values tie for the mode' : 'No mode: no value repeats', 50, 2, MODE_COLOR);
  textStyle(NORMAL);
}

function modeText(t) { return t.modes.length === 0 ? 'none' : t.modes.length > 3 ? 'many' : t.modes.join(', '); }
function signed(dv) {
  const r = Math.round(dv * 10) / 10;
  return (r > 0 ? '+' : r < 0 ? '−' : '') + nf(Math.abs(r), 1, 1);
}

// Start, now, and change for each measure
function drawTable(x, y, w, h, was, now, ts, lh) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const cols = [x + 12, x + w * 0.42, x + w * 0.64, x + w * 0.87];
  const rows = [['Measure', 'Start', 'Now', 'Change', 'black'],
    ['Mean', nf(was.mean, 1, 1), nf(now.mean, 1, 1), signed(now.mean - was.mean), MEAN_COLOR],
    ['Median', String(was.median), String(now.median), signed(now.median - was.median), MEDIAN_COLOR],
    ['Mode', modeText(was), modeText(now), modeText(was) === modeText(now) ? 'same' : 'changed', MODE_COLOR]];
  noStroke();
  textSize(ts);
  rows.forEach((r, i) => {
    textStyle(i === 0 ? BOLD : NORMAL);
    fill(i === 0 ? 'black' : r[4]);
    textAlign(LEFT, TOP);
    text(r[0], cols[0], y + 9 + i * lh);
    fill('black');
    textAlign(CENTER, TOP);
    for (let c = 1; c <= 3; c++) text(r[c], cols[c], y + 9 + i * lh);
  });
  textStyle(NORMAL);
  if (lh > 20) {
    fill('dimgray');
    textAlign(LEFT, TOP);
    text('Values now: ' + now.n + ' (started with ' + was.n + ')', cols[0], y + h - ts - 9);
  }
}

// One short reading of the current data, built from the statistics
function drawMessage(x, y, w, h, now, ts) {
  fill('lightyellow');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const diff = now.mean - now.median, k = now.modes.length;
  let msg;
  if (Math.abs(diff) < DIVERGE) {
    msg = 'The mean and the median are within ' + DIVERGE + ' points of each other, so the data are balanced around the center.';
    if (k === 0) msg += ' No value repeats, so there is no mode.';
    else if (k > 1) msg += ' But ' + (k > 3 ? 'several' : k) + ' values tie for the mode: with more than one peak, ' +
      'the center may not be a typical value.';
    else msg += Math.abs(now.modes[0] - now.median) <= STEP ? ' The mode is close by too, so all three describe a typical value.'
      : ' The mode, the tallest stack, sits away from them.';
  } else {
    msg = 'The mean is ' + nf(Math.abs(diff), 1, 1) + ' points ' + (diff > 0 ? 'above' : 'below') + ' the median. ' +
      'Extreme values pull the mean toward the long tail. The median depends only on the middle position, so it is ' +
      'the better description of a typical value here.';
    msg += k === 0 ? ' No value repeats, so there is no mode.' : k > 1 ? ' Several values tie for the mode.'
      : ' The mode marks the tallest stack.';
  }
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text('What to notice', x + 12, y + 8);
  textStyle(NORMAL);
  textSize(ts);
  text(msg, x + 12, y + 12 + ts + 5, w - 24, h - ts - 22);
}

// Index of the value whose dot is nearest the given position, or -1
function dotAt(x, y) {
  let best = -1, bestD = 13;
  for (const p of dots) {
    const dd = dist(x, y, p.x, p.y);
    if (dd < bestD) { bestD = dd; best = p.i; }
  }
  return best;
}

function mousePressed() {
  dragIdx = mouseY < drawHeight ? dotAt(mouseX, mouseY) : -1;
}

function mouseDragged() {
  if (dragIdx < 0) return;
  values[dragIdx] = snap((mouseX - plot.x0) / (plot.x1 - plot.x0) * 100);
  return false;
}

function mouseReleased() {
  dragIdx = -1;
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
