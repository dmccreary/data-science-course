// Feature Scaling Visualizer
// CANVAS_HEIGHT: 550
// Bloom L2 (Understand): students choose a scaling method and a dataset and compare the
// histogram and the summary statistics before and after, to explain what each method does to
// the numbers and to the shape of a distribution. Nothing moves on its own.
//
// Model (x is a value, x' the scaled value):
//   Min-Max    x' = (x - min) / (max - min)
//   Standard   x' = (x - mean) / std       std is the population value (ddof = 0), as StandardScaler uses
//   Robust     x' = (x - median) / IQR     quartiles by linear interpolation, as RobustScaler uses
//   Log        x' = ln(1 + x)              np.log1p
// Outliers are the values more than 1.5 IQR beyond the quartiles of the original data. The four
// datasets of 200 values come from a seeded generator, so they are the same on every visit.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 470;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;
let selectLeft = 85;        // x position of the two menus, after their labels

const BEFORE_COLOR = 'steelblue', AFTER_COLOR = 'mediumpurple', AFTER_INK = 'rebeccapurple';
const BINS = 24;
// fixed: the statistics that the method sets to 0 or 1
const METHODS = [
  { name: 'None (original)', formula: "x' = x", fixed: [], f: x => x },
  { name: 'Min-Max [0, 1]', formula: "x' = (x - min) / (max - min)", fixed: ['min', 'max', 'range'],
    f: (x, s) => (x - s.min) / s.range },
  { name: 'Standard (Z-score)', formula: "x' = (x - mean) / std", fixed: ['mean', 'sd'], f: (x, s) => (x - s.mean) / s.sd },
  { name: 'Robust (median, IQR)', formula: "x' = (x - median) / IQR", fixed: ['median', 'iqr'],
    f: (x, s) => (x - s.median) / s.iqr },
  { name: 'Log transform', formula: "x' = ln(1 + x)", fixed: [], f: x => Math.log1p(x) }
];
const DATASETS = ['Bell-shaped (heights)', 'Right-skewed (income)', 'With outliers (prices)', 'Two groups (bimodal)'];
const ROWS = [['Min', 'min'], ['Max', 'max'], ['Range', 'range'], ['Mean', 'mean'], ['Median', 'median'], ['Std', 'sd'],
  ['IQR', 'iqr'], ['Outliers', 'outliers']];

let methodSelect, datasetSelect, outlierCheckbox;
let values = [];

// Seeded data: a linear congruential generator and the Box-Muller transform for normal values
function makeData(k) {
  let seed = 4100 + 53 * k;
  const uniform = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return (seed + 0.5) / 4294967296; };
  const normal = (mean, sd) => mean + sd * Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
  const many = (n, make) => Array.from({ length: n }, make);
  let v;
  if (k === 0) v = many(200, () => normal(170, 8));
  else if (k === 1) v = many(200, () => Math.exp(normal(3.6, 0.6)));
  else if (k === 2) v = many(192, () => normal(50, 8)).concat(many(8, () => normal(150, 12)));
  else v = many(100, () => normal(35, 5)).concat(many(100, () => normal(65, 6)));
  return v.map(x => Math.round(x * 10) / 10);
}

// Summary statistics. Quartiles use linear interpolation between sorted values.
function summarize(v) {
  const s = v.slice().sort((a, b) => a - b), n = v.length;
  const q = p => { const pos = (n - 1) * p, lo = Math.floor(pos); return s[lo] + (pos - lo) * (s[Math.min(lo + 1, n - 1)] - s[lo]); };
  const mean = v.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n);
  const q1 = q(0.25), q3 = q(0.75), iqr = q3 - q1;
  const flags = v.map(x => x < q1 - 1.5 * iqr || x > q3 + 1.5 * iqr);
  return { min: s[0], max: s[n - 1], range: s[n - 1] - s[0], mean, median: q(0.5), sd, q1, q3, iqr, flags,
    outliers: flags.filter(f => f).length };
}

// One decimal for large values, two for small ones, and never "-0.00"
function fmt(v) {
  if (Math.abs(v) < 0.005) return '0.00';
  return Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2);
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  methodSelect = createSelect();
  methodSelect.parent(mainElement);
  methodSelect.position(selectLeft, drawHeight + 10);
  METHODS.forEach(m => methodSelect.option(m.name));
  methodSelect.selected(METHODS[1].name);
  methodSelect.style('font-size', '15px');
  methodSelect.style('width', '178px');

  datasetSelect = createSelect();
  datasetSelect.parent(mainElement);
  datasetSelect.position(selectLeft, drawHeight + 45);
  DATASETS.forEach(name => datasetSelect.option(name));
  datasetSelect.selected(DATASETS[2]);
  datasetSelect.style('font-size', '15px');
  datasetSelect.style('width', '178px');
  datasetSelect.changed(() => { values = makeData(DATASETS.indexOf(datasetSelect.value())); });

  outlierCheckbox = createCheckbox(' Mark outliers', true);
  outlierCheckbox.parent(mainElement);
  outlierCheckbox.position(selectLeft + 188, drawHeight + 46);
  outlierCheckbox.style('font-size', '16px');

  values = makeData(2);

  describe('Two histograms, one of the original data and one after scaling, beside a table that compares the ' +
    'minimum, maximum, range, mean, median, standard deviation, interquartile range, and number of outliers before ' +
    'and after. Menus choose the scaling method (none, min-max, standard, robust, or log) and one of four datasets. ' +
    'A checkbox colors the outliers red in both histograms.', LABEL);
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
  const method = METHODS.find(m => m.name === methodSelect.value());
  const before = summarize(values);
  const scaled = values.map(x => method.f(x, before));
  const after = summarize(scaled);
  const marks = outlierCheckbox.checked() ? before.flags : values.map(() => false);
  textWrap(WORD);

  noStroke();
  fill('black');
  textStyle(NORMAL);
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Feature Scaling Visualizer', canvasWidth / 2, 8);

  // histograms on the left and the comparison on the right, or stacked when narrow
  const top = narrow ? 32 : 44, fullW = canvasWidth - 2 * margin, bottom = drawHeight - 8;
  const plotW = narrow ? fullW : fullW * 0.57;
  const tableH = 150;                                        // height of the comparison panel when narrow
  const histH = ((narrow ? bottom - tableH - 6 : bottom) - top - 6) / 2;
  drawHistogram(margin, top, plotW, histH, 'Original data', values, marks, before, BEFORE_COLOR, 'black', narrow);
  drawHistogram(margin, top + histH + 6, plotW, histH, 'After: ' + method.name, scaled, marks, after, AFTER_COLOR,
    AFTER_INK, narrow);
  if (narrow) drawComparison(margin, bottom - tableH, fullW, tableH, method, before, after, scaled, narrow);
  else drawComparison(margin + plotW + 10, top, fullW - plotW - 10, bottom - top, method, before, after, scaled, narrow);

  // control labels
  noStroke();
  fill('black');
  textStyle(NORMAL);
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Scaling:', 10, drawHeight + 21);
  text('Dataset:', 10, drawHeight + 56);
}

// A histogram of BINS equal bins from the smallest to the largest value. Marked values are red.
function drawHistogram(x, y, w, h, title, data, marks, st, col, ink, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = narrow ? 11 : 13, px = x + 16, pw = w - 32;
  const top = y + (narrow ? 20 : 28), base = y + h - (narrow ? 17 : 22);
  const sx = v => px + (v - st.min) / st.range * pw;

  noStroke();
  fill(ink);
  textStyle(BOLD);
  textSize(ts + 1);
  textAlign(LEFT, TOP);
  text(title, x + 10, y + (narrow ? 5 : 8));
  textStyle(NORMAL);

  // count the values, and the marked values, in each bin
  const counts = new Array(BINS).fill(0), marked = new Array(BINS).fill(0);
  data.forEach((v, i) => {
    const b = Math.min(BINS - 1, Math.floor((v - st.min) / st.range * BINS));
    counts[b]++;
    if (marks[i]) marked[b]++;
  });
  const unit = (base - top - 14) / Math.max(...counts), bw = pw / BINS;
  stroke('white');
  strokeWeight(1);
  for (let b = 0; b < BINS; b++) {
    fill(col);
    rect(px + b * bw, base - counts[b] * unit, bw, (counts[b] - marked[b]) * unit);
    fill('crimson');
    rect(px + b * bw, base - marked[b] * unit, bw, marked[b] * unit);
  }

  // a gold band over the bars for the middle 50% of the data (Q1 to Q3), with a tick at the median
  noStroke();
  fill('goldenrod');
  rect(sx(st.q1), top, max(2, sx(st.q3) - sx(st.q1)), 7, 3);
  stroke('black');
  strokeWeight(2);
  line(sx(st.median), top - 2, sx(st.median), top + 9);
  noStroke();
  fill('darkgoldenrod');
  textSize(ts);
  textAlign(RIGHT, TOP);
  text('middle 50%: ' + fmt(st.q1) + ' to ' + fmt(st.q3), x + w - 10, y + (narrow ? 6 : 9));

  // axis with round tick values
  const rough = st.range / 8, power = Math.pow(10, Math.floor(Math.log10(rough)));
  const step = [1, 2, 5, 10].map(m => m * power).find(t => t >= rough);
  const decimals = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
  stroke('gray');
  strokeWeight(1);
  line(px, base, px + pw, base);
  for (let i = Math.ceil(st.min / step - 1e-9); i * step <= st.max + 1e-9; i++) {
    stroke('gray');
    line(sx(i * step), base, sx(i * step), base + 4);
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    text((i * step).toFixed(decimals), sx(i * step), base + 5);
  }
}

// What the chosen method did, in words. The numbers come from the data.
function explain(method, before, after) {
  const n = before.outliers, k = METHODS.indexOf(method);
  const far = fmt(Math.max(Math.abs(after.min), Math.abs(after.max)));
  const middle = 'the middle 50% of the data into ' + fmt(after.q1) + ' to ' + fmt(after.q3) + '.';
  if (k === 0) return 'No scaling yet. Red bars are outliers: values more than 1.5 × IQR beyond the quartiles. Choose ' +
    'a method to see what happens to the numbers and to the shape.';
  if (k === 1) return 'Every value now lies between 0 and 1. The bars keep their shape: only the axis changed. ' +
    (n > 0 ? 'The ' + n + ' outliers take the far end of the range and squeeze ' + middle : 'It puts ' + middle);
  if (k === 2) return 'The mean is now 0 and the standard deviation 1. The bars keep their shape. ' +
    (n > 0 ? 'Outliers stay extreme: the farthest value is ' + far + ' standard deviations from the mean.'
      : 'Each value now says how many standard deviations it is from the mean.');
  if (k === 3) return 'The median is now 0 and the IQR 1. The bars keep their shape. ' +
    (n > 0 ? 'The median and IQR hardly notice outliers, so the outliers do not distort the scale. They are still ' +
      'in the data, as far out as ' + far + '.' : 'Each value now says how many IQRs it is from the median.');
  return 'The only method here that changes the shape. Large values are pulled in far more than small ones. ' +
    (after.outliers < n ? 'Here the long right tail shrinks: ' : 'That helps a long right tail, but not this data: ') +
    n + ' values lie beyond the 1.5 × IQR fences before and ' + after.outliers + ' after.';
}

// Before and after statistics side by side, the formula, and the explanation
function drawComparison(x, y, w, h, method, before, after, scaled, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const pad = narrow ? 8 : 12, ts = narrow ? 11 : 14;
  const rowH = narrow ? 14.5 : 25;
  const tw = narrow ? (w - 2 * pad) * 0.5 : w - 2 * pad;     // table width
  const tx = x + pad, ty = y + pad - (narrow ? 2 : 0);
  const cols = [tx + tw * 0.36, tx + tw * 0.68, tx + tw];    // right edges of the label, Before, and After columns

  noStroke();
  textSize(ts);
  textStyle(BOLD);
  textAlign(RIGHT, CENTER);
  fill(BEFORE_COLOR);
  text('Before', cols[1] - 6, ty + rowH / 2);
  fill(AFTER_INK);
  text('After', cols[2] - 6, ty + rowH / 2);
  ROWS.forEach(([label, key], r) => {
    const ry = ty + (r + 1) * rowH, isCount = key === 'outliers';
    stroke('gainsboro');
    strokeWeight(1);
    line(tx, ry, tx + tw, ry);
    noStroke();
    if (method.fixed.includes(key)) {
      fill(147, 112, 219, 70);
      rect(cols[1] + 4, ry + 1, cols[2] - cols[1] - 4, rowH - 2, 4);
    }
    fill('black');
    textStyle(NORMAL);
    textAlign(LEFT, CENTER);
    text(label, tx + 2, ry + rowH / 2 + 1);
    textAlign(RIGHT, CENTER);
    text(isCount ? before[key] : fmt(before[key]), cols[1] - 6, ry + rowH / 2 + 1);
    fill(AFTER_INK);
    textStyle(method.fixed.includes(key) ? BOLD : NORMAL);
    text(isCount ? after[key] : fmt(after[key]), cols[2] - 6, ry + rowH / 2 + 1);
  });

  // formula and explanation: under the table when wide, beside it when narrow
  const nx = narrow ? tx + tw + 12 : tx, nw = narrow ? w - 2 * pad - tw - 12 : tw;
  const ny = narrow ? ty + 2 : ty + (ROWS.length + 1) * rowH + 12;
  noStroke();
  fill(AFTER_INK);
  textStyle(BOLD);
  textSize(narrow ? 12 : 15);
  textAlign(LEFT, TOP);
  text(method.formula, nx, ny);
  fill('black');
  textStyle(NORMAL);
  textSize(ts);
  textLeading(ts + (narrow ? 3 : 5));
  text(explain(method, before, after), nx, ny + (narrow ? 18 : 26), nw, y + h - ny - (narrow ? 20 : 30));
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
