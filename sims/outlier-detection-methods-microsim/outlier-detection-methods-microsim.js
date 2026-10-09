// Outlier Detection Playground
// CANVAS_HEIGHT: 585
// Bloom L3 (Apply): students apply the Z-score rule and the IQR rule to the same data, move
// the two thresholds, and see at once which points each rule flags and how the rules differ.
//
// Model:
//   Z-score rule  z = (x - mean) / std. A value is flagged when |z| > threshold.
//                 std is the population standard deviation (ddof = 0), as scipy.stats.zscore uses.
//   IQR rule      IQR = Q3 - Q1. A value is flagged when it is below Q1 - k IQR or above
//                 Q3 + k IQR. Quartiles use linear interpolation, as Series.quantile does.
// The four datasets come from a seeded generator, so they are the same on every visit.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 470;
let controlHeight = 115;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;
let labelWidth = 145;       // width of the label span in each control row
let valueWidth = 40;        // width of the value span in each slider row

const DATASETS = ['Heights in cm (bell-shaped)', 'Incomes in $1000s (skewed)', 'Commute minutes (two groups)',
  'Ages with entry errors'];
const Z_COLOR = 'rebeccapurple', IQR_COLOR = 'teal';

let datasetSelect, zRow, iqrRow;
let values = [];
let dots = [];              // the dots drawn this frame, for the hover label

// Seeded data: a linear congruential generator and the Box-Muller transform for normal values
function makeData(k) {
  let seed = [2024, 555, 2218, 2315][k];
  const uniform = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return (seed + 0.5) / 4294967296; };
  const normal = (mean, sd) => mean + sd * Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
  const many = (n, make) => Array.from({ length: n }, make);
  let v;
  if (k === 0) v = many(40, () => normal(170, 8)).concat([128, 205, 212]);          // three obvious extremes
  else if (k === 1) v = many(45, () => Math.exp(normal(3.7, 0.7)));                 // long right tail, all valid
  else if (k === 2) v = many(22, () => normal(22, 4)).concat(many(22, () => normal(58, 5)), [118]);
  else v = many(40, () => normal(34, 9)).concat([-5, 0, 150, 260]);                 // impossible ages
  return v.map(x => Math.round(x * 10) / 10);
}

// Quantile of a sorted array with linear interpolation (the pandas default)
function quantile(sorted, q) {
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos);
  return sorted[lo] + (pos - lo) * (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]);
}
const meanOf = a => a.reduce((sum, x) => sum + x, 0) / a.length;

// Statistics, limits, and flags for both rules
function analyze(v, zLimit, k) {
  const mean = meanOf(v), sd = Math.sqrt(meanOf(v.map(x => (x - mean) * (x - mean))));
  const s = v.slice().sort((a, b) => a - b);
  const q1 = quantile(s, 0.25), med = quantile(s, 0.5), q3 = quantile(s, 0.75), iqr = q3 - q1;
  const z = v.map(x => (x - mean) / sd);
  const zLo = mean - zLimit * sd, zHi = mean + zLimit * sd, iLo = q1 - k * iqr, iHi = q3 + k * iqr;
  return { n: v.length, mean, sd, q1, med, q3, iqr, z, zLo, zHi, iLo, iHi, min: s[0], max: s[s.length - 1],
    zFlag: z.map(t => Math.abs(t) > zLimit), iFlag: v.map(x => x < iLo || x > iHi) };
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  // first control row: the dataset menu, lined up with the slider labels
  const row = createDiv();
  row.parent(mainElement);
  row.position(10, drawHeight + 8);
  row.style('font-size', '16px');
  const labelSpan = createSpan('Dataset:');
  labelSpan.parent(row);
  labelSpan.style('display', 'inline-block');
  labelSpan.style('width', labelWidth + 'px');
  datasetSelect = createSelect();
  datasetSelect.parent(row);
  DATASETS.forEach(name => datasetSelect.option(name));
  datasetSelect.style('font-size', '15px');
  datasetSelect.changed(() => { values = makeData(DATASETS.indexOf(datasetSelect.value())); });

  zRow = makeSliderRow('Z-score threshold', 1.5, 4, 3, 0.1, 1);
  iqrRow = makeSliderRow('IQR multiplier', 1, 3, 1.5, 0.1, 2);
  resizeSliders();
  values = makeData(0);

  describe('Two dot plots of the same data, one for the Z-score rule and one for the IQR rule. Sliders set the ' +
    'Z-score threshold and the IQR multiplier, and a menu chooses one of four datasets. Flagged points turn red, ' +
    'shaded zones show where each rule flags values, and a panel lists the statistics, the limits, the flagged ' +
    'values, and how the two rules differ.', LABEL);
}

// A control row built from a div: the label and value sit in fixed-width spans so that
// every slider in the control area starts at the same x position.
function makeSliderRow(label, minValue, maxValue, startValue, step, rowIndex) {
  const row = createDiv();
  row.parent(document.querySelector('main'));
  row.position(10, drawHeight + 8 + rowIndex * 35);
  row.style('font-size', '16px');
  const labelSpan = createSpan(label + ':');
  labelSpan.parent(row);
  labelSpan.style('display', 'inline-block');
  labelSpan.style('width', labelWidth + 'px');
  const valueSpan = createSpan('');
  valueSpan.parent(row);
  valueSpan.style('display', 'inline-block');
  valueSpan.style('width', valueWidth + 'px');
  valueSpan.style('font-weight', 'bold');
  const slider = createSlider(minValue, maxValue, startValue, step);
  slider.parent(row);
  slider.style('vertical-align', 'middle');
  return { slider, valueSpan };
}

function resizeSliders() {
  const w = max(60, canvasWidth - labelWidth - valueWidth - 40);
  zRow.slider.size(w);
  iqrRow.slider.size(w);
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const zLimit = zRow.slider.value(), k = iqrRow.slider.value();
  zRow.valueSpan.html(nf(zLimit, 1, 1));
  iqrRow.valueSpan.html(nf(k, 1, 1));
  const st = analyze(values, zLimit, k);
  const narrow = canvasWidth < 600;
  textWrap(WORD);

  noStroke();
  fill('black');
  textStyle(NORMAL);
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Outlier Detection Playground', canvasWidth / 2, 8);

  // the two plots on the left and the results on the right, or stacked when narrow
  const top = narrow ? 32 : 44, fullW = canvasWidth - 2 * margin, bottom = drawHeight - 8;
  const plotW = narrow ? fullW : fullW * 0.6;
  const stripH = narrow ? 112 : (bottom - top - 8) / 2;
  dots = [];
  drawStrip(margin, top, plotW, stripH, 'Z-score rule:  |z| > ' + nf(zLimit, 1, 1), Z_COLOR, st.zLo, st.zHi,
    st.zFlag, st.iFlag, st, narrow, false);
  drawStrip(margin, top + stripH + 8, plotW, stripH, 'IQR rule:  beyond ' + nf(k, 1, 1) + ' × IQR', IQR_COLOR,
    st.iLo, st.iHi, st.iFlag, st.zFlag, st, narrow, true);
  if (narrow) drawResults(margin, top + 2 * stripH + 14, fullW, bottom - top - 2 * stripH - 14, st, narrow);
  else drawResults(margin + plotW + 10, top, fullW - plotW - 10, bottom - top, st, narrow);
  drawHover(narrow);
}

// Give each dot a stacking level so that dots closer than d pixels do not overlap
function dodge(xs, d) {
  const order = xs.map((x, i) => i).sort((a, b) => xs[a] - xs[b]);
  const lastX = [], levels = [];
  order.forEach(i => {
    let level = 0;
    while (lastX[level] !== undefined && xs[i] - lastX[level] < d) level++;
    lastX[level] = xs[i];
    levels[i] = level;
  });
  return levels;
}

// One dot plot with the zones where a rule flags values. flags: this rule. others: the other rule.
function drawStrip(x, y, w, h, title, col, lo, hi, flags, others, st, narrow, showBox) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = narrow ? 11 : 13, px = x + 14, pw = w - 28;
  const top = y + (narrow ? 20 : 28), base = y + h - (narrow ? 18 : 24);
  const pad = (st.max - st.min) * 0.06, axisMin = st.min - pad, axisMax = st.max + pad;
  const sx = v => px + (v - axisMin) / (axisMax - axisMin) * pw;
  const count = flags.filter(f => f).length;

  noStroke();
  textSize(ts + 1);
  textStyle(BOLD);
  fill(col);
  textAlign(LEFT, TOP);
  text(title, x + 10, y + (narrow ? 5 : 8));
  fill('crimson');
  textAlign(RIGHT, TOP);
  text(count + ' flagged', x + w - 10, y + (narrow ? 5 : 8));
  textStyle(NORMAL);

  // shaded zones outside the limits, and a dashed line at each limit that falls on the axis
  textSize(ts);
  for (const [limit, from, to, preferred] of [[lo, axisMin, lo, -1], [hi, hi, axisMax, 1]]) {
    if (limit <= axisMin || limit >= axisMax) continue;
    noStroke();
    fill(220, 20, 60, 30);
    rect(sx(from), top, sx(to) - sx(from), base - top);
    stroke(col);
    strokeWeight(1.5);
    drawingContext.setLineDash([5, 4]);
    line(sx(limit), top, sx(limit), base);
    drawingContext.setLineDash([]);
    // the value goes on the outer side of the line, or the inner side when there is no room
    const label = nf(limit, 1, 1), room = textWidth(label) + 6;
    let side = preferred;
    if (side < 0 && sx(limit) - room < px) side = 1;
    if (side > 0 && sx(limit) + room > px + pw) side = -1;
    noStroke();
    fill(col);
    textAlign(side > 0 ? LEFT : RIGHT, TOP);
    text(label, sx(limit) + 4 * side, top + 1);
  }

  // axis with round tick values
  const rough = (axisMax - axisMin) / 6, power = Math.pow(10, Math.floor(Math.log10(rough)));
  const step = [1, 2, 5, 10].map(m => m * power).find(t => t >= rough);
  stroke('gray');
  strokeWeight(1);
  line(px, base, px + pw, base);
  for (let t = Math.ceil(axisMin / step) * step; t <= axisMax; t += step) {
    stroke('gray');
    line(sx(t), base, sx(t), base + 4);
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    text(Math.round(t), sx(t), base + 5);
  }

  // what the rule is built on: the mean for Z-scores, the box from Q1 to Q3 for the IQR
  stroke(col);
  strokeWeight(2);
  if (showBox) {
    fill(0, 128, 128, 45);
    rect(sx(st.q1), top + (base - top) * 0.45, sx(st.q3) - sx(st.q1), (base - top) * 0.55);
    line(sx(st.med), top + (base - top) * 0.45, sx(st.med), base);
  } else {
    line(sx(st.mean), top + ts + 3, sx(st.mean), base);
    noStroke();
    fill(col);
    textAlign(CENTER, TOP);
    text('mean', sx(st.mean), top + 1);
  }

  // dots: red when flagged, with a dark ring when only this rule flags the point
  const d = narrow ? 7 : 9, xs = values.map(sx), levels = dodge(xs, d);
  const rise = min(d, (base - top - d - ts - 4) / max(1, max(levels)));     // stacks stay under the labels
  values.forEach((v, i) => {
    const dy = base - d / 2 - 1 - levels[i] * rise;
    stroke(flags[i] && !others[i] ? 'black' : 'white');
    strokeWeight(flags[i] && !others[i] ? 2 : 0.8);
    fill(flags[i] ? 'crimson' : 'steelblue');
    circle(xs[i], dy, d);
    dots.push({ x: xs[i], y: dy, v, z: st.z[i] });
  });
}

// A label for the dot under the mouse
function drawHover(narrow) {
  const hit = dots.find(p => dist(mouseX, mouseY, p.x, p.y) < 6);
  if (!hit) return;
  const label = 'value ' + hit.v + ',  z = ' + nf(hit.z, 1, 2);
  textSize(narrow ? 11 : 13);
  textStyle(BOLD);
  const w = textWidth(label) + 12, bx = constrain(hit.x - w / 2, 4, canvasWidth - w - 4);
  noStroke();
  fill('black');
  rect(bx, hit.y - 28, w, 20, 4);
  fill('white');
  textAlign(CENTER, CENTER);
  text(label, bx + w / 2, hit.y - 17);
  textStyle(NORMAL);
}

// Statistics, limits, flagged values, and how the two rules compare
function drawResults(x, y, w, h, st, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const pad = narrow ? 8 : 12, ts = narrow ? 11 : 14, lh = ts + (narrow ? 3 : 5);
  const f = v => nf(v, 1, 1);
  const flagged = flags => values.filter((v, i) => flags[i]).sort((a, b) => a - b);
  const describeRule = (name, lo, hi, flags) => {
    const list = flagged(flags);
    return name + ' flags values outside ' + f(lo) + ' to ' + f(hi) + '. Flagged: ' + list.length + ' of ' + st.n +
      ' (' + f(100 * list.length / st.n) + '%)' + (list.length > 0 ? ': ' + list.slice(0, 6).join(', ') : '') +
      (list.length > 6 ? ', ...' : '') + '.';
  };
  const keep = flags => f(meanOf(values.filter((v, i) => !flags[i])));
  const onlyI = values.filter((v, i) => st.iFlag[i] && !st.zFlag[i]).length;
  const onlyZ = values.filter((v, i) => st.zFlag[i] && !st.iFlag[i]).length;
  const points = n => n + (n === 1 ? ' point' : ' points');
  let insight = 'Each rule flags points the other one misses (dark rings).';
  if (onlyI === 0 && onlyZ === 0) insight = 'Both rules flag the same points at these settings.';
  else if (onlyZ === 0) insight = 'The IQR rule flags ' + points(onlyI) + ' that the Z-score rule misses (dark rings).';
  else if (onlyI === 0) insight = 'The Z-score rule flags ' + points(onlyZ) + ' that the IQR rule does not (dark rings).';

  // paragraphs: text, color, style, and the number of lines kept free for it
  const paragraphs = [
    ['n = ' + st.n + '    mean = ' + f(st.mean) + '    std = ' + f(st.sd) + '\nQ1 = ' + f(st.q1) + '    median = ' +
      f(st.med) + '    Q3 = ' + f(st.q3) + '    IQR = ' + f(st.iqr), 'black', NORMAL, narrow ? 2 : 3],
    [describeRule('Z-score rule', st.zLo, st.zHi, st.zFlag), Z_COLOR, NORMAL, narrow ? 2 : 4],
    [describeRule('IQR rule', st.iLo, st.iHi, st.iFlag), IQR_COLOR, NORMAL, narrow ? 3 : 4],
    ['Mean of all values ' + f(st.mean) + '. Without the Z-score flags ' + keep(st.zFlag) + '. Without the IQR flags ' +
      keep(st.iFlag) + '.', 'black', NORMAL, narrow ? 2 : 3],
    [insight + ' Which is right? A flag is a reason to look, not proof of an error.', 'black', BOLD, narrow ? 2 : 4]
  ];
  let cy = y + pad;
  noStroke();
  textSize(ts);
  textLeading(lh);
  textAlign(LEFT, TOP);
  for (const [str, col, style, lines] of paragraphs) {
    fill(col);
    textStyle(style);
    text(str, x + pad, cy, w - 2 * pad, lines * lh + 3);
    cy += lines * lh + (narrow ? 4 : 10);
    // a rule between paragraphs
    stroke('gainsboro');
    strokeWeight(1);
    if (style !== BOLD) line(x + pad, cy - (narrow ? 2 : 5), x + w - pad, cy - (narrow ? 2 : 5));
    noStroke();
  }
  textStyle(NORMAL);
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  resizeSliders();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
