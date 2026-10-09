// Box Plot Anatomy
// CANVAS_HEIGHT: 515
// Bloom L1 (Remember): students identify and recall the seven labeled parts of a box plot.
// They hover over or click a numbered part to read what it is, its value in this data, and
// the pandas code that computes it. Hiding the names turns the diagram into a self-test.
//
// Model (the default box plot of matplotlib and pandas):
//   Q1, median, Q3   quantiles 0.25, 0.5, 0.75 with linear interpolation (Series.quantile default)
//   IQR = Q3 - Q1    fences at Q1 - 1.5 IQR and Q3 + 1.5 IQR
//   whiskers         reach to the smallest and largest data values that lie inside the fences
//   outliers         values outside the fences, each drawn as a separate point

let containerWidth;
let canvasWidth = 400;
let drawHeight = 470;
let controlHeight = 45;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

// 21 quiz scores out of 100: one unusually low score and one unusually high score
const DATA = [18, 41, 44, 47, 50, 52, 54, 55, 57, 58, 60, 61, 63, 65, 66, 68, 70, 73, 76, 79, 97];
const AXIS_LO = 10, AXIS_HI = 104;
const BOX_HALF = 26;        // half the thickness of the box in pixels

const fmt = v => String(Math.round(v * 100) / 100);
const PARTS = [
  { name: 'Minimum (lower whisker end)', label: 'Minimum', color: 'royalblue',
    desc: 'The smallest value that is not an outlier. The lower whisker stops at the smallest data value that is ' +
      'still at or above the lower fence, Q1 − 1.5 × IQR.',
    value: s => 'lower fence = ' + fmt(s.q1) + ' − 1.5 × ' + fmt(s.iqr) + ' = ' + fmt(s.loFence) +
      '. Smallest score at or above it: ' + fmt(s.wLo),
    code: 's[s >= q1 - 1.5 * iqr].min()' },
  { name: 'Q1, the first quartile', label: 'Q1', color: 'seagreen',
    desc: 'About 25% of the values lie below Q1. It is the edge of the box on the low side.',
    value: s => 'Q1 = ' + fmt(s.q1) + '. ' + s.s.filter(v => v < s.q1).length + ' of the ' + s.n + ' scores are below it',
    code: 'q1 = s.quantile(0.25)' },
  { name: 'Median, the second quartile (Q2)', label: 'Median', color: 'crimson',
    desc: 'Half of the values lie below the median and half lie above it. It is the line inside the box.',
    value: s => 'median = ' + fmt(s.med) + '. ' + s.s.filter(v => v < s.med).length + ' scores are below it and ' +
      s.s.filter(v => v > s.med).length + ' are above it',
    code: 's.median()     # the same as s.quantile(0.5)' },
  { name: 'Q3, the third quartile', label: 'Q3', color: 'seagreen',
    desc: 'About 75% of the values lie below Q3. It is the edge of the box on the high side.',
    value: s => 'Q3 = ' + fmt(s.q3) + '. ' + s.s.filter(v => v < s.q3).length + ' of the ' + s.n + ' scores are below it',
    code: 'q3 = s.quantile(0.75)' },
  { name: 'Maximum (upper whisker end)', label: 'Maximum', color: 'royalblue',
    desc: 'The largest value that is not an outlier. The upper whisker stops at the largest data value that is ' +
      'still at or below the upper fence, Q3 + 1.5 × IQR.',
    value: s => 'upper fence = ' + fmt(s.q3) + ' + 1.5 × ' + fmt(s.iqr) + ' = ' + fmt(s.hiFence) +
      '. Largest score at or below it: ' + fmt(s.wHi),
    code: 's[s <= q3 + 1.5 * iqr].max()' },
  { name: 'IQR, the interquartile range', label: 'IQR', color: 'chocolate',
    desc: 'The length of the box, Q3 − Q1. The box holds the middle 50% of the data, so the IQR measures spread ' +
      'without being affected by outliers.',
    value: s => 'IQR = ' + fmt(s.q3) + ' − ' + fmt(s.q1) + ' = ' + fmt(s.iqr) + '. ' +
      s.s.filter(v => v >= s.q1 && v <= s.q3).length + ' of the ' + s.n + ' scores lie from Q1 to Q3',
    code: 'iqr = q3 - q1' },
  { name: 'Outliers', label: 'Outlier', color: 'purple',
    desc: 'Values more than 1.5 × IQR beyond the box, which means outside the two fences. Each outlier is drawn ' +
      'as its own point.',
    value: s => 'scores outside ' + fmt(s.loFence) + ' to ' + fmt(s.hiFence) + ': ' + s.outliers.map(fmt).join(' and '),
    code: 's[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]' }
];

let verticalBox, namesBox;
let st;                     // statistics of DATA
let selected = -1;          // index of the clicked part, or -1
let targets = [];           // clickable rectangles of the last frame: { part, x0, y0, x1, y1 }
let lay = {};               // geometry of the value axis for the current frame

// Quantile of a sorted array with linear interpolation (the pandas and NumPy default)
function quantile(sorted, q) {
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos);
  return sorted[lo] + (pos - lo) * (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]);
}

function computeStats(data) {
  const s = data.slice().sort((a, b) => a - b);
  const q1 = quantile(s, 0.25), med = quantile(s, 0.5), q3 = quantile(s, 0.75), iqr = q3 - q1;
  const loFence = q1 - 1.5 * iqr, hiFence = q3 + 1.5 * iqr;
  const inside = s.filter(v => v >= loFence && v <= hiFence);
  return { s, n: s.length, q1, med, q3, iqr, loFence, hiFence, wLo: inside[0], wHi: inside[inside.length - 1],
    outliers: s.filter(v => v < loFence || v > hiFence) };
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);
  st = computeStats(DATA);

  verticalBox = createCheckbox(' Vertical', false);
  verticalBox.parent(mainElement);
  verticalBox.position(10, drawHeight + 11);
  verticalBox.style('font-size', '16px');
  namesBox = createCheckbox(' Show names', true);
  namesBox.parent(mainElement);
  namesBox.position(125, drawHeight + 11);
  namesBox.style('font-size', '16px');

  describe('A box plot of 21 quiz scores with seven numbered parts: minimum, first quartile, median, third quartile, ' +
    'maximum, interquartile range, and outliers. Hovering over or clicking a part shows its meaning, its value, and ' +
    'the pandas code that computes it. Checkboxes switch the plot to vertical and hide the part names.', LABEL);
}

// Screen position of data value v, offset `off` pixels across the plot and `along` pixels toward higher values
function P(v, off, along = 0) {
  const p = lay.a + (v - AXIS_LO) / (AXIS_HI - AXIS_LO) * (lay.b - lay.a);
  return lay.vertical ? [lay.c + off, p - along] : [p + along, lay.c + off];
}
function seg(v1, o1, v2, o2) { line(...P(v1, o1), ...P(v2, o2)); }

function hitPart(x, y) {
  const t = targets.find(r => x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1);
  return t ? t.part : -1;
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
  const hovered = hitPart(mouseX, mouseY);
  cursor(hovered >= 0 ? HAND : ARROW);
  const active = hovered >= 0 ? hovered : selected;
  targets = [];

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Box Plot Anatomy', canvasWidth / 2, 7);
  fill('dimgray');
  textSize(narrow ? 12 : 14);
  text('A box plot of ' + st.n + ' quiz scores. Each gray dot is one score.', canvasWidth / 2, narrow ? 31 : 35);

  drawPlot(52, 268, verticalBox.checked(), active, narrow);
  drawPanel(276, drawHeight - 8, active, narrow);
}

function drawPlot(top, bottom, vertical, active, narrow) {
  const bh = BOX_HALF;
  // a and b are the pixel positions of the low and high ends of the value axis, c is the center line of the box
  lay = vertical ? { vertical, a: bottom - 8, b: top + 6, c: canvasWidth / 2 + 12 }
    : { vertical, a: margin + 28, b: canvasWidth - margin - 28, c: top + 83 };
  const axisOff = vertical ? -150 : 106, dotOff = vertical ? 138 : 84;
  const weight = id => (active === id ? 6 : 3);

  // value axis with tick marks
  stroke('gray');
  strokeWeight(1.5);
  seg(AXIS_LO, axisOff, AXIS_HI, axisOff);
  textSize(12);
  for (let v = 20; v <= 100; v += 20) {
    const p = P(v, axisOff);
    stroke('gray');
    if (vertical) line(p[0] - 5, p[1], p[0], p[1]); else line(p[0], p[1], p[0], p[1] + 5);
    noStroke();
    fill('dimgray');
    textAlign(vertical ? RIGHT : CENTER, vertical ? CENTER : TOP);
    text(v, p[0] - (vertical ? 8 : 0), p[1] + (vertical ? 0 : 7));
  }

  // the data, one dot per score, in three rows so that neighbors do not hide each other
  noStroke();
  fill(110, 110, 110, 180);
  st.s.forEach((v, i) => circle(...P(v, dotOff + (i % 3 - 1) * 7), 7));

  // fences, shown while a part that depends on them is active
  if (active === 0 || active === 4 || active === 6) {
    for (const f of [st.loFence, st.hiFence]) {
      stroke('gray');
      strokeWeight(1);
      drawingContext.setLineDash([4, 4]);
      seg(f, -bh - 30, f, dotOff + 12);
      drawingContext.setLineDash([]);
      noStroke();
      fill('dimgray');
      textSize(12);
      const p = P(f, -bh - 30);
      textAlign(vertical ? RIGHT : CENTER, vertical ? CENTER : BOTTOM);
      text('fence ' + fmt(f), p[0] - (vertical ? 4 : 0), p[1] - (vertical ? 0 : 2));
    }
  }

  // whiskers, with the caps that mark the minimum (part 0) and the maximum (part 4)
  stroke(PARTS[0].color);
  strokeWeight(3);
  seg(st.wLo, 0, st.q1, 0);
  seg(st.q3, 0, st.wHi, 0);
  strokeWeight(weight(0));
  seg(st.wLo, -12, st.wLo, 12);
  strokeWeight(weight(4));
  seg(st.wHi, -12, st.wHi, 12);

  // box: fill, the two long sides, the Q1 and Q3 edges, and the median line
  noStroke();
  fill(active === 5 ? color(255, 165, 0, 110) : 'white');
  quad(...P(st.q1, -bh), ...P(st.q3, -bh), ...P(st.q3, bh), ...P(st.q1, bh));
  stroke('dimgray');
  strokeWeight(2);
  seg(st.q1, -bh, st.q3, -bh);
  seg(st.q1, bh, st.q3, bh);
  stroke(PARTS[1].color);
  strokeWeight(weight(1));
  seg(st.q1, -bh, st.q1, bh);
  strokeWeight(weight(3));
  seg(st.q3, -bh, st.q3, bh);
  stroke(PARTS[2].color);
  strokeWeight(weight(2));
  seg(st.med, -bh, st.med, bh);

  // IQR bracket beside the box
  const bo = -bh - 10;
  stroke(PARTS[5].color);
  strokeWeight(active === 5 ? 4 : 2);
  seg(st.q1, bo, st.q3, bo);
  seg(st.q1, bo, st.q1, bo + 6);
  seg(st.q3, bo, st.q3, bo + 6);

  // outliers
  stroke(PARTS[6].color);
  strokeWeight(2);
  fill(active === 6 ? PARTS[6].color : 'white');
  for (const v of st.outliers) circle(...P(v, 0), active === 6 ? 14 : 10);

  // numbered callouts: the code gives the side of the badge on which the name is written
  // (v- and v+ along the value axis, o- and o+ across it)
  const ls = narrow ? 13 : 15;
  callout(0, P(st.wLo, 26), 'o+', active, ls);
  callout(1, P(st.q1, -bh, -15), 'v-', active, ls);
  callout(2, P(st.med, bh + 16), 'o+', active, ls);
  callout(3, P(st.q3, -bh, 15), 'v+', active, ls);
  callout(4, P(st.wHi, 26), 'o+', active, ls);
  callout(5, P((st.q1 + st.q3) / 2, bo - 17), 'o-', active, ls);
  for (const v of st.outliers) callout(6, P(v, -22), 'o-', active, ls);
}

// A numbered badge and, when names are shown, the part name next to it. Both are clickable.
function callout(part, pos, code, active, ls) {
  const [bx, by] = pos, r = active === part ? 12 : 10;
  stroke('white');
  strokeWeight(1.5);
  fill(PARTS[part].color);
  circle(bx, by, 2 * r);
  noStroke();
  fill('white');
  textStyle(BOLD);
  textSize(12);
  textAlign(CENTER, CENTER);
  text(part + 1, bx, by + 1);
  targets.push({ part, x0: bx - 13, y0: by - 13, x1: bx + 13, y1: by + 13 });
  if (namesBox.checked()) {
    const dirs = lay.vertical ? { 'v-': [0, 1], 'v+': [0, -1], 'o-': [-1, 0], 'o+': [1, 0] }
      : { 'v-': [-1, 0], 'v+': [1, 0], 'o-': [0, -1], 'o+': [0, 1] };
    const [dx, dy] = dirs[code];
    textSize(ls);
    const label = PARTS[part].label, w = textWidth(label);
    const lx = bx + dx * 15, ly = by + dy * 19;
    fill(PARTS[part].color);
    textAlign(dx < 0 ? RIGHT : dx > 0 ? LEFT : CENTER, CENTER);
    text(label, lx, ly);
    const x0 = dx < 0 ? lx - w : dx > 0 ? lx : lx - w / 2;
    targets.push({ part, x0, y0: ly - 9, x1: x0 + w, y1: ly + 9 });
  }
  textStyle(NORMAL);
}

// The text panel: a summary when no part is active, otherwise the details of the active part
function drawPanel(top, bottom, active, narrow) {
  const x = margin, w = canvasWidth - 2 * margin, ts = narrow ? 12 : 15, lh = ts + 5;
  const tx = x + 12, tw = w - 24;
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, top, w, bottom - top, 10);
  noStroke();
  textAlign(LEFT, TOP);
  textWrap(WORD);

  if (active < 0) {
    fill('black');
    textStyle(BOLD);
    textSize(ts + 2);
    text('What a box plot tells you at a glance', tx, top + 10);
    textStyle(NORMAL);
    textSize(ts);
    const half = st.med - st.q1 === st.q3 - st.med ? 'the middle' : st.med - st.q1 < st.q3 - st.med ? 'nearer Q1' : 'nearer Q3';
    const lines = ['Center: the median line (here ' + fmt(st.med) + ')',
      'Spread: the length of the box (IQR = ' + fmt(st.iqr) + ')',
      'Symmetry: where the median sits in the box (here ' + half + ')',
      'Outliers: the separate points (here ' + st.outliers.map(fmt).join(' and ') + ')'];
    lines.forEach((s, i) => text('•  ' + s, tx, top + 40 + i * (lh + 3)));
    fill('dimgray');
    text('Hover over or click a numbered part to read about it. Uncheck "Show names" to test yourself.',
      tx, top + 44 + 4 * (lh + 3), tw, 2 * lh + 4);
    return;
  }

  const p = PARTS[active];
  fill(p.color);
  textStyle(BOLD);
  textSize(ts + 2);
  text((active + 1) + '. ' + p.name, tx, top + 10);
  textStyle(NORMAL);
  textSize(ts);
  fill('black');
  const descRows = narrow ? 3 : 2, valueRows = narrow ? 2 : 1;
  let y = top + 36;
  text(p.desc, tx, y, tw, descRows * lh + 4);
  y += descRows * lh + 6;
  textStyle(BOLD);
  text('In this data: ' + p.value(st) + '.', tx, y, tw, valueRows * lh + 4);
  textStyle(NORMAL);
  y += valueRows * lh + 10;
  fill('whitesmoke');
  stroke('gainsboro');
  rect(tx - 5, y - 5, tw + 10, lh + 10, 6);
  noStroke();
  fill('darkslategray');
  text('pandas:   ' + p.code, tx + 2, y);
}

function mousePressed() {
  if (mouseY > drawHeight) return;
  const hit = hitPart(mouseX, mouseY);
  selected = hit === selected ? -1 : hit;     // clicking the same part or empty space clears the selection
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
