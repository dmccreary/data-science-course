// Correlation Visualizer
// CANVAS_HEIGHT: 585
// Bloom L4 (Analyze): students compare scatter plots of different strengths and shapes, drag
// points to make outliers, and examine how Pearson r, R squared, and Spearman's rank correlation respond.
//
// Model:
//   Pearson r   = sum((x - mean x)(y - mean y)) / sqrt(sum((x - mean x)^2) sum((y - mean y)^2))
//   R squared   = r^2, the share of the variance in y accounted for by the least-squares line
//   Spearman    = Pearson r of the ranks of x and the ranks of y (tied values share their average rank)
// The linear cloud is built so that its sample r equals the slider value exactly: y is a mix of the
// standardized x values and standardized noise that has been made uncorrelated with x.
// Data come from a seeded generator (mulberry32). New Sample moves to the next seed.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 470;
let controlHeight = 115;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;
let labelWidth = 100;       // width of the label span in each control row
let valueWidth = 50;        // width of the value span in the slider row

const N = 40;               // number of points
const PATTERNS = ['Linear cloud', 'Line with one outlier', 'U-shaped curve', 'Steady curved rise'];
const PEARSON_COLOR = 'royalblue', SPEARMAN_COLOR = 'chocolate';

let patternSelect, rRow, sampleButton, lineBox;
let pts = [];               // the data: { x, y } on a 0..100 scale
let outlierIdx = -1;        // index of the planted outlier, or -1
let sampleSeed = 1;
let rngState = 1;
let plot = {};              // plot rectangle of the last frame, used for dragging
let dragIdx = -1;

function uniform() {        // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function gauss() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }

const meanOf = a => a.reduce((s, v) => s + v, 0) / a.length;
// z-scores of an array (mean 0, mean square 1)
function standardize(a) {
  const m = meanOf(a), sd = Math.sqrt(meanOf(a.map(v => (v - m) * (v - m))));
  return a.map(v => (v - m) / sd);
}

function pearson(x, y) {
  const mx = meanOf(x), my = meanOf(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < x.length; i++) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) * (x[i] - mx);
    syy += (y[i] - my) * (y[i] - my);
  }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : 0;
}

// Ranks from 1 to n. Tied values share the average of the ranks they cover.
function ranks(a) {
  const order = a.map((v, i) => i).sort((i, j) => a[i] - a[j]), r = new Array(a.length);
  for (let i = 0; i < order.length;) {
    let j = i;
    while (j + 1 < order.length && a[order[j + 1]] === a[order[i]]) j++;
    for (let k = i; k <= j; k++) r[order[k]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return r;
}
function spearman(x, y) { return pearson(ranks(x), ranks(y)); }

// n points whose sample correlation is exactly r
function linearCloud(n, r) {
  const zx = standardize(Array.from({ length: n }, gauss));
  let e = standardize(Array.from({ length: n }, gauss));
  const b = meanOf(e.map((v, i) => v * zx[i]));           // remove the part of the noise that follows x
  e = standardize(e.map((v, i) => v - b * zx[i]));
  return zx.map((z, i) => ({ x: 50 + 14 * z, y: 50 + 14 * (r * z + Math.sqrt(1 - r * r) * e[i]) }));
}

function makeData() {
  rngState = 1000 + sampleSeed;
  const k = PATTERNS.indexOf(patternSelect.value()), r = rRow.slider.value();
  outlierIdx = -1;
  if (k === 0) pts = linearCloud(N, r);
  else if (k === 1) {
    pts = linearCloud(N - 1, r);
    pts.push({ x: 93, y: r >= 0 ? 6 : 94 });               // one point far from the trend
    outlierIdx = N - 1;
  } else {
    pts = Array.from({ length: N }, () => {
      const x = 5 + 90 * uniform(), t = (x - 5) / 90;
      return k === 2 ? { x, y: 12 + 0.036 * (x - 50) * (x - 50) + 4 * gauss() }
        : { x, y: 5 + 85 * (Math.exp(6 * t) - 1) / (Math.exp(6) - 1) + 0.4 * gauss() };
    });
  }
  for (const p of pts) { p.x = Math.min(98, Math.max(2, p.x)); p.y = Math.min(98, Math.max(2, p.y)); }
  // the r slider only applies to the two linear patterns
  if (k <= 1) rRow.slider.removeAttribute('disabled'); else rRow.slider.attribute('disabled', '');
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
  const labelSpan = createSpan('Pattern:');
  labelSpan.parent(row);
  labelSpan.style('display', 'inline-block');
  labelSpan.style('width', labelWidth + 'px');
  patternSelect = createSelect();
  patternSelect.parent(row);
  PATTERNS.forEach(name => patternSelect.option(name));
  patternSelect.style('font-size', '15px');
  patternSelect.changed(makeData);

  rRow = makeSliderRow('Target r', -1, 1, 0.8, 0.05, 1);
  rRow.slider.input(makeData);
  rRow.slider.size(max(60, canvasWidth - labelWidth - valueWidth - 40));

  sampleButton = createButton('New Sample');
  sampleButton.parent(mainElement);
  sampleButton.position(10, drawHeight + 80);
  sampleButton.mousePressed(() => { sampleSeed++; makeData(); });
  lineBox = createCheckbox(' Best-fit line', true);
  lineBox.parent(mainElement);
  lineBox.position(120, drawHeight + 80);
  lineBox.style('font-size', '16px');
  makeData();

  describe('A scatter plot of 40 points that can be dragged, with an optional best-fit line. A panel shows Pearson r, ' +
    'R squared, and Spearman rank correlation, with both correlations marked on a scale from minus one to plus one. ' +
    'A menu chooses a linear cloud, a line with one outlier, a U-shaped curve, or a steady curved rise, and a ' +
    'slider sets the correlation of the linear cloud.', LABEL);
}

// A control row built from a div: the label and value sit in fixed-width spans
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

const signedText = (v, d) => (v < 0 ? '−' : '+') + nf(Math.abs(v), 1, d);

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const k = PATTERNS.indexOf(patternSelect.value());
  rRow.valueSpan.html(k <= 1 ? signedText(rRow.slider.value(), 2) : '–');
  const narrow = canvasWidth < 600, w = canvasWidth - 2 * margin;
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  const r = pearson(xs, ys), rho = spearman(xs, ys);

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Correlation Visualizer', canvasWidth / 2, 7);
  fill('dimgray');
  textSize(narrow ? 12 : 14);
  text('Drag any point to see how it changes the correlation.', canvasWidth / 2, narrow ? 31 : 35);

  if (narrow) {
    drawScatter(margin, 50, w, 222, xs, ys, r);
    drawPanel(margin, 278, w, drawHeight - 286, r, rho, k, 12, true);
  } else {
    const plotW = Math.round(w * 0.56);
    drawScatter(margin, 56, plotW, drawHeight - 64, xs, ys, r);
    drawPanel(margin + plotW + 10, 56, w - plotW - 10, drawHeight - 64, r, rho, k, 15, false);
  }
}

function drawScatter(x0, y0, w, h, xs, ys, r) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const px = x0 + 40, py = y0 + 12, pw = w - 54, ph = h - 48;
  const gx = v => px + v / 100 * pw, gy = v => py + ph - v / 100 * ph;
  plot = { px, py, pw, ph };

  textSize(12);
  for (let v = 0; v <= 100; v += 20) {
    stroke('gainsboro');
    strokeWeight(1);
    line(gx(v), py, gx(v), py + ph);
    line(px, gy(v), px + pw, gy(v));
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    text(v, gx(v), py + ph + 5);
    textAlign(RIGHT, CENTER);
    text(v, px - 5, gy(v));
  }
  stroke('gray');
  strokeWeight(1.5);
  line(px, py, px, py + ph);
  line(px, py + ph, px + pw, py + ph);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  text('x', px + pw / 2, py + ph + 20);
  textAlign(CENTER, CENTER);
  text('y', x0 + 9, py + ph / 2);

  // least-squares line y = my + slope (x - mx), clipped to the plot area
  if (lineBox.checked()) {
    const mx = meanOf(xs), my = meanOf(ys);
    const sx = Math.sqrt(meanOf(xs.map(v => (v - mx) * (v - mx)))), sy = Math.sqrt(meanOf(ys.map(v => (v - my) * (v - my))));
    const slope = sx > 0 ? r * sy / sx : 0;
    drawingContext.save();
    drawingContext.beginPath();
    drawingContext.rect(px, py, pw, ph);
    drawingContext.clip();
    stroke('crimson');
    strokeWeight(2.5);
    line(gx(0), gy(my - slope * mx), gx(100), gy(my + slope * (100 - mx)));
    drawingContext.restore();
  }

  const over = dragIdx >= 0 ? dragIdx : pointAt(mouseX, mouseY);
  cursor(over >= 0 ? HAND : ARROW);
  pts.forEach((p, i) => {
    stroke(i === outlierIdx ? 'crimson' : 'white');
    strokeWeight(i === outlierIdx ? 2.5 : 1);
    fill(i === over ? 'gold' : color(65, 105, 225, 200));
    circle(gx(p.x), gy(p.y), i === over ? 13 : 9);
  });
}

// Words for the size of r (a common rule of thumb)
function strength(r) {
  const a = Math.abs(r);
  if (a < 0.1) return 'almost no linear relationship';
  return (a >= 0.995 ? 'perfect ' : a >= 0.7 ? 'strong ' : a >= 0.4 ? 'moderate ' : 'weak ') + (r > 0 ? 'positive' : 'negative');
}

// Statistics panel: Pearson r, a scale from -1 to +1, R squared, Spearman, and one thing to notice
function drawPanel(x, y, w, h, r, rho, k, ts, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const tx = x + 12, tw = w - 24, lh = ts + 4;
  noStroke();
  textAlign(LEFT, TOP);
  textWrap(WORD);
  textStyle(BOLD);
  textSize(ts + 4);
  fill(PEARSON_COLOR);
  const head = 'Pearson r = ' + signedText(r, 2);
  text(head, tx, y + 9);
  const headW = textWidth(head);
  textStyle(NORMAL);
  textSize(ts);
  fill('black');
  if (narrow) text(strength(r), tx + headW + 10, y + 13); else text(strength(r), tx, y + 36);
  let cy = y + (narrow ? 50 : 86);

  // scale from -1 to +1: triangle = Pearson r, ring = Spearman
  const m0 = x + 30, m1 = x + w - 30, mx = v => m0 + (v + 1) / 2 * (m1 - m0);
  fill('mistyrose');
  rect(m0, cy - 5, (m1 - m0) / 2, 10);
  fill('lightcyan');
  rect(mx(0), cy - 5, (m1 - m0) / 2, 10);
  textSize(11);
  for (const v of [-1, -0.5, 0, 0.5, 1]) {
    stroke('gray');
    strokeWeight(1);
    line(mx(v), cy - 5, mx(v), cy + 8);
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    text(v === 0 ? '0' : signedText(v, v % 1 ? 1 : 0), mx(v), cy + 10);
  }
  noFill();
  stroke(SPEARMAN_COLOR);
  strokeWeight(2.5);
  circle(mx(rho), cy, 13);
  noStroke();
  fill(PEARSON_COLOR);
  triangle(mx(r), cy - 5, mx(r) - 7, cy - 17, mx(r) + 7, cy - 17);

  cy += narrow ? 28 : 40;
  textSize(ts);
  textAlign(LEFT, TOP);
  fill('black');
  text('R² = ' + nf(r * r, 1, 2) + ': the best-fit line accounts for ' + round(100 * r * r) + '% of the variance in y.',
    tx, cy, tw, 2 * lh + 2);
  cy += (narrow ? 2 : 2) * lh + (narrow ? 2 : 12);
  textStyle(BOLD);
  fill(SPEARMAN_COLOR);
  text('Spearman ρ = ' + signedText(rho, 2), tx, cy);
  const rhoW = textWidth('Spearman ρ = ' + signedText(rho, 2));
  textStyle(NORMAL);
  fill('black');
  if (narrow) text('(Pearson r of the ranks)', tx + rhoW + 8, cy); else text('Pearson r of the ranks of x and y', tx, cy + lh);
  cy += narrow ? lh + 4 : 2 * lh + 14;

  const gap = Math.abs(r - rho);
  let note;
  if (k === 2 && Math.abs(r) < 0.3) note = 'A clear curve, yet r is near 0. Pearson r measures only straight-line ' +
    'relationships, so always look at the plot.';
  else if (k === 3 && rho - r >= 0.05) note = 'The points rise steadily along a curve. Spearman is near +1 because the ' +
    'rank order is almost perfect. Pearson is lower because the path is not straight.';
  else if (gap >= 0.1) note = 'Pearson and Spearman differ by ' + nf(gap, 1, 2) + '. Spearman uses only rank order, so ' +
    'one extreme point or a steady curve changes it less.';
  else note = 'Pearson and Spearman agree here. Drag one point far from the rest and see which one changes more.';
  fill('dimgray');
  text(note, tx, cy, tw, narrow ? y + h - cy - 4 : 4 * lh + 4);
  if (!narrow) {
    fill('black');
    text('Rule of thumb for the size of r: below 0.1 almost none, 0.1 to 0.4 weak, 0.4 to 0.7 moderate, 0.7 and ' +
      'above strong. The sign gives the direction.', tx, cy + 4 * lh + 16, tw, 4 * lh + 4);
  }
}

// Index of the point nearest the given position, or -1
function pointAt(x, y) {
  let best = -1, bestD = 12;
  pts.forEach((p, i) => {
    const d = dist(x, y, plot.px + p.x / 100 * plot.pw, plot.py + plot.ph - p.y / 100 * plot.ph);
    if (d < bestD) { bestD = d; best = i; }
  });
  return best;
}

function mousePressed() {
  dragIdx = mouseY < drawHeight ? pointAt(mouseX, mouseY) : -1;
}

function mouseDragged() {
  if (dragIdx < 0) return;
  pts[dragIdx].x = constrain((mouseX - plot.px) / plot.pw * 100, 0, 100);
  pts[dragIdx].y = constrain((plot.py + plot.ph - mouseY) / plot.ph * 100, 0, 100);
  return false;
}

function mouseReleased() {
  dragIdx = -1;
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  rRow.slider.size(max(60, canvasWidth - labelWidth - valueWidth - 40));
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
