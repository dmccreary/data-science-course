// Multiple Regression Anatomy
// CANVAS_HEIGHT: 605
// Bloom L2 (Understand): students move three feature sliders and watch each term of
// yhat = b0 + b1 x1 + b2 x2 + b3 x3 change on its own, then hover over or click a numbered part
// (the intercept, the three coefficient-times-feature terms, the prediction) to read what it means.
//
// Model: ordinary least squares with three predictors, fitted inside the sketch by solving the
// normal equations (X'X) b = X'y on centered columns with Gauss-Jordan elimination.
//   contribution of feature j = b_j * x_j          prediction = b0 + sum of the contributions
//   R^2 = 1 - SSE / SST          adjusted R^2 = 1 - (1 - R^2)(n - 1) / (n - p - 1)
// The 40 synthetic houses are built so that the fit returns the chapter's model exactly
// (price = 50000 + 150 square_feet + 10000 bedrooms - 1000 age): the random noise in each price is
// first made orthogonal to the features, so least squares cannot attribute any of it to them.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 490;
let controlHeight = 115;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;
let labelWidth = 150;       // width of the label span in each slider row
let valueWidth = 55;        // width of the value span in each slider row

const SEED = 9, N = 40, NOISE_SD = 25000, AXIS_MAX = 600000;
const TARGET = [50000, 150, 10000, -1000];        // intercept, per square foot, per bedroom, per year
const FEATURES = [
  { name: 'Square feet', unit: 'square foot', min: 800, max: 3000, start: 1500, step: 50 },
  { name: 'Bedrooms', unit: 'bedroom', min: 1, max: 6, start: 3, step: 1 },
  { name: 'Age', unit: 'year of age', min: 0, max: 60, start: 20, step: 1 }
];
// parts 0 to 4: the intercept, the three feature terms, the prediction
const SYMS = ['β₀', 'β₁x₁', 'β₂x₂', 'β₃x₃', 'ŷ'];
const NAMES = ['Intercept', 'Square feet term', 'Bedrooms term', 'Age term', 'Predicted price'];
const SHORT = ['Intercept', 'Square feet', 'Bedrooms', 'Age', 'Prediction ŷ'];

let rngState = 1;
let cols = [[], [], []], price = [];   // the 40 houses: one array per feature, and the prices
let fit = {};                          // least-squares fit to those houses
let rows = [];                         // slider rows
let contrib = [], pred = 0;            // value of each term and their sum for the slider settings
let locked = -1, sel = -1, hits = [];  // clicked part, part shown this frame, hover rectangles

function uniform() {                   // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }

// Least squares with an intercept. xs is a list of feature columns. The columns are centered,
// then (X'X) b = X'y is solved by Gauss-Jordan elimination with partial pivoting.
function ols(xs, y) {
  const n = y.length, p = xs.length, mean = a => a.reduce((s, v) => s + v, 0) / n;
  const mx = xs.map(mean), my = mean(y);
  const A = xs.map((cj, j) => {
    const row = xs.map((ck, k) => cj.reduce((s, v, i) => s + (v - mx[j]) * (ck[i] - mx[k]), 0));
    row.push(cj.reduce((s, v, i) => s + (v - mx[j]) * (y[i] - my), 0));
    return row;
  });
  for (let c = 0; c < p; c++) {
    let piv = c;
    for (let r = c + 1; r < p; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]];
    const d = A[c][c];
    A[c] = A[c].map(v => v / d);
    for (let r = 0; r < p; r++) {
      if (r === c) continue;
      const f = A[r][c];
      A[r] = A[r].map((v, k) => v - f * A[c][k]);
    }
  }
  const b = A.map(row => row[p]), b0 = my - b.reduce((s, v, j) => s + v * mx[j], 0);
  const resid = y.map((v, i) => v - b0 - b.reduce((s, bj, j) => s + bj * xs[j][i], 0));
  const sse = resid.reduce((s, e) => s + e * e, 0), sst = y.reduce((s, v) => s + (v - my) ** 2, 0);
  const r2 = 1 - sse / sst;
  return { b0, b, resid, r2, adjR2: 1 - (1 - r2) * (n - 1) / (n - p - 1) };
}

function makeHouses() {
  rngState = SEED;
  cols = [[], [], []];
  const noise = [];
  for (let i = 0; i < N; i++) {
    const sqft = 800 + 10 * Math.round(220 * uniform());
    cols[0].push(sqft);
    cols[1].push(Math.min(6, Math.max(1, Math.round(sqft / 600 + 0.8 * stdNormal()))));   // bigger houses, more bedrooms
    cols[2].push(Math.floor(61 * uniform()));
    noise.push(NOISE_SD * stdNormal());
  }
  const e = ols(cols, noise).resid;       // the part of the noise the features cannot explain
  price = e.map((v, i) => TARGET[0] + TARGET[1] * cols[0][i] + TARGET[2] * cols[1][i] + TARGET[3] * cols[2][i] + v);
  fit = ols(cols, price);
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));

  makeHouses();
  rows = FEATURES.map((f, j) => makeSliderRow(f.name + ' (x' + '₁₂₃'[j] + ')', f.min, f.max, f.start, f.step, j));
  resizeSliders();

  describe('The multiple regression equation for house price with an intercept and three terms for square feet, ' +
    'bedrooms, and age. Three sliders set the feature values. A waterfall chart shows the intercept and each ' +
    'coefficient times feature contribution adding up to the predicted price. Hovering over or clicking a numbered ' +
    'part explains it and shows its current value.', LABEL);
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
  rows.forEach(r => r.slider.size(w));
}

const num = v => (Math.round(v) < 0 ? '−' : '') + Math.abs(Math.round(v)).toLocaleString('en-US');
const signedMoney = v => (Math.round(v) < 0 ? '−$' : '+$') + Math.abs(Math.round(v)).toLocaleString('en-US');

// Blue intercept, gold prediction, green for a term that raises the prediction and red for one that lowers it.
// Parts other than the selected one are faded.
function tone(i, alpha) {
  const c = color(i === 0 ? 'royalblue' : i === 4 ? 'goldenrod' : contrib[i] < 0 ? 'crimson' : 'seagreen');
  c.setAlpha(sel < 0 || sel === i ? (alpha || 255) : (alpha ? 25 : 110));
  return c;
}

function badge(i, x, y, r) {
  stroke('white');
  strokeWeight(1.5);
  fill(tone(i));
  circle(x, y, 2 * r);
  noStroke();
  fill('white');
  textAlign(CENTER, CENTER);
  textStyle(BOLD);
  textSize(r + 3);
  text(i + 1, x, y + 1);
  textStyle(NORMAL);
}

function partAt(x, y) {
  for (const h of hits) if (x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h) return h.i;
  return -1;
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600, w = canvasWidth - 2 * margin;
  const x = rows.map(r => r.slider.value());
  rows.forEach((r, j) => r.valueSpan.html(x[j].toLocaleString('en-US')));
  contrib = [fit.b0, ...x.map((v, j) => fit.b[j] * v)];
  pred = contrib.reduce((s, v) => s + v, 0);

  const hov = partAt(mouseX, mouseY);
  sel = hov >= 0 ? hov : locked;
  cursor(hov >= 0 ? HAND : ARROW);
  hits = [];
  textWrap(WORD);

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Multiple Regression Anatomy', canvasWidth / 2, 8);

  if (narrow) {
    drawEquation(margin, 36, w, 96, true, x);
    drawWaterfall(margin, 138, w, 190, true);
    drawDetail(margin, 334, w, drawHeight - 342, true, x);
  } else {
    const leftW = Math.round(w * 0.58);
    drawEquation(margin, 44, w, 124, false, x);
    drawWaterfall(margin, 176, leftW, drawHeight - 184, false);
    drawDetail(margin + leftW + 10, 176, w - leftW - 10, drawHeight - 184, false, x);
  }
}

// The equation in symbols with the fitted numbers and the slider values underneath each term
function drawEquation(x0, y0, w, h, narrow, x) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const big = narrow ? 22 : 30, ts = narrow ? 12 : 15, r = narrow ? 9 : 11, cw = w / 5;
  const badgeY = y0 + (narrow ? 34 : 44), symY = y0 + (narrow ? 58 : 76), subY = y0 + (narrow ? 82 : 106);
  noStroke();
  fill('dimgray');
  textSize(ts);
  textAlign(CENTER, TOP);
  text('Least-squares fit to ' + N + ' houses:  R² = ' + nf(fit.r2, 1, 3) + ',  adjusted R² = ' + nf(fit.adjR2, 1, 3), x0 + w / 2, y0 + 7);

  const subs = [num(fit.b0), ...x.map((v, j) => num(fit.b[j]) + ' × ' + num(v)), '$' + num(pred)];
  [4, 0, 1, 2, 3].forEach((i, k) => {                 // ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃
    const cx = x0 + cw * (k + 0.5);
    if (sel === i) {
      noStroke();
      fill(tone(i, 40));
      rect(x0 + cw * k + 6, y0 + (narrow ? 22 : 28), cw - 12, h - (narrow ? 26 : 34), 8);
    }
    badge(i, cx, badgeY, r);
    noStroke();
    fill(tone(i));
    textAlign(CENTER, CENTER);
    textStyle(BOLD);
    textSize(big);
    text(SYMS[i], cx, symY);
    textStyle(i === 4 ? BOLD : NORMAL);
    textSize(narrow ? 11 : ts);
    text(subs[i], cx, subY);
    textStyle(NORMAL);
    if (k > 0) {                                      // the operator to the left of this term
      fill('black');
      textSize(big - 4);
      text(k === 1 ? '=' : '+', x0 + cw * k, symY);
      textSize(ts);
      text(k === 1 ? '=' : '+', x0 + cw * k, subY);
    }
    hits.push({ i, x: x0 + cw * k + 6, y: y0 + 24, w: cw - 12, h: h - 28 });
  });
}

// Waterfall chart: the intercept, then each contribution starting where the last one ended, then the total
function drawWaterfall(x0, y0, w, h, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const ts = narrow ? 12 : 15, r = narrow ? 9 : 11, labelW = narrow ? 112 : 150, valueW = narrow ? 76 : 96;
  const bx = x0 + labelW, bw = w - labelW - valueW, top = y0 + (narrow ? 26 : 32), rowH = (h - (narrow ? 48 : 58)) / 5;
  const gx = v => bx + constrain(v / AXIS_MAX, 0, 1) * bw, bh = Math.min(rowH - 8, 28);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text('How the terms add up to the prediction', x0 + 10, y0 + 7);
  textStyle(NORMAL);

  textSize(ts - 1);
  for (let v = 0; v <= AXIS_MAX; v += 200000) {
    stroke('gainsboro');
    strokeWeight(1);
    line(gx(v), top, gx(v), top + 5 * rowH);
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    text(v ? '$' + v / 1000 + 'k' : '$0', gx(v), top + 5 * rowH + 3);
  }

  let run = 0;
  for (let i = 0; i < 5; i++) {
    const from = i === 4 ? 0 : run, to = i === 4 ? pred : run + contrib[i];
    const ry = top + i * rowH, cy = ry + rowH / 2;
    if (sel === i) {
      noStroke();
      fill(tone(i, 40));
      rect(x0 + 4, ry + 1, w - 8, rowH - 2, 6);
    }
    if (i > 0 && i < 4) {                 // this bar starts where the running total stood
      stroke('gray');
      strokeWeight(1);
      drawingContext.setLineDash([3, 3]);
      line(gx(run), cy - rowH + bh / 2, gx(run), cy - bh / 2);
      drawingContext.setLineDash([]);
    }
    noStroke();
    fill(tone(i));
    rect(Math.min(gx(from), gx(to)), cy - bh / 2, Math.max(2, Math.abs(gx(to) - gx(from))), bh, 3);
    badge(i, x0 + 8 + r, cy, r);
    noStroke();
    fill('black');
    textSize(ts);
    textStyle(i === 4 ? BOLD : NORMAL);
    textAlign(LEFT, CENTER);
    text(SHORT[i], x0 + 14 + 2 * r, cy);
    textAlign(RIGHT, CENTER);
    text(i === 0 || i === 4 ? '$' + num(to) : signedMoney(contrib[i]), x0 + w - 8, cy);
    textStyle(NORMAL);
    if (i < 4) run = to;
    hits.push({ i, x: x0 + 4, y: ry, w: w - 8, h: rowH });
  }
  stroke('gray');
  strokeWeight(1);
  line(x0 + 8, top + 4 * rowH, x0 + w - 8, top + 4 * rowH);     // the sum line above the total
}

// What the selected part means and its value for the current slider settings
function partText(i, x) {
  if (i === 0) return ['The starting point of every prediction: the value the model gives when all three features are 0.',
    'β₀ = $' + num(fit.b0) + '. No house has 0 square feet, so read it as the baseline the other terms add to, not as the price of a real house.'];
  if (i === 4) return ['The prediction is the intercept plus the three contributions. Green terms push it up and red terms pull it down.',
    num(contrib[0]) + contrib.slice(1).map(c => (c < 0 ? ' − ' : ' + ') + num(Math.abs(c))).join('') + ' = $' + num(pred) + '.'];
  const f = FEATURES[i - 1], b = fit.b[i - 1];
  const others = FEATURES.filter(g => g !== f).map(g => g.name.toLowerCase()).join(' and ');
  return ['A coefficient times a feature value. β' + '₁₂₃'[i - 1] + ' = ' + num(b) + ': each extra ' + f.unit +
    (b < 0 ? ' lowers the predicted price by $' : ' raises the predicted price by $') + num(Math.abs(b)) + ', holding ' + others + ' constant.',
    num(b) + ' × ' + num(x[i - 1]) + ' = ' + signedMoney(contrib[i]) + '. Move only the ' + f.name + ' slider and only this term changes.'];
}

function drawDetail(x0, y0, w, h, narrow, x) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const ts = narrow ? 12 : 15, lh = ts + 5, tx = x0 + 12, tw = w - 24;
  noStroke();
  textAlign(LEFT, TOP);
  textSize(ts);
  if (sel < 0) {
    fill('dimgray');
    text('Each term is a coefficient multiplied by a feature value. Move one slider and watch which term changes ' +
      'while the others hold still.\n\nTry it: add one bedroom. The prediction rises by exactly $' + num(fit.b[1]) +
      ', the value of β₂, wherever the other two sliders are set.\n\nHover over a numbered part to read about it. ' +
      'Click it to keep it selected, and click again to release it.', tx, y0 + 12, tw, h - 20);
    return;
  }
  const [body, now] = partText(sel, x), boxH = (narrow ? 3 : 5) * lh + 10;
  fill(tone(sel));
  textStyle(BOLD);
  textSize(ts + 2);
  text((sel + 1) + '. ' + NAMES[sel] + ' (' + SYMS[sel] + ')', tx, y0 + 10);
  textStyle(NORMAL);
  fill('black');
  textSize(ts);
  text(body, tx, y0 + 14 + lh, tw, h - boxH - lh - 24);
  fill('lightyellow');
  stroke('silver');
  strokeWeight(1);
  rect(x0 + 6, y0 + h - boxH - 6, w - 12, boxH, 6);
  noStroke();
  fill('black');
  text('Right now: ' + now, tx, y0 + h - boxH, tw, boxH - 4);
}

function mousePressed() {
  if (mouseY < 0 || mouseY > drawHeight || mouseX < 0 || mouseX > canvasWidth) return;
  const p = partAt(mouseX, mouseY);
  locked = p === locked ? -1 : p;
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
