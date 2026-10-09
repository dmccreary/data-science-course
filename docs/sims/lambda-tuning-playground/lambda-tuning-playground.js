// Lambda Tuning Playground
// CANVAS_HEIGHT: 580
// Bloom L3 and L5 (Apply, Evaluate): students tune λ for a degree-10 polynomial model, judge from the
// fit, the scores, and the coefficients whether it is too small or too large, then check with 5-fold CV.
// Model: 15 training points at evenly spaced x from 0 to 10, y = 2 + 3x − 0.5x² + 0.05x³ + noise
// (the chapter's cubic, noise SD 3). Pipeline: t = (x − 5) / 5, features t, t², ..., t¹⁰, each
// standardized (mean 0, SD 1, ddof = 0), then a penalized linear model with an intercept.
// λ is scikit-learn's alpha, and each method minimizes scikit-learn's objective (n = rows fitted):
//   Ridge(alpha=λ)                       RSS + λ Σβ²                      solved from (XᵀX + λI) β = Xᵀy
//   Lasso(alpha=λ)                       RSS/(2n) + λ Σ|β|
//   ElasticNet(alpha=λ, l1_ratio=0.5)    RSS/(2n) + λ (0.5 Σ|β| + 0.25 Σβ²)
// Lasso and Elastic Net are solved exactly by an active-set method (see activeSet). Coordinate
// descent reaches the same answer but needs millions of passes on these highly correlated powers.
// Scores: R² = 1 − Σ(y − ŷ)² / Σ(y − mean)². CV R² is the R² of the out-of-fold predictions from
// 5 folds (every fifth point), with the scaler refit inside each fold. Test R² uses 200 new points.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 500;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let sliderLeftMargin = 170;
let defaultTextSize = 16;

const N = 15, N_TEST = 200, DEGREE = 10, FOLDS = 5, SIGMA = 3, LOG_LO = -4, STEP = 0.1, GRID_N = 71;
const alphaAt = g => Math.pow(10, LOG_LO + STEP * g);
const truth = x => 2 + 3 * x - 0.5 * x * x + 0.05 * x * x * x;
// rho is scikit-learn's l1_ratio: the share of the penalty that is L1
const METHODS = [{ name: 'Ridge', rho: 0, code: a => 'Ridge(alpha=' + a + ')' },
  { name: 'Lasso', rho: 1, code: a => 'Lasso(alpha=' + a + ')' },
  { name: 'Elastic Net', rho: 0.5, code: a => 'ElasticNet(alpha=' + a + ', l1_ratio=0.5)' }];

let rngState = 1, seed = 3;
let xs = [], ys = [], xTest = [], yTest = [];
let results = [];                 // one entry per method, built on first use and cleared by New Data
let lambdaSlider, methodSelect, bestButton, dataButton;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }

function makeData() {
  rngState = seed;
  xs = Array.from({ length: N }, (_, i) => 10 * i / (N - 1));
  ys = xs.map(x => truth(x) + SIGMA * stdNormal());
  xTest = Array.from({ length: N_TEST }, () => 10 * uniform());
  yTest = xTest.map(x => truth(x) + SIGMA * stdNormal());
  results = [];
}

function features(x) {                  // t, t², ..., t¹⁰ with t = (x − 5) / 5
  const t = (x - 5) / 5;
  return Array.from({ length: DEGREE }, (_, k) => Math.pow(t, k + 1));
}

// Solve A x = b by Gaussian elimination with partial pivoting
function solve(A, b) {
  const n = b.length, M = A.map((row, i) => row.concat(b[i]));
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) x[r] = (M[r][n] - x.reduce((s, v, k) => s + (k > r ? M[r][k] * v : 0), 0)) / M[r][r];
  return x;
}

// Standardize the features of the given rows and form G = ZᵀZ and c = Zᵀ(y − mean)
function prepare(rows) {
  const F = rows.map(i => features(xs[i])), m = rows.length;
  const mu = F[0].map((_, j) => F.reduce((s, r) => s + r[j], 0) / m);
  const sd = F[0].map((_, j) => Math.sqrt(F.reduce((s, r) => s + (r[j] - mu[j]) ** 2, 0) / m));
  const Z = F.map(r => r.map((v, j) => (v - mu[j]) / sd[j])), yMean = rows.reduce((s, i) => s + ys[i], 0) / m;
  const G = mu.map((_, j) => mu.map((_, k) => Z.reduce((s, r) => s + r[j] * r[k], 0)));
  const c = mu.map((_, j) => Z.reduce((s, r, i) => s + r[j] * (ys[rows[i]] - yMean), 0));
  return { G, c, mu, sd, yMean, m };
}

// Minimize ½ βᵀ(G + t2 I)β − cᵀβ + t1 Σ|β| exactly. Starting from a solution for a larger penalty:
// solve for the active (non-zero) coefficients with their signs held fixed, stop at the first
// coefficient that reaches zero and remove it, then add the inactive feature that most violates
// |c_j − Σ G_jk β_k| ≤ t1. When nothing violates it, the optimality conditions hold.
function activeSet(G, c, t1, t2, start) {
  const b = start.slice(), n = b.length;
  for (let it = 0; it < 400; it++) {
    const A = b.map((_, j) => j).filter(j => b[j] !== 0);
    if (A.length) {
      const target = solve(A.map(j => A.map(k => G[j][k] + (j === k ? t2 : 0))), A.map(j => c[j] - t1 * Math.sign(b[j])));
      let step = 1, hit = -1;
      A.forEach((j, i) => {
        if (Math.sign(target[i]) === Math.sign(b[j])) return;
        const fraction = b[j] / (b[j] - target[i]);
        if (fraction < step) { step = fraction; hit = j; }
      });
      A.forEach((j, i) => { b[j] += step * (target[i] - b[j]); });
      if (hit >= 0) { b[hit] = 0; continue; }
    }
    let worst = -1, size = t1 * (1 + 1e-10), sign = 0;
    for (let j = 0; j < n; j++) {
      if (b[j] !== 0) continue;
      let g = c[j];
      for (const k of A) g -= G[j][k] * b[k];
      if (Math.abs(g) > size) { size = Math.abs(g); worst = j; sign = Math.sign(g); }
    }
    if (worst < 0) break;
    b[worst] = sign * 1e-300;         // enters with the sign of its correlation; the next solve sets its size
  }
  return b;
}

// The coefficients at every λ of the grid for one set of rows. Largest λ first, each fit starting from the last.
function pathOn(rows, rho) {
  const p = prepare(rows), path = [];
  let b = new Array(DEGREE).fill(0);
  for (let g = GRID_N - 1; g >= 0; g--) {
    const a = alphaAt(g);
    path[g] = b = rho === 0 ? solve(p.G.map((row, j) => row.map((v, k) => v + (j === k ? a : 0))), p.c)
      : activeSet(p.G, p.c, p.m * a * rho, p.m * a * (1 - rho), b);
  }
  return { p, path };
}
function predictAt(p, b, x) { return p.yMean + features(x).reduce((s, v, j) => s + (v - p.mu[j]) / p.sd[j] * b[j], 0); }
function rSquared(y, yHat) {
  const m = y.reduce((s, v) => s + v, 0) / y.length;
  return 1 - y.reduce((s, v, i) => s + (v - yHat[i]) ** 2, 0) / y.reduce((s, v) => s + (v - m) ** 2, 0);
}

// Fit one method at every λ on all rows and on each training fold, then score it
function analyze(rho) {
  const all = xs.map((_, i) => i), full = pathOn(all, rho);
  const outOfFold = Array.from({ length: GRID_N }, () => new Array(N));
  for (let f = 0; f < FOLDS; f++) {
    const fold = pathOn(all.filter(i => i % FOLDS !== f), rho);
    for (let g = 0; g < GRID_N; g++) for (let i = f; i < N; i += FOLDS) outOfFold[g][i] = predictAt(fold.p, fold.path[g], xs[i]);
  }
  const r = { p: full.p, path: full.path };
  r.train = r.path.map(b => rSquared(ys, xs.map(x => predictAt(r.p, b, x))));
  r.test = r.path.map(b => rSquared(yTest, xTest.map(x => predictAt(r.p, b, x))));
  r.cv = outOfFold.map(pred => rSquared(ys, pred));
  r.best = r.cv.indexOf(Math.max(...r.cv));
  return r;
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  lambdaSlider = createSlider(0, GRID_N - 1, 5, 1);
  lambdaSlider.parent(mainElement);
  lambdaSlider.position(sliderLeftMargin, drawHeight + 8);
  lambdaSlider.size(canvasWidth - sliderLeftMargin - margin);

  methodSelect = createSelect();
  methodSelect.parent(mainElement);
  methodSelect.position(10, drawHeight + 45);
  METHODS.forEach(m => methodSelect.option(m.name));
  methodSelect.style('font-size', '15px');

  bestButton = createButton('Find Best λ');
  bestButton.parent(mainElement);
  bestButton.position(125, drawHeight + 45);
  bestButton.mousePressed(() => lambdaSlider.value(current().best));

  dataButton = createButton('New Data');
  dataButton.parent(mainElement);
  dataButton.position(225, drawHeight + 45);
  dataButton.mousePressed(() => { seed++; makeData(); });

  makeData();

  describe('A scatter plot of fifteen data points with a degree ten polynomial fit that changes as a slider sets lambda. ' +
    'A panel reports training and cross-validation R squared and warns of overfitting or underfitting. A chart shows ' +
    'both scores across all values of lambda with the best cross-validation value starred, and a bar chart shows the ' +
    'size of the ten coefficients. A menu chooses ridge, lasso, or elastic net, and buttons find the best lambda or draw new data.', LABEL);
}

function current() {
  const index = METHODS.findIndex(m => m.name === methodSelect.value());
  if (!results[index]) results[index] = analyze(METHODS[index].rho);
  return Object.assign(results[index], { method: METHODS[index] });
}

const signed = (v, digits) => (Math.abs(v) < 0.5 * Math.pow(10, -digits) ? 0 : v).toFixed(digits).replace('-', '−');
const showAlpha = a => Number(a.toPrecision(3)).toString();
const showR2 = v => v < -9.99 ? 'below −10' : signed(v, 3);

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600, w = canvasWidth - 2 * margin;
  const r = current(), sel = lambdaSlider.value();
  noStroke();
  fill('black');
  textStyle(NORMAL);
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Lambda Tuning Playground', canvasWidth / 2, 8);

  if (narrow) {
    drawFit(margin, 34, w, 150, r, sel, narrow);
    drawScores(margin, 190, w, 112, r, sel, narrow);
    drawCurve(margin, 308, (w - 6) / 2, drawHeight - 316, r, sel, narrow);
    drawBars(margin + (w + 6) / 2, 308, (w - 6) / 2, drawHeight - 316, r, sel, narrow);
  } else {
    drawFit(margin, 42, w * 0.56, 212, r, sel, narrow);
    drawScores(margin + w * 0.56 + 8, 42, w * 0.44 - 8, 212, r, sel, narrow);
    drawCurve(margin, 262, (w - 8) / 2, drawHeight - 270, r, sel, narrow);
    drawBars(margin + (w + 8) / 2, 262, (w - 8) / 2, drawHeight - 270, r, sel, narrow);
  }

  // control label
  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('λ (alpha) = ' + showAlpha(alphaAt(sel)), 10, drawHeight + 18);
}

function drawPanelBox(x, y, w, h, title, ts) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text(title, x + 10, y + 7);
  textStyle(NORMAL);
  textSize(ts);
}

// A horizontal gridline with its value at the left
function gridLine(px, pw, y, label, col) {
  stroke(col || 'gainsboro');
  strokeWeight(1);
  line(px, y, px + pw, y);
  noStroke();
  fill('dimgray');
  textAlign(RIGHT, CENTER);
  text(label, px - 4, y);
}

// The zone of the chosen λ: within 0.02 of the best CV R² is the sweet spot
function zoneOf(r, sel) {
  if (r.cv[sel] >= r.cv[r.best] - 0.02) return { name: 'Sweet spot', ink: 'darkgreen', curve: 'forestgreen' };
  return sel < r.best ? { name: 'Overfitting warning', ink: 'firebrick', curve: 'crimson' }
    : { name: 'Underfitting warning', ink: 'chocolate', curve: 'darkorange' };
}

// The data, the true curve, and the fitted polynomial at the chosen λ
function drawFit(x0, y0, w, h, r, sel, narrow) {
  const ts = narrow ? 11 : 13, zone = zoneOf(r, sel);
  drawPanelBox(x0, y0, w, h, r.method.name + ' fit, degree ' + DEGREE + (narrow ? '' : ' polynomial') + '  (dashed: true curve)', ts);
  const px = x0 + 34, py = y0 + 28, pw = w - 46, ph = h - 50, yLo = -10, yHi = 45;
  const gx = v => px + v / 10 * pw, gy = v => py + ph - (constrain(v, yLo - 40, yHi + 40) - yLo) / (yHi - yLo) * ph;
  textSize(narrow ? 11 : 12);
  for (let v = yLo; v <= yHi; v += 10) gridLine(px, pw, gy(v), v);
  textAlign(CENTER, TOP);
  for (let v = 0; v <= 10; v += 2) text(v, gx(v), py + ph + 3);
  stroke('gray');
  strokeWeight(1.5);
  line(px, py, px, py + ph);
  line(px, py + ph, px + pw, py + ph);

  push();
  drawingContext.beginPath();
  drawingContext.rect(px, py, pw, ph);
  drawingContext.clip();
  const curve = fn => { beginShape(); for (let i = 0; i <= 200; i++) vertex(gx(i / 20), gy(fn(i / 20))); endShape(); };
  noFill();
  stroke('dimgray');
  strokeWeight(1.5);
  drawingContext.setLineDash([4, 3]);
  curve(truth);
  drawingContext.setLineDash([]);
  stroke(zone.curve);
  strokeWeight(3);
  curve(x => predictAt(r.p, r.path[sel], x));
  pop();
  stroke('white');
  strokeWeight(1);
  fill('royalblue');
  xs.forEach((x, i) => circle(gx(x), gy(ys[i]), narrow ? 7 : 9));
}

// Scores at the chosen λ, the scikit-learn call that gives them, and a verdict
function drawScores(x0, y0, w, h, r, sel, narrow) {
  const ts = narrow ? 12 : 14, lh = ts + (narrow ? 4 : 6), ix = x0 + 10, iw = w - 20, zone = zoneOf(r, sel);
  const nonZero = r.path[sel].filter(v => v !== 0).length, atBest = sel === r.best;
  drawPanelBox(x0, y0, w, h, r.method.code(showAlpha(alphaAt(sel))), ts);
  let y = y0 + 9 + lh;
  const half = narrow ? iw / 2 : 0;
  fill('mediumblue');
  text('Training R²: ' + showR2(r.train[sel]), ix, y);
  fill('darkgreen');
  text('CV R² (5-fold): ' + showR2(r.cv[sel]), ix + half, narrow ? y : y + lh);
  y += narrow ? lh : 2 * lh;
  fill('black');
  text('Non-zero coefficients: ' + nonZero + ' of ' + DEGREE, ix, y);
  fill(atBest ? 'black' : 'gray');
  text(atBest ? 'Test R² (200 new points): ' + showR2(r.test[sel]) : narrow ? 'Test R²: shown at best λ' : 'Test R²: shown at the CV-best λ',
    ix + half, narrow ? y : y + lh);
  y += (narrow ? lh : 2 * lh) + 4;

  const msg = {
    'Sweet spot': atBest ? 'This λ has the highest cross-validation score, so it is the one to keep. The test score is the final check.'
      : 'CV R² is within 0.02 of its best value. Press Find Best λ to see the exact peak.',
    'Overfitting warning': 'λ is too small. The penalty is too weak to stop the curve from chasing the noise. Training R² is ' +
      showR2(r.train[sel]) + (r.cv[sel] < 0 ? ', but on held-out points the model does worse than predicting the mean.'
        : ', but CV R² is only ' + showR2(r.cv[sel]) + '.') + ' Increase λ.',
    'Underfitting warning': 'λ is too large. ' + (nonZero === 0 ? 'Every coefficient is 0, so the model predicts the mean.'
      : 'The penalty now holds the curve back from the pattern, and CV R² is ' + signed(r.cv[r.best] - r.cv[sel], 2) + ' below its best.') + ' Decrease λ.'
  }[zone.name];
  textWrap(WORD);
  fill(zone.ink);
  if (narrow) {
    text(zone.name + ': ' + msg, ix, y, iw, y0 + h - y - 2);
    return;
  }
  textStyle(BOLD);
  textSize(ts + 2);
  text(zone.name, ix, y);
  textStyle(NORMAL);
  textSize(ts);
  fill('black');
  text(msg, ix, y + lh + 2, iw, y0 + h - y - lh - 4);
}

// Training and cross-validation R² at every λ, with the chosen λ and the best one marked
function drawCurve(x0, y0, w, h, r, sel, narrow) {
  const ts = narrow ? 11 : 12;
  drawPanelBox(x0, y0, w, h, narrow ? 'R² against λ' : 'Score against λ', ts + 1);
  const px = x0 + (narrow ? 30 : 38), py = y0 + (narrow ? 44 : 30), pw = x0 + w - 12 - px, ph = y0 + h - (narrow ? 30 : 34) - py;
  const yLo = -0.2, gx = g => px + g / (GRID_N - 1) * pw, gy = v => py + ph - (Math.max(v, yLo - 0.5) - yLo) / (1 - yLo) * ph;
  drawLegend([['train', 'royalblue'], ['CV', 'forestgreen'], ['best', 'star']], narrow ? x0 + 10 : x0 + w - 150, y0 + (narrow ? 32 : 15));
  for (let v = 0; v <= 1; v += 0.25) gridLine(px, pw, gy(v), narrow && v % 0.5 ? '' : v, v === 0 ? 'gray' : 'gainsboro');
  textAlign(CENTER, TOP);
  for (let e = LOG_LO; e <= 3; e++) {
    if (narrow && e % 2 !== 0) continue;
    text(e < 0 ? '0.' + '0'.repeat(-e - 1) + '1' : String(Math.pow(10, e)), gx((e - LOG_LO) / STEP), py + ph + 3);
  }
  fill('black');
  text('λ (log scale)', px + pw / 2, py + ph + (narrow ? 15 : 17));

  push();
  drawingContext.beginPath();
  drawingContext.rect(px, py - 2, pw + 2, ph + 2);
  drawingContext.clip();
  noFill();
  strokeWeight(2.5);
  for (const [scores, col] of [[r.train, 'royalblue'], [r.cv, 'forestgreen']]) {
    stroke(col);
    beginShape();
    for (let g = 0; g < GRID_N; g++) vertex(gx(g), gy(scores[g]));
    endShape();
  }
  pop();
  stroke('black');
  strokeWeight(1.5);
  drawingContext.setLineDash([5, 4]);
  line(gx(sel), py, gx(sel), py + ph);
  drawingContext.setLineDash([]);
  stroke('white');
  strokeWeight(1);
  fill('royalblue');
  circle(gx(sel), gy(r.train[sel]), 9);
  if (r.cv[sel] >= yLo) { fill('forestgreen'); circle(gx(sel), gy(r.cv[sel]), 9); }
  stroke('black');
  fill('gold');
  star(gx(r.best), gy(r.cv[r.best]), 8);
}

function star(cx, cy, radius) {
  beginShape();
  for (let i = 0; i < 10; i++) vertex(cx + (i % 2 ? 0.45 : 1) * radius * sin(i * PI / 5), cy - (i % 2 ? 0.45 : 1) * radius * cos(i * PI / 5));
  endShape(CLOSE);
}

// Legend entries [label, color] in a row. The color 'star' draws the marker of the best λ.
function drawLegend(items, x, y) {
  textAlign(LEFT, CENTER);
  for (const [label, col] of items) {
    noStroke();
    fill(col === 'star' ? 'goldenrod' : col);
    if (col === 'star') star(x + 6, y, 6); else rect(x, y - 5, 11, 10);
    fill('black');
    text(label, x + 16, y);
    x += textWidth(label) + 28;
  }
}

// Sizes of the ten standardized coefficients on a log scale. A coefficient that is exactly 0 has no bar.
function drawBars(x0, y0, w, h, r, sel, narrow) {
  const ts = narrow ? 11 : 12, b = r.path[sel];
  drawPanelBox(x0, y0, w, h, narrow ? 'Coefficient size' : 'Coefficient size |β| (log scale)', ts + 1);
  const px = x0 + (narrow ? 30 : 38), py = y0 + (narrow ? 44 : 30), pw = x0 + w - 12 - px, ph = y0 + h - (narrow ? 30 : 34) - py;
  const eLo = -2, eHi = 3, gy = v => py + ph - constrain((Math.log10(v) - eLo) / (eHi - eLo), 0, 1) * ph;
  drawLegend([['positive', 'royalblue'], ['negative', 'tomato']], narrow ? x0 + 10 : x0 + w - 150, y0 + (narrow ? 32 : 15));
  for (let e = eLo; e <= eHi; e++) gridLine(px, pw, gy(Math.pow(10, e)), narrow && e % 2 === 0 ? '' : Math.pow(10, e));
  stroke('gray');
  strokeWeight(1.5);
  line(px, py + ph, px + pw, py + ph);
  const slot = pw / DEGREE;
  for (let j = 0; j < DEGREE; j++) {
    const cx = px + (j + 0.5) * slot, zero = b[j] === 0;
    noStroke();
    fill(b[j] > 0 ? 'royalblue' : 'tomato');
    if (!zero) rect(cx - slot * 0.35, Math.min(gy(Math.abs(b[j])), py + ph - 2), slot * 0.7, Math.max(py + ph - gy(Math.abs(b[j])), 2));
    fill(zero ? 'gray' : 'black');
    textAlign(CENTER, TOP);
    text(j + 1, cx, py + ph + 3);
    if (zero) { textAlign(CENTER, BOTTOM); text('0', cx, py + ph - 2); }
  }
  fill('black');
  textAlign(CENTER, TOP);
  text('power of t', px + pw / 2, py + ph + (narrow ? 15 : 17));
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  lambdaSlider.size(canvasWidth - sliderLeftMargin - margin);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
