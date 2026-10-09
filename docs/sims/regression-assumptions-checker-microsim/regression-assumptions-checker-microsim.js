// Regression Assumptions Checker
// CANVAS_HEIGHT: 580
// Bloom L4 (Analyze): students examine four coordinated diagnostic plots for five data sets,
// decide which regression assumption each one violates, and then compare their reading with a
// rule-based diagnosis.
//
// Model: ordinary least squares fit, residual e = y - yhat, s = sqrt(SSE / (n - 2)), standardized
// residual z = e / s. Rules behind the lights (0 green, 1 yellow, 2 red):
//   Linearity         share of the residual variation explained by adding an x^2 term (squared
//                     correlation of e with x^2 after its straight-line part is removed):
//                     green below 0.10, yellow below 0.25, red otherwise
//   Homoscedasticity  spread ratio: interquartile range of e in the third of the points with the
//                     largest fitted values, divided by the IQR in the third with the smallest.
//                     With r = the larger of ratio and 1/ratio: green below 2, yellow below 2.5, red otherwise
//   Normality         Jarque-Bera statistic n/6 (skew^2 + kurt^2 / 4), kurt = excess kurtosis:
//                     green below 9.21 (chi-square with 2 df, p = 0.01), yellow below 25, red otherwise
//   Independence      cannot be judged from these plots
// Q-Q plot: sorted z against standard normal quantiles at probabilities (i - 0.5) / n.
// Data come from a seeded generator (mulberry32). New Sample moves to the next seed.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 500;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const N = 90;
const SETS = ['Good data', 'Non-linear', 'Heteroscedastic', 'Non-normal residuals', 'Outliers present'];
const FIRST_SEED = [1, 1, 1, 1, 1];        // first sample of each data set
const ROWS = [
  { name: 'Linearity', ask: 'Do the residuals bend in a curve, or scatter evenly around 0?',
    say: ['No curve in the residual plot: a straight line fits.', 'A slight bend in the residual plot. Keep an eye on it.',
      'The residuals bend in a curve: a straight line misses it.'] },
  { name: 'Independence', ask: 'Not visible in any plot. Ask how the data were collected.', say: [] },
  { name: 'Homoscedasticity', ask: 'Is the up-and-down spread the same from left to right?',
    say: ['The residuals have about the same spread everywhere.', 'The spread about doubles from one side to the other.',
      'Funnel shape: the spread changes with the fitted value.'] },
  { name: 'Normality', ask: 'Is the histogram bell-shaped? Do Q-Q points follow the line?',
    say: ['Bell-shaped histogram, and the Q-Q points follow the line.', 'Some Q-Q points drift off the line. Mild departure.',
      'Q-Q points leave the line: skewed residuals or outliers.'] }
];
const LEVELS = [['OK', 'green', 'limegreen'], ['WATCH', 'darkgoldenrod', 'gold'], ['PROBLEM', 'firebrick', 'red']];

let setSelect, sampleButton, showBox;
let sampleNo = 0;
let A = null;               // analysis of the current data
let rngState = 1;

function uniform() {        // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function gauss() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }

// Data set k: x is uniform on 0..10 and the error term carries the violation
function makeData(k, seed) {
  rngState = 7919 * (k + 1) + seed;
  const xs = Array.from({ length: N }, () => 10 * uniform());
  const ys = xs.map(x => {
    if (k === 1) return 12 + 0.75 * x * x + 5 * gauss();                 // curved
    if (k === 2) return 20 + 5 * x + (1 + 1.2 * x) * gauss();            // spread grows with x
    if (k === 3) return 20 + 5 * x + 7 * (-Math.log(uniform()) - 1);     // right-skewed (exponential) errors
    return 20 + 5 * x + 5 * gauss();
  });
  if (k === 4) for (const sign of [1, -1]) ys[Math.floor(N * uniform())] += sign * (40 + 10 * uniform());
  return { xs, ys };
}

const meanOf = a => a.reduce((s, v) => s + v, 0) / a.length;
function ols(xs, ys) {
  const mx = meanOf(xs), my = meanOf(ys);
  let sxy = 0, sxx = 0;
  for (let i = 0; i < xs.length; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) * (xs[i] - mx); }
  const b1 = sxy / sxx, b0 = my - b1 * mx;
  return { b0, b1, res: xs.map((x, i) => ys[i] - (b0 + b1 * x)) };
}
function corr(a, b) {
  const ma = meanOf(a), mb = meanOf(b);
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
  return sab / Math.sqrt(saa * sbb);
}
// Quantile with linear interpolation between order statistics (the numpy default)
function quantile(a, p) {
  const s = a.slice().sort((u, v) => u - v), h = (s.length - 1) * p, lo = Math.floor(h);
  return s[lo] + (h - lo) * (s[Math.min(lo + 1, s.length - 1)] - s[lo]);
}
// Inverse of the standard normal CDF (Acklam's rational approximation, relative error below 1.2e-9)
function normalQuantile(p) {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  if (p > 1 - 0.02425) return -normalQuantile(1 - p);
  if (p < 0.02425) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

// Fit the line and compute everything the plots and the lights need
function analyze(xs, ys) {
  const n = xs.length, f = ols(xs, ys), fitted = xs.map(x => f.b0 + f.b1 * x);
  const sse = f.res.reduce((s, e) => s + e * e, 0), sd = Math.sqrt(sse / (n - 2)), z = f.res.map(e => e / sd);
  const quad = ols(xs, xs.map(x => x * x));                  // x^2 with its straight-line part removed
  const bend = ols(quad.res, f.res).b1;                      // residual trend = bend * (that curved part)
  // the points in three equal groups from the smallest to the largest fitted value, with the quartiles of e in each
  const order = fitted.map((v, i) => i).sort((i, j) => fitted[i] - fitted[j]);
  const thirds = [0, 1, 2].map(t => {
    const idx = order.slice(t * n / 3, (t + 1) * n / 3), e = idx.map(i => f.res[i]);
    return { lo: fitted[idx[0]], hi: fitted[idx[idx.length - 1]], q1: quantile(e, 0.25), q3: quantile(e, 0.75) };
  });
  const spread = (thirds[2].q3 - thirds[2].q1) / (thirds[0].q3 - thirds[0].q1);
  const m2 = sse / n, skew = meanOf(f.res.map(e => e ** 3)) / m2 ** 1.5, kurt = meanOf(f.res.map(e => e ** 4)) / (m2 * m2) - 3;
  const curve = corr(f.res, quad.res) ** 2;
  const jb = n / 6 * (skew * skew + kurt * kurt / 4);
  const level = (v, lo, hi) => v < lo ? 0 : v < hi ? 1 : 2;
  return { xs, ys, n, b0: f.b0, b1: f.b1, res: f.res, fitted, sd, z, quad, bend, thirds, curve, spread, skew, kurt, jb,
    lights: [level(curve, 0.10, 0.25), -1, level(Math.max(spread, 1 / spread), 2, 2.5), level(jb, 9.21, 25)],
    stats: ['curve share ' + Math.round(100 * curve) + '%', '', 'spread right ÷ left = ' + spread.toFixed(2),
      'skew ' + signed(skew) + ', excess kurtosis ' + signed(kurt)] };
}
const signed = v => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(2);

function newData() {
  const k = SETS.indexOf(setSelect.value()), d = makeData(k, FIRST_SEED[k] + sampleNo);
  A = analyze(d.xs, d.ys);
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
  row.style('white-space', 'nowrap');
  createSpan('Data set: ').parent(row);
  setSelect = createSelect();
  setSelect.parent(row);
  SETS.forEach(s => setSelect.option(s));
  setSelect.style('font-size', '15px');
  setSelect.changed(newData);
  sampleButton = createButton('New Sample');
  sampleButton.parent(row);
  sampleButton.style('margin-left', '10px');
  sampleButton.mousePressed(() => { sampleNo++; newData(); });
  showBox = createCheckbox(' Show diagnosis', false);
  showBox.parent(mainElement);
  showBox.position(10, drawHeight + 45);
  showBox.style('font-size', '16px');
  newData();

  describe('Four diagnostic plots of a simple linear regression on 90 points: the data with the fitted line, residuals ' +
    'against fitted values, a histogram of standardized residuals with a normal curve, and a normal Q-Q plot. A menu ' +
    'chooses data that meet the assumptions or that are curved, heteroscedastic, skewed, or contain outliers. A ' +
    'checkbox reveals green, yellow, or red lights for linearity, homoscedasticity, and normality.', LABEL);
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600, w = canvasWidth - 2 * margin, show = showBox.checked();
  textWrap(WORD);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Regression Assumptions Checker', canvasWidth / 2, 8);

  const top = narrow ? 36 : 44, gridW = narrow ? w : Math.round(w * 0.64);
  const cw = (gridW - 6) / 2, ch = narrow ? 138 : (drawHeight - top - 14) / 2;
  drawScatter(margin, top, cw, ch, show, narrow);
  drawResiduals(margin + cw + 6, top, cw, ch, show, narrow);
  drawHistogram(margin, top + ch + 6, cw, ch, narrow);
  drawQQ(margin + cw + 6, top + ch + 6, cw, ch, show, narrow);
  if (narrow) drawPanel(margin, top + 2 * ch + 10, w, drawHeight - (top + 2 * ch + 10) - 6, show, true);
  else drawPanel(margin + gridW + 8, top, w - gridW - 8, drawHeight - top - 8, show, false);
}

const tick = v => Math.abs(v) < 10 && v % 1 !== 0 ? nf(v, 1, 1) : String(Math.round(v)).replace('-', '−');

// A small plot: panel, title, grid lines and labels at the given ticks. Returns the data-to-pixel maps.
function frame(x0, y0, w, h, title, xt, yt, xName, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 8);
  const px = x0 + (narrow ? 30 : 40), py = y0 + (narrow ? 22 : 28), pw = w - (narrow ? 40 : 54), ph = h - (narrow ? 52 : 64);
  const gx = v => px + (v - xt[0]) / (xt[xt.length - 1] - xt[0]) * pw;
  const gy = v => py + ph - (v - yt[0]) / (yt[yt.length - 1] - yt[0]) * ph;
  noStroke();
  fill('black');
  textStyle(BOLD);
  textSize(narrow ? 11 : 13);
  textAlign(LEFT, TOP);
  text(title, x0 + 8, y0 + 6);
  textStyle(NORMAL);
  textSize(narrow ? 11 : 12);
  for (const v of xt) {
    stroke('gainsboro');
    strokeWeight(1);
    line(gx(v), py, gx(v), py + ph);
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    text(tick(v), gx(v), py + ph + 3);
  }
  for (const v of yt) {
    stroke('gainsboro');
    strokeWeight(1);
    line(px, gy(v), px + pw, gy(v));
    noStroke();
    fill('dimgray');
    textAlign(RIGHT, CENTER);
    text(tick(v), px - 4, gy(v));
  }
  stroke('gray');
  strokeWeight(1);
  line(px, py, px, py + ph);
  line(px, py + ph, px + pw, py + ph);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  text(xName, px + pw / 2, py + ph + (narrow ? 15 : 18));
  return { gx, gy, px, py, pw, ph };
}

function dots(xs, ys, g, narrow) {
  noStroke();
  fill(40, 100, 160, 170);
  xs.forEach((x, i) => circle(g.gx(x), g.gy(ys[i]), narrow ? 5.5 : 7.5));
}
// A red ring around every point whose standardized residual is beyond 3
function rings(xs, ys, zs, g) {
  noFill();
  stroke('red');
  strokeWeight(2);
  xs.forEach((x, i) => { if (Math.abs(zs[i]) > 3) circle(g.gx(x), g.gy(ys[i]), 15); });
}

function drawScatter(x0, y0, w, h, show, narrow) {
  const lo = Math.floor(Math.min(...A.ys) / 20) * 20, hi = Math.ceil(Math.max(...A.ys) / 20) * 20;
  const g = frame(x0, y0, w, h, 'Data and fitted line', [0, 5, 10], [lo, (lo + hi) / 2, hi], 'x', narrow);
  stroke('crimson');
  strokeWeight(2.5);
  line(g.gx(0), g.gy(A.b0), g.gx(10), g.gy(A.b0 + 10 * A.b1));
  dots(A.xs, A.ys, g, narrow);
  if (show) rings(A.xs, A.ys, A.z, g);
}

function drawResiduals(x0, y0, w, h, show, narrow) {
  const lo = Math.floor(Math.min(...A.fitted) / 10) * 10, hi = Math.ceil(Math.max(...A.fitted) / 10) * 10;
  const R = Math.ceil(Math.max(...A.res.map(Math.abs)) / 5) * 5;
  const g = frame(x0, y0, w, h, 'Residuals vs fitted values', [lo, (lo + hi) / 2, hi], [-R, 0, R], 'fitted value ŷ', narrow);
  stroke('gray');
  strokeWeight(1.5);
  drawingContext.setLineDash([5, 4]);
  line(g.px, g.gy(0), g.px + g.pw, g.gy(0));
  drawingContext.setLineDash([]);
  dots(A.fitted, A.res, g, narrow);
  if (!show) return;
  rings(A.fitted, A.res, A.z, g);
  noFill();
  strokeWeight(2.5);
  if (A.lights[0] > 0) {                                     // the curve the straight line missed
    stroke('darkorange');
    beginShape();
    for (let x = 0; x <= 10.001; x += 0.25) vertex(g.gx(A.b0 + A.b1 * x), g.gy(A.bend * (x * x - A.quad.b0 - A.quad.b1 * x)));
    endShape();
  }
  // the middle half of the residuals in the left, middle, and right third: equal heights mean equal spread
  stroke('purple');
  strokeWeight(1.5);
  fill(128, 0, 128, 40);
  for (const t of A.thirds) rect(g.gx(t.lo), g.gy(t.q3), g.gx(t.hi) - g.gx(t.lo), g.gy(t.q1) - g.gy(t.q3));
}

function drawHistogram(x0, y0, w, h, narrow) {
  const zR = Math.max(3, Math.ceil(Math.max(...A.z.map(Math.abs)))), bins = new Array(4 * zR).fill(0);
  A.z.forEach(v => bins[Math.min(bins.length - 1, Math.floor((v + zR) * 2))]++);
  const top = Math.ceil(Math.max(...bins, 13) / 4) * 4;
  const g = frame(x0, y0, w, h, 'Histogram of residuals', [-zR, 0, zR], [0, top / 2, top],
    narrow ? 'standardized residual' : 'std. residual (red curve: normal)', narrow);
  stroke('white');
  strokeWeight(1);
  fill('steelblue');
  bins.forEach((c, i) => { if (c > 0) rect(g.gx(-zR + i / 2), g.gy(c), g.gx(0.5) - g.gx(0), g.gy(0) - g.gy(c)); });
  // expected counts for a normal distribution: n * bin width * density
  noFill();
  stroke('crimson');
  strokeWeight(2);
  beginShape();
  for (let v = -zR; v <= zR + 0.001; v += 0.1) vertex(g.gx(v), g.gy(A.n * 0.5 * Math.exp(-v * v / 2) / Math.sqrt(2 * Math.PI)));
  endShape();
}

function drawQQ(x0, y0, w, h, show, narrow) {
  const zR = Math.max(3, Math.ceil(Math.max(...A.z.map(Math.abs))));
  const g = frame(x0, y0, w, h, 'Normal Q-Q plot of residuals', [-zR, 0, zR], [-zR, 0, zR], 'normal quantile', narrow);
  stroke('crimson');
  strokeWeight(2);
  line(g.gx(-zR), g.gy(-zR), g.gx(zR), g.gy(zR));
  const sorted = A.z.slice().sort((p, q) => p - q), qs = sorted.map((v, i) => normalQuantile((i + 0.5) / A.n));
  dots(qs, sorted, g, narrow);
  if (show) rings(qs, sorted, sorted, g);
}

// The four assumptions: a question to answer while the diagnosis is hidden, a light and a verdict when it is shown
function drawPanel(x0, y0, w, h, show, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const ts = narrow ? 12 : 13, lh = ts + 4, tx = x0 + 10, tw = w - 20, rowH = narrow ? 33 : 80;
  let y = y0 + (narrow ? 6 : 34);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  if (!narrow) {
    textStyle(BOLD);
    textSize(15);
    text(show ? 'Diagnosis' : 'What to look for', tx, y0 + 9);
  }
  ROWS.forEach((row, i) => {
    const lv = show ? A.lights[i] : -1, L = LEVELS[lv];
    stroke('gray');
    strokeWeight(1);
    fill(L ? L[2] : 'silver');
    circle(tx + 8, y + 8, 15);
    noStroke();
    fill('black');
    textStyle(BOLD);
    textSize(ts);
    textAlign(LEFT, TOP);
    text(row.name, tx + 22, y);
    if (L) {
      fill(L[1]);
      text(L[0], tx + 30 + textWidth(row.name), y);
    }
    textStyle(NORMAL);
    fill('black');
    text(L ? row.say[lv] : row.ask, tx, y + lh + 1, tw, narrow ? lh : 2 * lh);
    if (L) {
      fill('dimgray');
      textSize(narrow ? 11 : 12);
      textAlign(narrow ? RIGHT : LEFT, TOP);
      text(A.stats[i], narrow ? tx + tw : tx, narrow ? y + 1 : y + 3 * lh + 1);
    }
    y += rowH;
  });
  const far = A.z.filter(v => Math.abs(v) > 3).length;
  fill(show && far ? 'firebrick' : 'dimgray');
  textSize(narrow ? 11 : ts);
  textAlign(LEFT, TOP);
  text(!show ? 'Study the four plots, decide which assumption fails, then check Show diagnosis.'
    : (far ? 'Circled in red: ' + far + (far > 1 ? ' residuals' : ' residual') + ' more than 3 standard deviations from 0. ' : '') +
      'One violation can set off another light, so deal with a red light first.', tx, y + (narrow ? 0 : 6), tw, y0 + h - y - 4);
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
