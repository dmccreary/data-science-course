// Polynomial Degree Explorer
// CANVAS_HEIGHT: 585
// Bloom L3 and L5 (Apply, Evaluate): students set the degree of a polynomial fit for five data
// shapes and noise levels, read how the error splits into bias and variance, and judge which
// degree gives the best trade-off.
//
// Model: 20 training points at evenly spaced x from 0 to 10, y = f(x) + noise, noise ~ Normal(0, σ).
// The least-squares polynomial of each degree is built from polynomials that are orthogonal on the
// training x values (Forsythe's recurrence), which stays accurate at degree 15.
// Because the fit is linear in y, its behaviour over repeated samples is known exactly:
//   expected fit   = the same polynomial fit applied to the noise-free f(x)
//   bias²(d)       = average over x of (expected fit − f(x))²
//   variance(d)    = σ² × average over x of h(x),  h(x) = Σ p_k(x)² / Σ_i p_k(x_i)²,  k = 0..d
//   expected test MSE = σ² + bias² + variance      (x averaged over a fine grid on 0 to 10)
// The shaded band is the fitted curve ± 2 σ √h(x): how far the curve moves when the noise is redrawn.
// R² = 1 − Σ(y − ŷ)² / Σ(y − mean)², on the training points and on 20 test points at random x.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 470;
let controlHeight = 115;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;
let labelWidth = 150;       // width of the label span in each slider row
let valueWidth = 40;        // width of the value span in each slider row

const N = 20, N_TEST = 20, MAX_DEG = 15, GRID = 200, TOLERANCE = 1.05;
const SHAPES = [
  { name: 'Linear', f: x => 4 + 2.4 * x },
  { name: 'Quadratic', f: x => 28 - 0.9 * (x - 5) * (x - 5) },
  { name: 'Cubic (chapter example)', f: x => 2 + 3 * x - 0.5 * x * x + 0.05 * x * x * x },
  { name: 'Sine wave', f: x => 15 + 10 * Math.sin(x) },
  { name: 'Step', f: x => x < 5 ? 8 : 22 }
];
const ZONES = {
  under: { name: 'Underfitting: high bias', curve: 'darkorange', ink: 'chocolate' },
  good: { name: 'Good trade-off', curve: 'forestgreen', ink: 'darkgreen' },
  over: { name: 'Overfitting: high variance', curve: 'crimson', ink: 'firebrick' }
};

let rngState = 1, seed = 3;
const xs = Array.from({ length: N }, (_, i) => 10 * i / (N - 1));
let alpha = [], beta = [], norm = [];       // recurrence terms of the orthogonal polynomials
let gridBasis = [], leverage = [];          // p_k and h(x) for each degree at every grid x
let zTrain = [], xTest = [], zTest = [];    // standard normal noise and test positions
let fit = {};                               // everything derived from the current shape and noise
let degreeRow, noiseRow, shapeSelect, dataButton;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }

// Orthogonal polynomials on the training x values. With t = (x − 5) / 5:
//   p_0 = 1,  p_(k+1) = (t − a_k) p_k − b_k p_(k−1)
function buildBasis() {
  const t = xs.map(x => (x - 5) / 5);
  let pPrev = t.map(() => 0), p = t.map(() => 1);
  for (let k = 0; k <= MAX_DEG; k++) {
    norm[k] = p.reduce((s, v) => s + v * v, 0);
    alpha[k] = p.reduce((s, v, i) => s + t[i] * v * v, 0) / norm[k];
    beta[k] = k === 0 ? 0 : norm[k] / norm[k - 1];
    const next = p.map((v, i) => (t[i] - alpha[k]) * v - beta[k] * pPrev[i]);
    pPrev = p; p = next;
  }
  for (let g = 0; g <= GRID; g++) {
    gridBasis[g] = basisAt(10 * g / GRID);
    let h = 0;
    leverage[g] = gridBasis[g].map((v, k) => (h += v * v / norm[k]));
  }
}
function basisAt(x) {                   // p_0(x) ... p_15(x)
  const t = (x - 5) / 5, out = [];
  let pPrev = 0, p = 1;
  for (let k = 0; k <= MAX_DEG; k++) {
    out.push(p);
    const next = (t - alpha[k]) * p - beta[k] * pPrev;
    pPrev = p; p = next;
  }
  return out;
}
// c_k = <y, p_k> / <p_k, p_k>. The degree-d fit is c_0 p_0 + ... + c_d p_d.
function coefficients(y) { return norm.map((nk, k) => xs.reduce((s, x, i) => s + y[i] * basisAt(x)[k], 0) / nk); }
function predict(coef, basis, d) { let s = 0; for (let k = 0; k <= d; k++) s += coef[k] * basis[k]; return s; }
function rSquared(y, yHat) {
  const m = y.reduce((s, v) => s + v, 0) / y.length;
  return 1 - y.reduce((s, v, i) => s + (v - yHat[i]) ** 2, 0) / y.reduce((s, v) => s + (v - m) ** 2, 0);
}

function newNoise() {
  rngState = seed;
  zTrain = xs.map(() => stdNormal());
  xTest = Array.from({ length: N_TEST }, () => 10 * uniform());
  zTest = xTest.map(() => stdNormal());
}

// Fit every degree and split the expected test error into noise, bias², and variance
function analyze(shape, sigma) {
  const f = shape.f;
  const y = xs.map((x, i) => f(x) + sigma * zTrain[i]), yTest = xTest.map((x, i) => f(x) + sigma * zTest[i]);
  const coef = coefficients(y), coefTrue = coefficients(xs.map(f));
  const bias2 = [], vari = [], total = [], zone = [];
  for (let d = 1; d <= MAX_DEG; d++) {
    let b = 0, h = 0;
    for (let g = 0; g <= GRID; g++) {
      b += (predict(coefTrue, gridBasis[g], d) - f(10 * g / GRID)) ** 2;
      h += leverage[g][d];
    }
    bias2[d] = b / (GRID + 1);
    vari[d] = sigma * sigma * h / (GRID + 1);
    total[d] = sigma * sigma + bias2[d] + vari[d];
  }
  // best: the simplest degree whose expected error equals the minimum (ties happen when σ = 0)
  const lowest = Math.min(...total.slice(1));
  let best = 1;
  while (total[best] > lowest + 1e-9) best++;
  // zones: within 5% of the lowest expected error is a good trade-off. With no noise there is
  // nothing to trade, so any degree with bias² under 0.1 counts as good.
  const slack = sigma > 0 ? 0.01 : 0.1;
  for (let d = 1; d <= MAX_DEG; d++) zone[d] = total[d] <= TOLERANCE * lowest + slack ? 'good' : d < best ? 'under' : 'over';
  const truth = gridBasis.map((_, g) => f(10 * g / GRID));
  fit = { shape, sigma, seed, y, yTest, coef, bias2, vari, total, zone, best, lowest, truth };
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  degreeRow = makeSliderRow('Polynomial degree', 1, MAX_DEG, 1, 1, 0);
  noiseRow = makeSliderRow('Noise level (SD)', 0, 6, 2, 0.5, 1);
  resizeSliders();

  shapeSelect = createSelect();
  shapeSelect.parent(mainElement);
  shapeSelect.position(10, drawHeight + 80);
  SHAPES.forEach(s => shapeSelect.option(s.name));
  shapeSelect.selected(SHAPES[2].name);
  shapeSelect.style('font-size', '15px');

  dataButton = createButton('New Data');
  dataButton.parent(mainElement);
  dataButton.position(225, drawHeight + 80);
  dataButton.mousePressed(() => { seed++; newNoise(); });

  buildBasis();
  newNoise();

  describe('A scatter plot of twenty training points and twenty test points with a fitted polynomial curve, the true ' +
    'curve, and a shaded band showing how far the fit would move with new noise. Sliders set the polynomial degree ' +
    'and the noise level, a menu chooses the shape of the data, and a button draws new data. A panel reports training ' +
    'and test R squared, splits the expected test error into noise, bias, and variance, and gives a verdict.', LABEL);
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
  degreeRow.slider.size(w);
  noiseRow.slider.size(w);
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
  const d = degreeRow.slider.value(), sigma = noiseRow.slider.value();
  const shape = SHAPES.find(s => s.name === shapeSelect.value());
  degreeRow.valueSpan.html(d);
  noiseRow.valueSpan.html(nf(sigma, 1, 1));
  if (fit.shape !== shape || fit.sigma !== sigma || fit.seed !== seed) analyze(shape, sigma);

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Polynomial Degree Explorer', canvasWidth / 2, 8);

  const w = canvasWidth - 2 * margin;
  if (narrow) {
    drawData(margin, 34, w, 204, d, narrow);
    drawVerdict(margin, 244, w, drawHeight - 252, d, narrow);
  } else {
    const dataW = w * 0.58;
    drawData(margin, 42, dataW, drawHeight - 50, d, narrow);
    drawVerdict(margin + dataW + 10, 42, w - dataW - 10, drawHeight - 50, d, narrow);
  }
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

// The data, the true curve, the fitted curve of the chosen degree, and its ±2 SD band
function drawData(x0, y0, w, h, d, narrow) {
  const ts = narrow ? 11 : 13, zone = ZONES[fit.zone[d]];
  drawPanelBox(x0, y0, w, h, 'Degree ' + d + ' fit to ' + N + ' training points', ts);
  const px = x0 + 36, py = y0 + (narrow ? 44 : 52), pw = w - 48, ph = y0 + h - py - 36;
  const pad = 2.5 * fit.sigma + 2;
  const yLo = Math.floor((Math.min(...fit.truth) - pad) / 5) * 5, yHi = Math.ceil((Math.max(...fit.truth) + pad) / 5) * 5;
  const gx = v => px + v / 10 * pw, gy = v => py + ph - (constrain(v, yLo - 60, yHi + 60) - yLo) / (yHi - yLo) * ph;

  // legend
  textSize(ts);
  textAlign(LEFT, CENTER);
  let lx = x0 + 10;
  const ly = y0 + (narrow ? 32 : 37);
  for (const [label, kind] of [['training', 'dot'], ['test', 'box'], ['true curve', 'dash'], ['fit ± 2 SD', 'band']]) {
    stroke(kind === 'dash' ? 'dimgray' : zone.curve);
    strokeWeight(2);
    if (kind === 'dash') { drawingContext.setLineDash([4, 3]); line(lx, ly, lx + 16, ly); drawingContext.setLineDash([]); }
    if (kind === 'band') { noStroke(); fill(160, 160, 160, 90); rect(lx, ly - 6, 16, 12); stroke(zone.curve); line(lx, ly, lx + 16, ly); }
    noStroke();
    if (kind === 'dot') { fill('royalblue'); circle(lx + 8, ly, 8); }
    if (kind === 'box') { fill('darkorange'); square(lx + 4, ly - 4, 8); }
    fill('black');
    text(label, lx + 20, ly);
    lx += textWidth(label) + (narrow ? 30 : 36);
  }

  const step = yHi - yLo > 40 ? 10 : 5;
  textSize(narrow ? 11 : 12);
  for (let v = Math.ceil(yLo / step) * step; v <= yHi; v += step) {
    stroke('gainsboro');
    strokeWeight(1);
    line(px, gy(v), px + pw, gy(v));
    noStroke();
    fill('dimgray');
    textAlign(RIGHT, CENTER);
    text(v, px - 4, gy(v));
  }
  textAlign(CENTER, TOP);
  for (let v = 0; v <= 10; v += 2) text(v, gx(v), py + ph + 3);
  fill('black');
  text('x', px + pw / 2, py + ph + 18);
  stroke('gray');
  strokeWeight(1.5);
  line(px, py, px, py + ph);
  line(px, py + ph, px + pw, py + ph);

  // a high-degree curve and its band can leave the plot near the ends
  push();
  drawingContext.beginPath();
  drawingContext.rect(px, py, pw, ph);
  drawingContext.clip();
  const curve = gridBasis.map(b => predict(fit.coef, b, d));
  const halfBand = leverage.map(h => 2 * fit.sigma * Math.sqrt(h[d]));
  noStroke();
  fill(160, 160, 160, 90);
  beginShape();
  for (let g = 0; g <= GRID; g++) vertex(gx(10 * g / GRID), gy(curve[g] + halfBand[g]));
  for (let g = GRID; g >= 0; g--) vertex(gx(10 * g / GRID), gy(curve[g] - halfBand[g]));
  endShape(CLOSE);
  noFill();
  stroke('dimgray');
  strokeWeight(1.5);
  drawingContext.setLineDash([4, 3]);
  beginShape();
  fit.truth.forEach((v, g) => vertex(gx(10 * g / GRID), gy(v)));
  endShape();
  drawingContext.setLineDash([]);
  stroke(zone.curve);
  strokeWeight(3);
  beginShape();
  curve.forEach((v, g) => vertex(gx(10 * g / GRID), gy(v)));
  endShape();
  pop();

  stroke('white');
  strokeWeight(1);
  fill('darkorange');
  xTest.forEach((x, i) => square(gx(x) - 3.5, gy(fit.yTest[i]) - 3.5, 7));
  fill('royalblue');
  xs.forEach((x, i) => circle(gx(x), gy(fit.y[i]), narrow ? 8 : 9));
}

const signed = (v, digits) => (Math.abs(v) < 0.5 * Math.pow(10, -digits) ? 0 : v).toFixed(digits).replace('-', '−');

// R² of the chosen degree, the three parts of its expected test error, and a verdict
function drawVerdict(x0, y0, w, h, d, narrow) {
  const ts = narrow ? 12 : 14, lh = ts + 6, zone = ZONES[fit.zone[d]], ix = x0 + 10, iw = w - 20;
  drawPanelBox(x0, y0, w, h, 'Degree ' + d + ' polynomial: ' + (d + 1) + ' coefficients', ts);
  const yHat = xs.map(x => predict(fit.coef, basisAt(x), d)), yHatTest = xTest.map(x => predict(fit.coef, basisAt(x), d));
  const r2Train = rSquared(fit.y, yHat), r2Test = rSquared(fit.yTest, yHatTest);
  const showR2 = v => !isFinite(v) ? 'undefined' : v < -9.99 ? 'below −10' : signed(v, 3);
  let y = y0 + 9 + lh;
  fill('mediumblue');
  text('Training R²: ' + showR2(r2Train), ix, y);
  fill('chocolate');
  text('Test R²: ' + showR2(r2Test), narrow ? ix + iw / 2 : ix, narrow ? y : y + lh);
  y += (narrow ? 1 : 2) * lh + 4;

  // expected test MSE as a stacked bar: noise, then bias², then variance
  const noise = fit.sigma * fit.sigma, b2 = fit.bias2[d], v = fit.vari[d];
  fill('black');
  textStyle(BOLD);
  text('Expected test MSE: ' + signed(fit.total[d], 2), ix, y);
  textStyle(NORMAL);
  y += lh + 2;
  const cap = Math.max(2.2 * fit.lowest, 1), barH = narrow ? 14 : 18, sx = v2 => ix + Math.min(v2, cap) / cap * iw;
  fill('whitesmoke');
  stroke('silver');
  strokeWeight(1);
  rect(ix, y, iw, barH);
  noStroke();
  const parts = [['Noise σ²', noise, 'darkgray'], ['Bias²', b2, 'royalblue'], ['Variance', v, 'darkorange']];
  let run = 0;
  for (const [, value, col] of parts) {
    fill(col);
    rect(sx(run), y, sx(run + value) - sx(run), barH);
    run += value;
  }
  // marker at the lowest expected test MSE that any degree reaches
  stroke('black');
  strokeWeight(2);
  line(sx(fit.lowest), y - 4, sx(fit.lowest), y + barH + 4);
  noStroke();
  fill('black');
  textSize(ts - 1);
  textAlign(sx(fit.lowest) > ix + iw / 2 ? RIGHT : LEFT, TOP);
  text('lowest at any degree: ' + signed(fit.lowest, 2), sx(fit.lowest) + (sx(fit.lowest) > ix + iw / 2 ? 0 : -1), y + barH + 5);
  if (fit.total[d] > cap) {
    fill('white');
    textAlign(RIGHT, CENTER);
    text('off the scale »', ix + iw - 5, y + barH / 2 + 1);
  }
  y += barH + lh + 4;
  textAlign(LEFT, TOP);
  textSize(ts);
  let lx = ix;
  for (const [label, value, col] of parts) {
    fill(col);
    square(lx, y + 2, ts - 3);
    fill('black');
    const str = label + ' ' + signed(value, 2);
    text(str, lx + ts + 1, y);
    if (narrow) lx += textWidth(str) + ts + 14; else y += lh;
  }
  y += narrow ? lh + 4 : 8;

  // past the best degree the bias can rise again: the curve swings between the points near the ends
  const swings = b2 > fit.bias2[fit.best] + 0.5;
  const msg = {
    under: 'Too stiff. A degree ' + d + ' curve cannot bend enough to follow this pattern, so it misses in the same places ' +
      'whichever sample it is fit to. Squared bias adds ' + signed(b2, 2) + ' to the expected test error.',
    good: (d === fit.best ? 'This degree has the lowest expected test error for this shape and noise level.'
      : fit.sigma === 0 ? 'With no noise there is nothing to overfit, and this curve already follows the pattern closely.'
        : 'Expected test error is within 5% of the lowest possible. ' + (d > fit.best ? 'A lower degree does at least as well with fewer coefficients.'
          : 'It is also simpler than the degree with the very lowest error.')) +
      (b2 > 1 ? ' The bias that remains is the price of keeping the variance low.' : ''),
    over: v > 0 ? 'Too flexible. With ' + (d + 1) + ' coefficients for ' + N + ' points, the curve bends to follow the noise. ' +
      'Variance adds ' + signed(v, 2) + ' to the expected test error' + (swings ? ', and the curve swings between the points near the ends' : '') +
      '. Press New Data and watch how far the curve moves.'
      : 'Even with no noise, a degree ' + d + ' curve swings between the points near the ends, so its error rises again.'
  }[fit.zone[d]];
  textWrap(WORD);
  fill(zone.ink);
  const zoneName = fit.zone[d] === 'over' && v === 0 ? 'Too flexible' : zone.name;      // no noise, no variance
  if (narrow) {
    text(zoneName + '. ' + msg, ix, y, iw, y0 + h - y - 2);
    return;
  }
  textStyle(BOLD);
  textSize(ts + 2);
  text(zoneName, ix, y);
  textStyle(NORMAL);
  textSize(ts);
  fill('black');
  text(msg, ix, y + lh + 4, iw, y0 + h - y - lh - 6);
  fill('dimgray');
  textSize(ts - 1);
  text('Gray band: the fit ± 2 standard deviations, which is how far the curve moves when the noise is drawn again. ' +
    'Dashed line: the true curve behind the data.', ix, y0 + h - 76, iw, 72);
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
