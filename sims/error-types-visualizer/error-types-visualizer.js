// Error Types Visualizer
// CANVAS_HEIGHT: 650
// Bloom L4-L5 (Analyze, Evaluate): students compare training error with test error as model
// complexity changes, judge whether the model is underfitting, a good fit, or overfitting, and
// then reveal the diagnosis and the error curves to check that judgment.
//
// Model: y = 5 + 3 (4t^3 - 3t) + noise with t = (x - 5) / 5, x uniform on [0, 10], and normal
// noise of standard deviation sigma. A polynomial of the chosen degree is fit to the training
// points by least squares (QR factorization of a Chebyshev basis, which stays accurate at
// degree 12). MSE = mean((y - prediction)^2), measured on the training points and on 40 test
// points the fit never saw. Diagnosis: "good fit" when the test MSE is within 25% of the lowest
// test MSE over degrees 1 to 12; otherwise underfitting below that degree, overfitting above it.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 500;
let controlHeight = 150;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;
let labelWidth = 150;       // width of the label span in each slider row
let valueWidth = 45;        // width of the value span in each slider row

const MAX_DEG = 12, MAX_TRAIN = 60, N_TEST = 40, Y_MIN = -2, Y_MAX = 12;
let rngState = 1, dataSeed = 7, fitKey = '';
let raw = null;                         // seeded draws: { trainX, trainE, testX, testE }
let train = [], test = [];              // current points { x, y }
let coefs = [], trainMSE = [], testMSE = [], best = 1;   // indexed by degree 1..MAX_DEG
let degreeRow, sizeRow, noiseRow, newDataButton, revealCheckbox;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }

function trueF(x) { const t = (x - 5) / 5; return 5 + 3 * (4 * t * t * t - 3 * t); }

// Chebyshev polynomials T0..T(p-1) at x, after mapping [0, 10] to [-1, 1]
function chebRow(x, p) {
  const t = (x - 5) / 5, row = [1, t];
  for (let j = 2; j < p; j++) row.push(2 * t * row[j - 1] - row[j - 2]);
  return row;
}
function predict(c, x) { const row = chebRow(x, c.length); return c.reduce((s, cj, j) => s + cj * row[j], 0); }
function mse(pts, c) { return pts.reduce((s, p) => s + (p.y - predict(c, p.x)) ** 2, 0) / pts.length; }
// A wildly overfit polynomial can miss test points by a huge margin, so large errors are abbreviated
function fmt(v) { return v < 1000 ? nf(v, 1, 2) : 'over 1000'; }

// x positions and standard normal noise for one data set. The noise slider only rescales the noise.
function drawRaw() {
  rngState = 500 + dataSeed;
  const zeros = n => new Array(n).fill(0);
  raw = { trainX: zeros(MAX_TRAIN).map(() => 10 * uniform()), trainE: zeros(MAX_TRAIN).map(stdNormal),
    testX: zeros(N_TEST).map(() => 10 * uniform()), testE: zeros(N_TEST).map(stdNormal) };
}

// Least-squares fits of every degree at once. Gram-Schmidt (two passes) gives A = QR for the
// degree-12 basis; the fit of degree d solves the leading (d+1) x (d+1) block of R c = Q^T y.
function fitAllDegrees(pts) {
  const p = MAX_DEG + 1, rows = pts.map(pt => chebRow(pt.x, p)), ys = pts.map(pt => pt.y);
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const Q = [], R = Array.from({ length: p }, () => new Array(p).fill(0)), qty = [];
  for (let j = 0; j < p; j++) {
    let v = rows.map(r => r[j]);
    for (let pass = 0; pass < 2; pass++) {
      for (let i = 0; i < j; i++) {
        const r = dot(Q[i], v);
        R[i][j] += r;
        v = v.map((vk, k) => vk - r * Q[i][k]);
      }
    }
    R[j][j] = Math.sqrt(dot(v, v));
    Q.push(v.map(vk => vk / R[j][j]));
    qty.push(dot(Q[j], ys));
  }
  const out = [];
  for (let d = 1; d <= MAX_DEG; d++) {
    const c = new Array(d + 1).fill(0);
    for (let i = d; i >= 0; i--) {
      let s = qty[i];
      for (let k = i + 1; k <= d; k++) s -= R[i][k] * c[k];
      c[i] = s / R[i][i];
    }
    out[d] = c;
  }
  return out;
}

function refit(n, sigma) {
  train = raw.trainX.slice(0, n).map((x, i) => ({ x, y: trueF(x) + sigma * raw.trainE[i] }));
  test = raw.testX.map((x, i) => ({ x, y: trueF(x) + sigma * raw.testE[i] }));
  coefs = fitAllDegrees(train);
  best = 1;
  for (let d = 1; d <= MAX_DEG; d++) {
    trainMSE[d] = mse(train, coefs[d]);
    testMSE[d] = mse(test, coefs[d]);
    if (testMSE[d] < testMSE[best]) best = d;
  }
}

// Verdict for degree d, worded so that every statement follows from the numbers on screen
function diagnose(d, n) {
  const f = fmt, tr = f(trainMSE[d]), te = f(testMSE[d]);
  if (testMSE[d] <= 1.25 * testMSE[best]) {
    return { name: 'Good fit', color: 'green', text: 'the test MSE (' + te + ') is within 25% of the lowest value any degree reaches here (' +
      f(testMSE[best]) + ' at degree ' + best + '), so this model predicts new data about as well as any degree can.' };
  }
  if (d < best) {
    return { name: 'Underfitting', color: 'darkorange', text: 'a degree ' + d + ' model is too simple. Its training MSE (' + tr + ') and test MSE (' + te +
      ') are both higher than at degree ' + best + ' (' + f(trainMSE[best]) + ' and ' + f(testMSE[best]) + '). Raise the degree.' };
  }
  return { name: 'Overfitting', color: 'firebrick', text: 'a degree ' + d + ' model follows the noise in its ' + n + ' training points. Its training MSE (' + tr +
    ') is lower than at degree ' + best + ' (' + f(trainMSE[best]) + '), but its test MSE (' + te + ') is higher (' + f(testMSE[best]) +
    '). Lower the degree or add training data.' };
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  degreeRow = makeSliderRow('Polynomial degree', 1, MAX_DEG, 1, 1, 0);
  sizeRow = makeSliderRow('Training set size', 15, MAX_TRAIN, 20, 5, 1);
  noiseRow = makeSliderRow('Noise level (σ)', 0.2, 1.5, 0.8, 0.1, 2);
  newDataButton = createButton('New Data');
  newDataButton.parent(mainElement);
  newDataButton.position(10, drawHeight + 115);
  newDataButton.mousePressed(() => { dataSeed++; drawRaw(); });
  revealCheckbox = createCheckbox(' Show diagnosis and error curves', false);
  revealCheckbox.parent(mainElement);
  revealCheckbox.position(100, drawHeight + 116);
  revealCheckbox.style('font-size', '16px');
  resizeSliders();
  drawRaw();

  describe('Two scatter plots share one fitted polynomial curve: training points on the left and unseen test points on the right, ' +
    'each with residual lines and its mean squared error. Bars compare the two errors, and a chart of error against polynomial ' +
    'degree can be revealed together with a diagnosis of underfitting, good fit, or overfitting.', LABEL);
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
  for (const r of [degreeRow, sizeRow, noiseRow]) r.slider.size(w);
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const d = degreeRow.slider.value(), n = sizeRow.slider.value(), sigma = noiseRow.slider.value();
  degreeRow.valueSpan.html(d);
  sizeRow.valueSpan.html(n);
  noiseRow.valueSpan.html(nf(sigma, 1, 1));
  const key = dataSeed + '|' + n + '|' + sigma;
  if (key !== fitKey) { refit(n, sigma); fitKey = key; }
  const reveal = revealCheckbox.checked(), narrow = canvasWidth < 600, ts = narrow ? 11 : 14;
  const w = canvasWidth - 2 * margin, gap = 10, half = (w - gap) / 2;

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textStyle(NORMAL);
  textSize(narrow ? 18 : 24);
  text('Training Error vs Test Error', canvasWidth / 2, 7);
  fill('dimgray');
  textSize(narrow ? 11 : 13);
  text(narrow ? 'Green: degree ' + d + ' fit to the training points. Dashed: true pattern.'
    : 'One degree ' + d + ' polynomial (green), fit to the training points only. Dashed gray: the true pattern.', canvasWidth / 2, narrow ? 30 : 37);

  const top = narrow ? 48 : 58, plotH = 222;
  drawScatter(margin, top, half, plotH, narrow ? 'Training data' : 'Training data (n = ' + n + ')', train, 'royalblue', trainMSE[d], coefs[d], narrow);
  drawScatter(margin + half + gap, top, half, plotH, narrow ? 'Test data' : 'Test data (' + N_TEST + ' new points)', test, 'darkorange', testMSE[d], coefs[d], narrow);

  const rowY = top + plotH + 8, rowH = 140, barsW = narrow ? w * 0.42 : w * 0.4;
  const cap = 1.25 * max(trainMSE[1], testMSE[1]);          // shared scale for the bars and the chart
  drawBars(margin, rowY, barsW, rowH, d, sigma, cap, narrow, ts);
  drawCurves(margin + barsW + gap, rowY, w - barsW - gap, rowH, d, cap, reveal, narrow);

  // the student's judgment, then the verdict
  const textY = rowY + rowH + 8, verdict = diagnose(d, n);
  noStroke();
  textAlign(LEFT, TOP);
  textSize(narrow ? 11 : 15);
  textWrap(WORD);
  fill(reveal ? verdict.color : 'black');
  text(reveal ? verdict.name + ': ' + verdict.text
    : 'Your call: is the degree ' + d + ' model underfitting, a good fit, or overfitting? Compare the two errors, decide, ' +
      'then check the box below to see the diagnosis and the error curves.', margin, textY, w, drawHeight - textY - 2);
}

// One scatter plot with the fitted curve, the true pattern, and a residual line for every point
function drawScatter(x0, y0, w, h, title, pts, col, err, c, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  noStroke();
  textStyle(BOLD);
  textSize(narrow ? 12 : 15);
  textAlign(LEFT, TOP);
  fill('black');
  text(title, x0 + 8, y0 + 7);
  textAlign(RIGHT, TOP);
  fill(col);
  text('MSE = ' + fmt(err), x0 + w - 8, y0 + 7);
  textStyle(NORMAL);

  const px = x0 + (narrow ? 24 : 32), py = y0 + 30, pw = w - (narrow ? 32 : 44), ph = h - 52;
  const gx = v => px + v / 10 * pw, gy = v => py + ph - (v - Y_MIN) / (Y_MAX - Y_MIN) * ph;
  stroke('gainsboro');
  for (const v of [0, 5, 10]) { line(px, gy(v), px + pw, gy(v)); line(gx(v), py, gx(v), py + ph); }
  noFill();
  stroke('gray');
  rect(px, py, pw, ph);
  noStroke();
  fill('dimgray');
  textSize(narrow ? 11 : 12);
  for (const v of [0, 5, 10]) {
    textAlign(CENTER, TOP);
    text(v, gx(v), py + ph + 4);
    textAlign(RIGHT, CENTER);
    text(v, px - 4, gy(v));
  }

  push();                                       // clip: a high-degree curve can leave the frame
  drawingContext.beginPath();
  drawingContext.rect(px, py, pw, ph);
  drawingContext.clip();
  const curve = f => { beginShape(); for (let i = 0; i <= 120; i++) vertex(gx(i / 12), gy(f(i / 12))); endShape(); };
  noFill();
  stroke('gray');
  strokeWeight(1.5);
  drawingContext.setLineDash([5, 4]);
  curve(trueF);
  drawingContext.setLineDash([]);
  const rgb = color(col);
  stroke(red(rgb), green(rgb), blue(rgb), 150);
  strokeWeight(1);
  for (const p of pts) line(gx(p.x), gy(p.y), gx(p.x), gy(predict(c, p.x)));
  stroke('green');
  strokeWeight(2.5);
  curve(x => predict(c, x));
  stroke('white');
  strokeWeight(1);
  fill(col);
  for (const p of pts) circle(gx(p.x), gy(p.y), narrow ? 6 : 8);
  pop();
}

// Training MSE and test MSE at the current degree, with the noise variance as a reference line
function drawBars(x0, y0, w, h, d, sigma, cap, narrow, ts) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(narrow ? 12 : 15);
  text('Error at degree ' + d, x0 + 8, y0 + 7);
  textStyle(NORMAL);
  textSize(ts);
  const labelW = narrow ? 34 : 104, valueW = narrow ? 34 : 44;
  const bx = x0 + 8 + labelW, bw = w - 16 - labelW - valueW, sx = v => bx + min(v, cap) / cap * bw;
  const rows = [[narrow ? 'Train' : 'Training MSE', trainMSE[d], 'royalblue'], [narrow ? 'Test' : 'Test MSE', testMSE[d], 'darkorange']];
  rows.forEach(([name, v, col], i) => {
    const y = y0 + 34 + i * 30;
    noStroke();
    fill('black');
    textAlign(LEFT, CENTER);
    text(name, x0 + 8, y + 10);
    fill(col);
    rect(bx, y, sx(v) - bx, 20);
    textAlign(LEFT, CENTER);
    text(v < 1000 ? nf(v, 1, 2) : '>1000', bx + bw + 5, y + 10);
  });
  // noise variance: the error a perfect model would still make on new data
  stroke('dimgray');
  strokeWeight(1.5);
  drawingContext.setLineDash([4, 3]);
  line(sx(sigma * sigma), y0 + 30, sx(sigma * sigma), y0 + 88);
  line(x0 + 8, y0 + 94 + ts / 2, x0 + 28, y0 + 94 + ts / 2);
  drawingContext.setLineDash([]);
  noStroke();
  fill('dimgray');
  textAlign(LEFT, TOP);
  text((narrow ? 'noise σ² = ' : 'noise variance σ² = ') + nf(sigma * sigma, 1, 2), x0 + 34, y0 + 94);
  fill('black');
  text((narrow ? 'Gap = ' : 'Gap (test − train) = ') + fmt(testMSE[d] - trainMSE[d]), x0 + 8, y0 + 94 + ts + 6);
}

// MSE against degree. Only the current degree is plotted until the curves are revealed.
function drawCurves(x0, y0, w, h, d, cap, reveal, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(narrow ? 12 : 15);
  text('MSE by degree', x0 + 8, y0 + 7);
  textStyle(NORMAL);
  textSize(narrow ? 11 : 12);
  textAlign(RIGHT, TOP);
  fill(reveal ? 'green' : 'dimgray');
  text(reveal ? (narrow ? 'best: degree ' : 'lowest test MSE: degree ') + best : 'curves hidden', x0 + w - 8, y0 + 9);

  const px = x0 + 36, py = y0 + 30, pw = w - 50, ph = h - 62;
  const gx = v => px + (v - 0.5) / MAX_DEG * pw, gy = v => py + ph - v / cap * ph;
  stroke('gray');
  strokeWeight(1);
  line(px, py, px, py + ph);
  line(px, py + ph, px + pw, py + ph);
  noStroke();
  fill('dimgray');
  textAlign(CENTER, TOP);
  for (let k = 1; k <= MAX_DEG; k++) if (!narrow || k % 2 === 1) text(k, gx(k), py + ph + 3);
  fill('black');
  text('polynomial degree', px + pw / 2, py + ph + 16);
  fill('dimgray');
  textAlign(RIGHT, CENTER);
  text('0', px - 4, py + ph);
  text(nf(cap, 1, 1), px - 4, py + 4);

  push();
  drawingContext.beginPath();
  drawingContext.rect(px, py - 8, pw + 6, ph + 9);
  drawingContext.clip();
  stroke('silver');
  strokeWeight(1);
  line(gx(d), py, gx(d), py + ph);
  if (reveal) {
    stroke('green');
    strokeWeight(1.5);
    drawingContext.setLineDash([4, 3]);
    line(gx(best), py, gx(best), py + ph);
    drawingContext.setLineDash([]);
  }
  for (const [series, col] of [[trainMSE, 'royalblue'], [testMSE, 'darkorange']]) {
    if (reveal) {
      noFill();
      stroke(col);
      strokeWeight(2);
      beginShape();
      for (let k = 1; k <= MAX_DEG; k++) vertex(gx(k), gy(series[k]));
      endShape();
    }
    stroke('white');
    strokeWeight(1.5);
    fill(col);
    circle(gx(d), gy(min(series[d], cap)), 11);
  }
  pop();
  if (testMSE[d] > cap) {
    noStroke();
    fill('darkorange');
    textAlign(LEFT, TOP);
    text('↑', gx(d) + 7, py - 8);
  }
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
