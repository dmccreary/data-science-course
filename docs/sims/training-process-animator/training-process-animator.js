// Training Process Animator
// CANVAS_HEIGHT: 535
// Bloom L2-L3 (Understand, Apply): students step through the training loop of a linear model,
// watch the line and its parameters change with each update, and choose a learning rate and a
// starting line to see fast, slow, and failed training.
//
// Data: 40 seeded students, score = 2 + 0.75 hours + noise. The first 30 are training rows and
// the last 10 are validation rows. Hours are standardized with the training mean and standard
// deviation, z = (hours - mean) / sd, as a scaler fit on the training set would do.
// Model: prediction = w z + b. Cost: J = mean((prediction - score)^2) over the training rows.
// One iteration is one pass over all training rows (full-batch gradient descent):
//   dJ/dw = (2/n) sum((prediction - score) z),  dJ/db = (2/n) sum(prediction - score)
//   w = w - eta dJ/dw,  b = b - eta dJ/db
// Validation MSE uses the same w and b on rows that never enter the gradient. Training stops
// when the gradient length falls below 0.02, when the MSE passes 1,000,000, or at 200 iterations.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 455;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let sliderLeftMargin = 200;
let defaultTextSize = 16;

const N = 40, N_TRAIN = 30, DATA_SEED = 38, MAX_ITER = 200, GRAD_TOL = 0.02, Z_MAX = 2.2, Y_MAX = 12;
const STARTS = [{ name: 'Start at w = 0, b = 0', w: 0, b: 0 }, { name: 'Start at w = 5, b = 2', w: 5, b: 2 },
  { name: 'Start at w = −2, b = 9', w: -2, b: 9 }];
let rngState = 1;
let train = [], val = [];       // rows { hours, z, y }
let best = null;                // closed-form least-squares fit on the training rows: { w, b, mse }
let hist = [];                  // one record per iteration: { w, b, gw, gb, train, val }
let done = null;                // null while training can continue, else 'converged', 'diverged', or 'capped'
let running = false, lastStepFrame = 0;
let stepButton, startButton, resetButton, startSelect, rateSlider;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }

function makeData() {
  rngState = DATA_SEED;
  const rows = [];
  for (let i = 0; i < N; i++) {
    const hours = Math.round((1 + 8 * uniform()) * 10) / 10;
    rows.push({ hours, y: Math.min(10, Math.max(0, Math.round((2 + 0.75 * hours + 0.8 * stdNormal()) * 10) / 10)) });
  }
  train = rows.slice(0, N_TRAIN);
  val = rows.slice(N_TRAIN);
  const mean = train.reduce((s, r) => s + r.hours, 0) / N_TRAIN;
  const sd = Math.sqrt(train.reduce((s, r) => s + (r.hours - mean) ** 2, 0) / N_TRAIN);
  for (const r of rows) r.z = (r.hours - mean) / sd;
  // least squares on the training rows, for comparison with where gradient descent ends
  const mz = train.reduce((s, r) => s + r.z, 0) / N_TRAIN, my = train.reduce((s, r) => s + r.y, 0) / N_TRAIN;
  let szy = 0, szz = 0;
  for (const r of train) { szy += (r.z - mz) * (r.y - my); szz += (r.z - mz) ** 2; }
  best = { w: szy / szz, b: my - szy / szz * mz };
  best.mse = mseOn(train, best.w, best.b);
}

function mseOn(rows, w, b) { return rows.reduce((s, r) => s + (w * r.z + b - r.y) ** 2, 0) / rows.length; }

// Parameters, gradient of the training MSE, and both errors at one iteration
function record(w, b) {
  let gw = 0, gb = 0;
  for (const r of train) { const err = w * r.z + b - r.y; gw += err * r.z; gb += err; }
  return { w, b, gw: 2 * gw / N_TRAIN, gb: 2 * gb / N_TRAIN, train: mseOn(train, w, b), val: mseOn(val, w, b) };
}

function resetTraining() {
  const s = STARTS.find(o => o.name === startSelect.value());
  hist = [record(s.w, s.b)];
  done = null;
  running = false;
  updateButtons();
}

// One iteration of gradient descent on the training rows
function stepOnce() {
  if (done) return;
  const cur = hist[hist.length - 1], eta = rateSlider.value();
  const next = record(cur.w - eta * cur.gw, cur.b - eta * cur.gb);
  hist.push(next);
  if (!isFinite(next.train) || next.train > 1e6) done = 'diverged';
  else if (Math.hypot(next.gw, next.gb) < GRAD_TOL) done = 'converged';
  else if (hist.length > MAX_ITER) done = 'capped';
  if (done) running = false;
  updateButtons();
}

function updateButtons() {
  startButton.html(running ? 'Pause' : 'Start');
  for (const b of [stepButton, startButton]) { if (done) b.attribute('disabled', ''); else b.removeAttribute('disabled'); }
}

// What the last update did, worded from the recorded numbers
function statusMessage() {
  const k = hist.length - 1, cur = hist[k], f = v => nf(v, 1, 3);
  if (k === 0) return ['Starting parameters: this line was not fit to the data. Press Step for one update of w and b, or Start to train.', 'black'];
  const prev = hist[k - 1], change = prev.train - cur.train;
  const flipped = prev.gw * cur.gw + prev.gb * cur.gb < 0;      // the update jumped past the best line
  if (done === 'diverged') return ['Diverged after ' + k + ' iterations: every update overshoots farther and the error explodes. Lower the learning rate and press Reset.', 'firebrick'];
  if (done === 'converged') {
    return ['Converged after ' + k + (k === 1 ? ' iteration' : ' iterations') + ': the gradient is nearly zero, so w and b have stopped changing. ' +
      'The least-squares formula gives w = ' + f(best.w) + ', b = ' + f(best.b) + ', MSE = ' + f(best.mse) + '.', 'green'];
  }
  if (done === 'capped') {
    return ['Stopped at ' + MAX_ITER + ' iterations without converging. ' + (flipped ? 'The line keeps jumping from one side of the best fit to the other: lower the learning rate.'
      : 'The updates are too small: raise the learning rate.'), 'darkorange'];
  }
  if (change < -1e-9) return ['Overshoot: this update made the training MSE go UP, from ' + f(prev.train) + ' to ' + f(cur.train) + '. The learning rate is too large.', 'firebrick'];
  if (change < 1e-9) return ['No progress: the update jumped to a line that is exactly as wrong on the other side. Lower the learning rate.', 'darkorange'];
  if (flipped) {
    return ['Jumped past the best line: the gradient reversed, but the training MSE still fell, from ' + f(prev.train) + ' to ' + f(cur.train) +
      '. A smaller learning rate would be steadier.', 'darkorange'];
  }
  return [(change > 0.01 * hist[0].train ? 'Big adjustment' : 'Fine-tuning') + ': this update lowered the training MSE by ' + f(change) +
    ', from ' + f(prev.train) + ' to ' + f(cur.train) + '.', 'darkgreen'];
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  const makeButton = (label, x, action) => {
    const b = createButton(label);
    b.parent(mainElement);
    b.position(x, drawHeight + 10);
    b.mousePressed(action);
    return b;
  };
  stepButton = makeButton('Step', 10, () => { running = false; stepOnce(); });
  startButton = makeButton('Start', 60, () => { running = !running; lastStepFrame = frameCount; updateButtons(); });
  resetButton = makeButton('Reset', 116, () => resetTraining());
  startSelect = createSelect();
  startSelect.parent(mainElement);
  startSelect.position(178, drawHeight + 10);
  STARTS.forEach(s => startSelect.option(s.name));
  startSelect.style('font-size', '15px');
  startSelect.changed(resetTraining);
  rateSlider = createSlider(0.01, 1.1, 0.1, 0.01);
  rateSlider.parent(mainElement);
  rateSlider.position(sliderLeftMargin, drawHeight + 45);
  rateSlider.size(canvasWidth - sliderLeftMargin - margin);

  makeData();
  resetTraining();

  describe('A scatter plot of quiz score against standardized study hours with a regression line that moves at every iteration of ' +
    'gradient descent. Panels show the iteration number, the weight and bias, their gradients, and the training and validation ' +
    'mean squared error, and a chart plots both errors against the iteration. A slider sets the learning rate.', LABEL);
}

function draw() {
  updateCanvasSize();
  if (running && frameCount - lastStepFrame >= (hist.length > 30 ? 2 : 6)) { stepOnce(); lastStepFrame = frameCount; }

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600, w = canvasWidth - 2 * margin, ts = narrow ? 11 : 15;
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textStyle(NORMAL);
  textSize(narrow ? 18 : 24);
  text('Training a Model, One Iteration at a Time', canvasWidth / 2, 8);

  // layout: scatter plot, then the numbers and the loss chart (beside the plot, or below it when narrow)
  const top = narrow ? 36 : 44, statusH = narrow ? 58 : 50, bodyH = drawHeight - top - statusH - 16;
  let scatter, info, chart;
  if (narrow) {
    scatter = { x: margin, y: top, w, h: bodyH - 136 };
    info = { x: margin, y: top + bodyH - 130, w: w * 0.47, h: 130 };
    chart = { x: margin + w * 0.47 + 8, y: info.y, w: w * 0.53 - 8, h: 130 };
  } else {
    scatter = { x: margin, y: top, w: w * 0.56, h: bodyH };
    info = { x: margin + w * 0.56 + 10, y: top, w: w * 0.44 - 10, h: 190 };
    chart = { x: info.x, y: top + 198, w: info.w, h: bodyH - 198 };
  }
  drawScatter(scatter, narrow);
  drawInfo(info, ts, narrow);
  drawChart(chart, narrow);

  const [msg, col] = statusMessage(), sy = drawHeight - statusH - 8;
  fill('white');
  stroke(col === 'black' ? 'silver' : col);
  strokeWeight(1.5);
  rect(margin, sy, w, statusH, 10);
  noStroke();
  fill(col);
  textAlign(LEFT, TOP);
  textSize(ts);
  textWrap(WORD);
  text(msg, margin + 10, sy + 6, w - 20, statusH - 8);

  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Learning rate η: ' + nf(rateSlider.value(), 1, 2), 10, drawHeight + 56);
}

// Data, earlier lines as fading ghosts, the current line, and its residuals on the training rows
function drawScatter(r, narrow) {
  const cur = hist[hist.length - 1], k = hist.length - 1;
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(r.x, r.y, r.w, r.h, 10);
  // header: legend and the current equation
  const small = narrow ? 11 : 13;
  noStroke();
  textSize(small);
  textAlign(LEFT, CENTER);
  fill('royalblue');
  circle(r.x + 14, r.y + 14, 9);
  fill('black');
  text(narrow ? 'train (30)' : 'training rows (30)', r.x + 23, r.y + 14);
  const lx = r.x + (narrow ? 88 : 150);
  noFill();
  stroke('darkorange');
  strokeWeight(2);
  circle(lx, r.y + 14, 8);
  noStroke();
  fill('black');
  text(narrow ? 'validation (10)' : 'validation rows (10)', lx + 9, r.y + 14);
  textAlign(RIGHT, CENTER);
  textStyle(BOLD);
  text('ŷ = ' + nf(cur.w, 1, 2) + ' z ' + (cur.b < 0 ? '− ' : '+ ') + nf(Math.abs(cur.b), 1, 2), r.x + r.w - 10, r.y + 14);
  textStyle(NORMAL);

  const px = r.x + (narrow ? 36 : 46), py = r.y + 28, pw = r.w - (narrow ? 46 : 58), ph = r.h - (narrow ? 60 : 66);
  const gx = v => px + (v + Z_MAX) / (2 * Z_MAX) * pw, gy = v => py + ph - v / Y_MAX * ph;
  stroke('gainsboro');
  strokeWeight(1);
  for (let v = -2; v <= 2; v++) line(gx(v), py, gx(v), py + ph);
  for (let v = 0; v <= Y_MAX; v += 2) line(px, gy(v), px + pw, gy(v));
  noFill();
  stroke('gray');
  rect(px, py, pw, ph);
  noStroke();
  fill('dimgray');
  textSize(narrow ? 11 : 12);
  textAlign(CENTER, TOP);
  for (let v = -2; v <= 2; v++) text(v, gx(v), py + ph + 3);
  textAlign(RIGHT, CENTER);
  for (let v = 0; v <= Y_MAX; v += 2) text(v, px - 4, gy(v));
  fill('black');
  textAlign(CENTER, TOP);
  text('hours studied, standardized (z)', px + pw / 2, py + ph + 17);
  push();
  translate(r.x + (narrow ? 10 : 13), py + ph / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, CENTER);
  text('quiz score', 0, 0);
  pop();

  push();
  drawingContext.beginPath();
  drawingContext.rect(px, py, pw, ph);
  drawingContext.clip();
  const drawLine = h => line(gx(-Z_MAX), gy(h.b - h.w * Z_MAX), gx(Z_MAX), gy(h.b + h.w * Z_MAX));
  for (let i = Math.max(0, k - 8); i < k; i++) {             // the last eight lines, fading with age
    stroke(120, 120, 120, 25 + 20 * (i - (k - 8)));
    strokeWeight(1.5);
    if (isFinite(hist[i].train)) drawLine(hist[i]);
  }
  // line color: red at the starting error, green at the least-squares error
  const span = Math.max(hist[0].train - best.mse, 1e-9);
  const lineColor = lerpColor(color('green'), color('red'), Math.sqrt(constrain((cur.train - best.mse) / span, 0, 1)));
  if (isFinite(cur.train)) {
    stroke(red(lineColor), green(lineColor), blue(lineColor), 110);
    strokeWeight(1);
    for (const p of train) line(gx(p.z), gy(p.y), gx(p.z), gy(cur.w * p.z + cur.b));
    stroke(lineColor);
    strokeWeight(3);
    drawLine(cur);
  }
  stroke('white');
  strokeWeight(1);
  fill('royalblue');
  for (const p of train) circle(gx(p.z), gy(p.y), narrow ? 7 : 9);
  noFill();
  stroke('darkorange');
  strokeWeight(2);
  for (const p of val) circle(gx(p.z), gy(p.y), narrow ? 6 : 8);
  pop();
}

// Parameters, gradients, and errors at the current iteration
function drawInfo(r, ts, narrow) {
  const cur = hist[hist.length - 1], f = v => nf(Math.abs(v) < 0.0005 ? 0 : v, 1, 3);
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(r.x, r.y, r.w, r.h, 10);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text('Iteration ' + (hist.length - 1) + (narrow ? '' : '  (stops at ' + MAX_ITER + ')'), r.x + 10, r.y + 7);
  textStyle(NORMAL);
  textSize(ts);
  const lines = [['Weight w = ' + f(cur.w), 'black'], ['Bias b = ' + f(cur.b), 'black'],
    ['Gradient ∂J/∂w = ' + f(cur.gw), 'dimgray'], ['Gradient ∂J/∂b = ' + f(cur.gb), 'dimgray'],
    ['Training MSE = ' + f(cur.train), 'royalblue'], ['Validation MSE = ' + f(cur.val), 'darkorange']];
  const lh = (r.h - 30) / lines.length;
  lines.forEach(([str, col], i) => { fill(col); text(str, r.x + 10, r.y + 28 + i * lh); });
}

// Training and validation MSE at every iteration so far
function drawChart(r, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(r.x, r.y, r.w, r.h, 10);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(narrow ? 12 : 14);
  text('MSE by iteration', r.x + 10, r.y + 6);
  textStyle(NORMAL);
  // the vertical scale covers the largest error so far, so a diverging run stays on the chart
  const k = hist.length - 1, n = Math.max(20, k), yMax = 1.1 * Math.max(...hist.flatMap(h => [h.train, h.val]).filter(isFinite));
  const px = r.x + 40, py = r.y + 28, pw = r.w - 54, ph = r.h - 48;
  const gx = i => px + i / n * pw, gy = v => py + ph - v / yMax * ph;
  stroke('gray');
  line(px, py, px, py + ph);
  line(px, py + ph, px + pw, py + ph);
  noStroke();
  fill('dimgray');
  textSize(narrow ? 11 : 12);
  textAlign(RIGHT, CENTER);
  text(yMax < 1e5 ? nf(yMax, 1, 0) : yMax.toExponential(0), px - 4, py + 3);
  text('0', px - 4, py + ph);
  textAlign(CENTER, TOP);
  text('0', px, py + ph + 3);
  text(n, px + pw, py + ph + 3);
  text('iteration', px + pw / 2, py + ph + 3);
  push();
  drawingContext.beginPath();
  drawingContext.rect(px - 5, py - 5, pw + 12, ph + 10);
  drawingContext.clip();
  for (const [key, col] of [['val', 'darkorange'], ['train', 'royalblue']]) {
    noFill();
    stroke(col);
    strokeWeight(2);
    beginShape();
    hist.forEach((h, i) => { if (isFinite(h[key])) vertex(gx(i), gy(h[key])); });
    endShape();
    stroke('white');
    strokeWeight(1);
    fill(col);
    if (isFinite(hist[k][key])) circle(gx(k), gy(hist[k][key]), 8);
  }
  pop();
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  rateSlider.size(canvasWidth - sliderLeftMargin - margin);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
