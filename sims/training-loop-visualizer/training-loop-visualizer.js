// Training Loop Visualizer
// CANVAS_HEIGHT: 620
// Bloom L3-L4 (Apply, Analyze): students run the five lines of a PyTorch training loop one at a
// time on a one-neuron model, read the weight, bias, gradients, and loss after every line, and
// analyze how the learning rate and a missing zero_grad() change the loss curve.
//
// Model: pred = w x + b (nn.Linear(1, 1)) fitted to 8 seeded points, all 8 in one batch, with
// nn.MSELoss and optim.SGD. Each line is computed here the way PyTorch computes it:
//   optimizer.zero_grad()   .grad of w and b is reset to None (PyTorch 2.x; older versions stored 0)
//   pred = model(x)         pred_i = w x_i + b
//   loss = criterion(...)   loss = mean((pred - y)^2)
//   loss.backward()         ADDS mean(2 (pred - y) x) to w.grad and mean(2 (pred - y)) to b.grad
//   optimizer.step()        w = w - lr * w.grad,  b = b - lr * b.grad
// Because backward() adds to .grad, a loop without zero_grad() updates with the sum of every
// gradient so far. Checked against PyTorch 2.10.0 running the same loop on the same data.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 540;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let sliderLeftMargin = 165;
let defaultTextSize = 16;

const STEPS = [
  { name: 'Zero gradients', code: 'optimizer.zero_grad()', color: 'royalblue',
    why: 'backward() adds each new gradient to whatever is already stored in .grad. zero_grad() clears the stored gradients, so the next backward() starts from nothing.' },
  { name: 'Forward pass', code: 'pred = model(x)', color: 'seagreen',
    why: 'The model predicts a value for every point in the batch: pred\u00a0=\u00a0w\u00a0×\u00a0x\u00a0+\u00a0b. The predictions are the green rings on the line in the plot.' },
  { name: 'Compute loss', code: 'loss = criterion(pred, y)', color: 'darkorange',
    why: 'MSELoss averages the squared errors (pred − y)² over the 8 points. The orange bars in the plot are the errors. One number now says how wrong the model is.' },
  { name: 'Backward pass', code: 'loss.backward()', color: 'purple',
    why: 'Backpropagation finds how the loss changes when w or b changes and adds those slopes to .grad. The weight and bias do not change yet.' },
  { name: 'Update weights', code: 'optimizer.step()', color: 'crimson',
    why: 'SGD moves each parameter a small step against its gradient: new value = old value − lr × gradient.' }
];
const SEED = 31;
const X = [], Y = [];               // the 8 training points, generated once from SEED
let w0, b0;                         // seeded starting weight and bias
let startLoss, bestLoss;            // loss of the starting model and of the least-squares line
let m;                              // model and loop state, see resetModel()
let stepButton, runButton, resetButton, skipBox, lrSlider;

// Seeded data y = 2x + 1 + noise, and starting values drawn from (-1, 1) as nn.Linear(1, 1) does
function makeData() {
  let seed = SEED;
  const rnd = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;
  for (let i = 0; i < 8; i++) {
    const x = Math.round((-2 + 4 * rnd()) * 10) / 10;
    X.push(x);
    Y.push(Math.round((2 * x + 1 + (rnd() + rnd() + rnd() - 1.5) * 0.8) * 10) / 10);
  }
  w0 = Math.round((2 * rnd() - 1) * 100) / 100;
  b0 = Math.round((2 * rnd() - 1) * 100) / 100;
  // least-squares line: the lowest loss any w and b can reach
  const mx = mean(X), my = mean(Y);
  const wBest = mean(X.map((x, i) => (x - mx) * (Y[i] - my))) / mean(X.map(x => (x - mx) ** 2));
  startLoss = lossAt(w0, b0);
  bestLoss = lossAt(wBest, my - wBest * mx);
}

function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }
function lossAt(w, b) { return mean(X.map((x, i) => (w * x + b - Y[i]) ** 2)); }

function resetModel() {
  m = { w: w0, b: b0, gw: null, gb: null, loss: null, pred: null, iter: 0, stage: 0, updates: 0,
    history: [], lines: [], changed: [], skipped: false, diverged: false };
}

function fmt(v) {
  if (v === null) return 'None';
  return (Math.abs(v) >= 1e4 ? v.toExponential(2) : v.toFixed(4)).replace(/-/g, '−');
}
// negative numbers go in brackets inside a calculation: a model value, and a data value
function par(v) { return v < 0 ? '(' + fmt(v) + ')' : fmt(v); }
function dat(v) { return v < 0 ? '(−' + (-v) + ')' : String(v); }

// Run the next line of the loop. m.stage counts the lines already run in the current iteration.
function doStep() {
  if (m.diverged) return;
  if (m.iter === 0 || m.stage === 5) { m.iter++; m.stage = 0; }
  const lr = lrSlider.value();
  m.changed = [];
  m.skipped = m.stage === 0 && skipBox.checked();
  if (m.stage === 0) {
    if (m.skipped) {
      m.lines = ['w.grad stays ' + fmt(m.gw) + ',  b.grad stays ' + fmt(m.gb)];
    } else {
      m.lines = ['w.grad: ' + fmt(m.gw) + ' → None', 'b.grad: ' + fmt(m.gb) + ' → None', 'None means that no gradient is stored.'];
      m.gw = m.gb = null;
      m.changed = ['gw', 'gb'];
    }
  } else if (m.stage === 1) {
    m.pred = X.map(x => m.w * x + m.b);
    m.lines = ['pred[0] = ' + fmt(m.w) + ' × ' + dat(X[0]) + ' + ' + par(m.b) + ' = ' + fmt(m.pred[0]),
      'The true value is y[0] = ' + String(Y[0]).replace('-', '−') + '.'];
  } else if (m.stage === 2) {
    m.loss = mean(m.pred.map((p, i) => (p - Y[i]) ** 2));
    m.history.push(m.loss);
    m.lines = ['(pred[0] − y[0])² = (' + fmt(m.pred[0]) + ' − ' + dat(Y[0]) + ')² = ' + fmt((m.pred[0] - Y[0]) ** 2),
      'loss = mean of the 8 squared errors = ' + fmt(m.loss)];
    m.changed = ['loss'];
    if (!isFinite(m.loss) || m.loss > 1e6) m.diverged = true;
  } else if (m.stage === 3) {
    const gw = mean(m.pred.map((p, i) => 2 * (p - Y[i]) * X[i])), gb = mean(m.pred.map((p, i) => 2 * (p - Y[i])));
    if (m.gw === null) {
      m.lines = ['w.grad = mean(2 (pred − y) x) = ' + fmt(gw), 'b.grad = mean(2 (pred − y)) = ' + fmt(gb)];
    } else {                                                 // zero_grad() was skipped: the new gradient is added to the old one
      m.lines = ['w.grad = ' + fmt(m.gw) + ' + ' + par(gw) + ' = ' + fmt(m.gw + gw), 'b.grad = ' + fmt(m.gb) + ' + ' + par(gb) + ' = ' + fmt(m.gb + gb),
        'Old gradient + new gradient: they pile up.'];
    }
    m.gw = (m.gw === null ? 0 : m.gw) + gw;
    m.gb = (m.gb === null ? 0 : m.gb) + gb;
    m.changed = ['gw', 'gb'];
  } else {
    const w1 = m.w - lr * m.gw, b1 = m.b - lr * m.gb;
    m.lines = ['w = ' + fmt(m.w) + ' − ' + lr.toFixed(2) + ' × ' + par(m.gw) + ' = ' + fmt(w1),
      'b = ' + fmt(m.b) + ' − ' + lr.toFixed(2) + ' × ' + par(m.gb) + ' = ' + fmt(b1)];
    m.w = w1;
    m.b = b1;                                                // .grad keeps its value until the next zero_grad()
    m.updates++;
    m.changed = ['w', 'b'];
  }
  m.stage++;
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);
  makeData();
  resetModel();

  stepButton = createButton('Step');
  stepButton.parent(mainElement);
  stepButton.position(10, drawHeight + 8);
  stepButton.mousePressed(doStep);
  runButton = createButton('Run 10 Iterations');
  runButton.parent(mainElement);
  runButton.position(60, drawHeight + 8);
  runButton.mousePressed(() => {
    const target = m.updates + 10;
    while (m.updates < target && !m.diverged) doStep();
  });
  resetButton = createButton('Reset');
  resetButton.parent(mainElement);
  resetButton.position(186, drawHeight + 8);
  resetButton.mousePressed(resetModel);
  skipBox = createCheckbox(' Skip zero_grad()', false);
  skipBox.parent(mainElement);
  skipBox.position(246, drawHeight + 8);
  skipBox.style('font-size', '16px');

  lrSlider = createSlider(0.01, 1, 0.1, 0.01);
  lrSlider.parent(mainElement);
  lrSlider.position(sliderLeftMargin, drawHeight + 45);
  lrSlider.size(canvasWidth - sliderLeftMargin - margin);

  describe('The five lines of a PyTorch training loop: zero gradients, forward pass, compute loss, backward pass, and ' +
    'update weights. A Step button runs one line at a time on a one-neuron model. Panels show the line that just ran, ' +
    'an explanation with the actual numbers, the weight, bias, gradients, and loss, a plot of the data with the model ' +
    'line, and the loss curve. A slider sets the learning rate and a checkbox skips zero_grad.', LABEL);
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
  for (const b of [stepButton, runButton]) { if (m.diverged) b.attribute('disabled', ''); else b.removeAttribute('disabled'); }
  textWrap(WORD);

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Training Loop Visualizer', canvasWidth / 2, narrow ? 7 : 8);

  if (narrow) {
    const half = Math.round(w / 2);
    drawCode(margin, 34, w, 136, true);
    drawExplain(margin, 176, w, 128, true);
    drawState(margin, 310, half - 3, 118, true);
    drawScatter(margin + half + 3, 310, w - half - 3, 118, true);
    drawLoss(margin, 434, w, drawHeight - 442, true);
  } else {
    const lw = Math.round(w * 0.52), rx = margin + lw + 10, rw = w - lw - 10;
    drawCode(margin, 44, lw, 222, false);
    drawExplain(margin, 274, lw, drawHeight - 282, false);
    drawState(rx, 44, rw, 146, false);
    drawScatter(rx, 198, rw, 170, false);
    drawLoss(rx, 376, rw, drawHeight - 384, false);
  }

  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Learning rate: ' + lrSlider.value().toFixed(2), 10, drawHeight + 56);
}

function panel(x0, y0, w, h, bg) {
  fill(bg);
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
}

function panelTitle(s, x0, y0, ts, c) {
  noStroke();
  fill(c || 'black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text(s, x0 + 10, y0 + 6);
  textStyle(NORMAL);
  textSize(ts);
}

// The loop as PyTorch code. The line that just ran is highlighted.
function drawCode(x0, y0, w, h, narrow) {
  panel(x0, y0, w, h, 'white');
  const ts = narrow ? 12 : 14, headH = narrow ? 22 : 30, skip = skipBox.checked();
  panelTitle('PyTorch training loop', x0, y0, ts);
  const rows = narrow ? [] : [[-1, 'model = nn.Linear(1, 1)'], [-1, 'criterion = nn.MSELoss()']];
  rows.push([-1, 'optimizer = optim.SGD(model.parameters(), lr=' + lrSlider.value().toFixed(2) + ')'], [-1, 'for epoch in range(epochs):']);
  STEPS.forEach((s, i) => rows.push([i, i === 0 && skip ? '# ' + s.code : s.code]));
  const pitch = (h - headH - 6) / rows.length;
  rows.forEach(([i, code], k) => {
    const y = y0 + headH + k * pitch + pitch / 2, current = i >= 0 && i === m.stage - 1, off = i === 0 && skip;
    if (current) {
      const tint = color(off ? 'red' : STEPS[i].color);
      tint.setAlpha(40);
      noStroke();
      fill(tint);
      rect(x0 + 4, y - pitch / 2 + 1, w - 8, pitch - 2, 5);
    }
    noStroke();
    textSize(ts);
    if (i >= 0) {                                            // numbered loop line with its name on the right
      fill(off ? 'silver' : STEPS[i].color);
      circle(x0 + 34, y, narrow ? 15 : 19);
      fill('white');
      textAlign(CENTER, CENTER);
      textStyle(BOLD);
      textSize(narrow ? 11 : 12);
      text(i + 1, x0 + 34, y + 1);
      textSize(ts);
      fill(off ? 'firebrick' : STEPS[i].color);
      textAlign(RIGHT, CENTER);
      text(off ? 'skipped' : STEPS[i].name, x0 + w - 10, y + 1);
      textStyle(current ? BOLD : NORMAL);
    }
    fill(off ? 'firebrick' : 'black');
    textAlign(LEFT, CENTER);
    text(code, x0 + (i >= 0 ? 50 : 10), y + 1);
    textStyle(NORMAL);
  });
}

// What the line that just ran did, with the actual numbers
function drawExplain(x0, y0, w, h, narrow) {
  const ts = narrow ? 12 : 14, i = m.stage - 1;
  let title = 'Before training', c = 'black', body = 'The model starts with a random weight and bias, so its line does not fit ' +
    'the points yet. Press Step to run the loop one line at a time.', lines = ['w = ' + fmt(w0) + ',  b = ' + fmt(b0)];
  if (m.diverged) {
    title = 'The loss is exploding';
    c = 'firebrick';
    body = 'The learning rate is too large, so every update overshoots the best line by more than the last one did. ' +
      'Lower the learning rate and press Reset.';
    lines = ['loss = ' + fmt(m.loss) + ' in iteration ' + m.iter];
  } else if (i >= 0) {
    title = 'Step ' + (i + 1) + ' of 5: ' + STEPS[i].name + '   (iteration ' + m.iter + ')';
    c = m.skipped ? 'firebrick' : STEPS[i].color;
    body = m.skipped ? 'Skipped. The gradients from the last iteration stay in .grad, and the next backward() will add to them ' +
      'instead of replacing them.' : STEPS[i].why;
    lines = m.lines;
  }
  panel(x0, y0, w, h, m.diverged || m.skipped ? 'mistyrose' : 'lightyellow');
  panelTitle(title, x0, y0, ts, c);
  fill('black');
  const bodyH = narrow ? 48 : 78;
  text(body, x0 + 10, y0 + ts + 14, w - 20, bodyH);
  textStyle(BOLD);
  text(lines.join('\n'), x0 + 10, y0 + ts + 18 + bodyH, w - 20, h - ts - 22 - bodyH);
  textStyle(NORMAL);
  if (!narrow) drawBatch(x0 + 10, y0 + h - 86, w - 20);
}

// The batch as a table: inputs and targets, then predictions and squared errors once they are current
function drawBatch(x0, y0, w) {
  const live = m.stage >= 2 && m.stage <= 4, colW = (w - 84) / X.length;
  const cell = v => (Math.abs(v) >= 100 ? v.toExponential(0) : v.toFixed(2)).replace(/-/g, '−');
  const rows = [['x', X.map(v => String(v).replace('-', '−')), 'black'], ['y', Y.map(v => String(v).replace('-', '−')), 'black'],
    ['pred', live ? m.pred.map(cell) : null, 'seagreen'], ['(pred − y)²', live && m.stage >= 3 ? m.pred.map((p, i) => cell((p - Y[i]) ** 2)) : null, 'chocolate']];
  stroke('silver');
  strokeWeight(1);
  line(x0, y0 - 3, x0 + w, y0 - 3);
  noStroke();
  textSize(12);
  rows.forEach(([label, values, c], r) => {
    const y = y0 + 9 + r * 20;
    fill(c);
    textAlign(LEFT, CENTER);
    text(label, x0, y);
    textAlign(RIGHT, CENTER);
    X.forEach((x, i) => text(values ? values[i] : '·', x0 + 84 + (i + 1) * colW - 3, y));
  });
}

// Weight, bias, stored gradients, and loss. Values changed by the last line are highlighted.
function drawState(x0, y0, w, h, narrow) {
  panel(x0, y0, w, h, 'white');
  const ts = narrow ? 12 : 14, headH = narrow ? 22 : 28, i = m.stage - 1;
  panelTitle(narrow ? 'Iteration ' + m.iter : 'Model state', x0, y0, ts);
  fill('dimgray');
  textAlign(RIGHT, TOP);
  text((narrow ? '' : 'iteration ' + m.iter + ',  ') + m.updates + ' updates', x0 + w - 10, y0 + 7);
  const rows = [['weight w', m.w, 'w'], ['bias b', m.b, 'b'], ['w.grad', m.gw, 'gw'], ['b.grad', m.gb, 'gb'], ['loss', m.loss, 'loss']];
  const pitch = (h - headH - 6) / rows.length;
  rows.forEach(([label, value, key], k) => {
    const y = y0 + headH + k * pitch + pitch / 2, hot = m.changed.includes(key);
    if (hot) {
      const tint = color(STEPS[i].color);
      tint.setAlpha(45);
      noStroke();
      fill(tint);
      rect(x0 + 4, y - pitch / 2 + 1, w - 8, pitch - 2, 5);
    }
    noStroke();
    fill('black');
    textStyle(hot ? BOLD : NORMAL);
    textAlign(LEFT, CENTER);
    text(label, x0 + 10, y + 1);
    textAlign(RIGHT, CENTER);
    text(key === 'loss' && value === null ? 'not computed yet' : fmt(value), x0 + w - 10, y + 1);
    textStyle(NORMAL);
  });
}

// The 8 points and the model line. Predictions appear after the forward pass, errors after the loss.
function drawScatter(x0, y0, w, h, narrow) {
  panel(x0, y0, w, h, 'white');
  const top = narrow ? 8 : 28, px = x0 + 30, py = y0 + top, pw = w - 42, ph = h - top - 22;
  const gx = v => px + (v + 2.5) / 5 * pw, gy = v => py + (6 - v) / 10 * ph;
  if (!narrow) panelTitle('Data and the model line  pred = w x + b', x0, y0, 14);
  textSize(narrow ? 11 : 12);
  for (const v of [-2, 0, 2]) {
    stroke('gainsboro');
    strokeWeight(1);
    line(gx(v), py, gx(v), py + ph);
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    text(v, gx(v), py + ph + 4);
  }
  for (const v of [-4, 0, 4]) {
    stroke('gainsboro');
    line(px, gy(v), px + pw, gy(v));
    noStroke();
    fill('dimgray');
    textAlign(RIGHT, CENTER);
    text(v, px - 4, gy(v));
  }
  drawingContext.save();
  drawingContext.beginPath();
  drawingContext.rect(px, py, pw, ph);
  drawingContext.clip();
  stroke('seagreen');
  strokeWeight(2.5);
  line(gx(-2.5), gy(-2.5 * m.w + m.b), gx(2.5), gy(2.5 * m.w + m.b));
  if (m.stage >= 2 && m.stage <= 4) {                       // pred belongs to the current w and b
    X.forEach((x, i) => {
      if (m.stage >= 3) { stroke('darkorange'); strokeWeight(3); line(gx(x), gy(Y[i]), gx(x), gy(m.pred[i])); }
      stroke('seagreen');
      strokeWeight(1.5);
      fill('white');
      circle(gx(x), gy(m.pred[i]), 7);
    });
  }
  drawingContext.restore();
  stroke('white');
  strokeWeight(1);
  fill('black');
  X.forEach((x, i) => circle(gx(x), gy(Y[i]), narrow ? 7 : 9));
}

// Loss computed in each iteration, with the lowest loss any line can reach as a dashed reference
function drawLoss(x0, y0, w, h, narrow) {
  panel(x0, y0, w, h, 'white');
  const hist = m.history, top = narrow ? 24 : 30, px = x0 + 46, py = y0 + top, pw = w - 62, ph = h - top - 22;
  const N = Math.max(20, hist.length), biggest = Math.max(startLoss, ...hist), unit = Math.pow(10, Math.floor(Math.log10(biggest)));
  const yMax = unit * [1, 1.5, 2, 3, 5, 7.5, 10].find(k => k * unit >= biggest);
  const gx = k => px + (k - 1) / (N - 1) * pw, gy = v => py + ph - Math.min(v, yMax) / yMax * ph;
  panelTitle('Loss by iteration', x0, y0, narrow ? 12 : 14);
  textSize(narrow ? 11 : 12);
  fill('seagreen');
  textAlign(RIGHT, TOP);
  text('dashed: lowest possible loss, ' + fmt(bestLoss), x0 + w - 10, y0 + 8);
  for (const v of [0, yMax / 2, yMax]) {
    stroke('gainsboro');
    strokeWeight(1);
    line(px, gy(v), px + pw, gy(v));
    noStroke();
    fill('dimgray');
    textAlign(RIGHT, CENTER);
    text(v >= 1e4 ? v.toExponential(1) : v, px - 4, gy(v));
  }
  textAlign(CENTER, TOP);
  for (const k of [1, Math.round(N / 2), N]) text(k, gx(k), py + ph + 4);
  text('iteration', gx(N * 0.75), py + ph + 4);
  stroke('seagreen');
  strokeWeight(1.5);
  drawingContext.setLineDash([5, 4]);
  line(px, gy(bestLoss), px + pw, gy(bestLoss));
  drawingContext.setLineDash([]);
  noFill();
  stroke('darkorange');
  strokeWeight(2.5);
  beginShape();
  hist.forEach((v, k) => vertex(gx(k + 1), gy(v)));
  endShape();
  if (hist.length) {
    stroke('white');
    strokeWeight(1.5);
    fill('darkorange');
    circle(gx(hist.length), gy(hist[hist.length - 1]), 10);
  }
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  lrSlider.size(canvasWidth - sliderLeftMargin - margin);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
