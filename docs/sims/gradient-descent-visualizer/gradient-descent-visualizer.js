// Gradient Descent Visualizer
// CANVAS_HEIGHT: 545
// Bloom L2-L3 (Understand, Apply): students take gradient descent steps on a contour map of a
// cost function, read the gradient and the next position at each step, and change the learning
// rate to produce slow progress, zigzag, and divergence.
//
// Update rule: w_new = w - eta * grad J(w), using the exact gradient of each cost function.
//   Simple bowl       J = w1^2 + w2^2                    grad J = (2 w1, 2 w2)      diverges for eta > 1
//   Elongated valley  J = 0.2 w1^2 + 2 w2^2              grad J = (0.4 w1, 4 w2)    diverges for eta > 0.5
//   Two valleys       J = (w1^2 - 1)^2 + 0.3 w1 + w2^2   grad J = (4 w1^3 - 4 w1 + 0.3, 2 w2)
// The two valleys have minima near w1 = -1.04 (global) and w1 = 0.96 (local). Every minimum is
// located numerically at startup. A run stops when |grad J| < 0.001 (converged), when J passes
// 10,000 (diverged), or after 300 steps.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 465;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let sliderLeftMargin = 200;
let defaultTextSize = 16;

const W1_MAX = 3, W2_MAX = 2, MAX_STEPS = 300, TOLERANCE = 0.001, BANDS = 14;
const LANDSCAPES = [
  { name: 'Simple bowl', formula: 'J(w) = w₁² + w₂²', gradient: '∇J = (2w₁, 2w₂)', power: 0.5, start: [-2, 1.5], seeds: [[1, 1]],
    f: (a, b) => a * a + b * b, g: (a, b) => [2 * a, 2 * b] },
  { name: 'Elongated valley', formula: 'J(w) = 0.2w₁² + 2w₂²', gradient: '∇J = (0.4w₁, 4w₂)', power: 0.5, start: [-2.5, 1.5], seeds: [[1, 1]],
    f: (a, b) => 0.2 * a * a + 2 * b * b, g: (a, b) => [0.4 * a, 4 * b] },
  { name: 'Two valleys', formula: 'J(w) = (w₁² − 1)² + 0.3w₁ + w₂²', gradient: '∇J = (4w₁³ − 4w₁ + 0.3, 2w₂)', power: 0.3, start: [1.6, 1.4],
    seeds: [[-1, 0.5], [1, 0.5]],
    f: (a, b) => (a * a - 1) ** 2 + 0.3 * a + b * b, g: (a, b) => [4 * a * a * a - 4 * a + 0.3, 2 * b] }
];

let li = 0;                 // index of the current landscape
let start = [-2, 1.5];
let path = [];              // iterates: { w: [w1, w2], J, g: [g1, g2] }
let state = 'ready';        // ready, moving, converged, diverged, capped
let running = false, lastStepFrame = 0;
let heat = null, heatKey = '', plot = null;
let stepButton, startButton, resetButton, landscapeSelect, rateSlider;

// Locate each minimum by running gradient descent with a small step from a nearby seed
function findMinima() {
  for (const L of LANDSCAPES) {
    L.minima = L.seeds.map(seed => {
      let w = seed.slice();
      for (let i = 0; i < 5000; i++) { const g = L.g(w[0], w[1]); w = [w[0] - 0.05 * g[0], w[1] - 0.05 * g[1]]; }
      return { w, J: L.f(w[0], w[1]) };
    });
    L.jMin = Math.min(...L.minima.map(m => m.J));
  }
}

// Cost and gradient at a position (not named point: that is a p5.js function)
function evaluate(w) {
  const L = LANDSCAPES[li];
  return { w, J: L.f(w[0], w[1]), g: L.g(w[0], w[1]) };
}

function resetRun() {
  path = [evaluate(start)];
  state = 'ready';
  running = false;
  updateButtons();
}

// One gradient descent update: w_new = w - eta * grad J(w)
function stepOnce() {
  if (state !== 'ready' && state !== 'moving') return;
  const cur = path[path.length - 1], eta = rateSlider.value();
  const next = evaluate([cur.w[0] - eta * cur.g[0], cur.w[1] - eta * cur.g[1]]);
  path.push(next);
  state = 'moving';
  if (!isFinite(next.J) || next.J > 10000) state = 'diverged';
  else if (Math.hypot(next.g[0], next.g[1]) < TOLERANCE) state = 'converged';
  else if (path.length > MAX_STEPS) state = 'capped';
  if (state !== 'moving') running = false;
  updateButtons();
}

function updateButtons() {
  startButton.html(running ? 'Pause' : 'Start');
  for (const b of [stepButton, startButton]) {
    if (state === 'ready' || state === 'moving') b.removeAttribute('disabled'); else b.attribute('disabled', '');
  }
}

// What the last step did, worded from the numbers in the path
function statusMessage() {
  const L = LANDSCAPES[li], k = path.length - 1, cur = path[k], f = v => nf(v, 1, 3);
  if (k === 0) return ['Press Step to take one gradient step, or Start to run. Click the map to choose a new starting point.', 'black'];
  const prev = path[k - 1], steps = k + (k === 1 ? ' step' : ' steps'), eps = 1e-9 * Math.max(1, Math.abs(prev.J));
  if (state === 'diverged') {
    return ['Diverged after ' + steps + '. Each step overshoots farther than the one before, so the cost explodes. Lower the learning rate and press Reset.', 'firebrick'];
  }
  if (state === 'converged') {
    let msg = 'Converged in ' + steps + ': the gradient is nearly zero, so the steps have stopped.';
    const d = m => Math.hypot(m.w[0] - cur.w[0], m.w[1] - cur.w[1]);
    const here = L.minima.reduce((a, b) => d(b) < d(a) ? b : a), other = L.minima.find(m => m !== here);
    if (d(here) > 0.05) msg += ' But this flat spot is a saddle point between the valleys, not a minimum.';
    else if (other) {
      msg += here.J <= other.J ? ' This is the global minimum.'
        : ' But this is only a local minimum: J = ' + f(here.J) + ' here, and J = ' + f(other.J) + ' in the other valley.';
    }
    return [msg, 'green'];
  }
  if (state === 'capped') {
    const falling = path.slice(-20).every((pt, i, a) => i === 0 || pt.J <= a[i - 1].J);
    return ['Stopped after ' + MAX_STEPS + ' steps without converging. ' +
      (falling ? 'The steps are too small: raise the learning rate.' : 'The steps keep overshooting: lower the learning rate.'), 'darkorange'];
  }
  if (cur.J > prev.J + eps) {
    return ['Overshoot: the cost went UP from ' + f(prev.J) + ' to ' + f(cur.J) + ' on this step. The learning rate is too large here.', 'firebrick'];
  }
  if (prev.g[0] * cur.g[0] + prev.g[1] * cur.g[1] < 0) {
    return ['Zigzag: the step jumped across the valley floor, so the gradient now points back the other way. ' + (cur.J < prev.J - eps
      ? 'The cost still fell, from ' + f(prev.J) + ' to ' + f(cur.J) + '.' : 'It landed just as high on the far side: the cost is stuck at ' + f(cur.J) + '.'), 'darkorange'];
  }
  return ['Descending: the cost fell from ' + f(prev.J) + ' to ' + f(cur.J) + ' on this step.', 'darkgreen'];
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
  resetButton = makeButton('Reset', 116, resetRun);
  landscapeSelect = createSelect();
  landscapeSelect.parent(mainElement);
  landscapeSelect.position(178, drawHeight + 10);
  LANDSCAPES.forEach(L => landscapeSelect.option(L.name));
  landscapeSelect.style('font-size', '15px');
  landscapeSelect.changed(() => {
    li = LANDSCAPES.findIndex(L => L.name === landscapeSelect.value());
    start = LANDSCAPES[li].start.slice();
    resetRun();
  });
  rateSlider = createSlider(0.01, 1.1, 0.1, 0.01);
  rateSlider.parent(mainElement);
  rateSlider.position(sliderLeftMargin, drawHeight + 45);
  rateSlider.size(canvasWidth - sliderLeftMargin - margin);

  findMinima();
  resetRun();

  describe('A contour map of a cost function of two weights. A red path shows the gradient descent steps taken so far and an arrow ' +
    'shows the next step. Panels list the current position, cost, gradient, and next step, and a chart plots the cost at every step. ' +
    'A slider sets the learning rate and a menu chooses a bowl, an elongated valley, or a surface with two valleys.', LABEL);
}

function draw() {
  updateCanvasSize();
  // while running, take a step every few frames (faster once the run is long)
  if (running && frameCount - lastStepFrame >= (path.length > 40 ? 2 : 8)) { stepOnce(); lastStepFrame = frameCount; }

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600, w = canvasWidth - 2 * margin, ts = narrow ? 11 : 14;
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textStyle(NORMAL);
  textSize(narrow ? 18 : 24);
  text('Gradient Descent on a Cost Surface', canvasWidth / 2, 8);

  // layout: map with the cost chart below it and text panels beside it (panels move below when narrow)
  let info, chart, status;
  if (narrow) {
    plot = { x: (canvasWidth - 300) / 2 + 12, y: 38, w: 300, h: 200 };
    info = { x: margin, y: 270, w: w * 0.58, h: 130 };
    chart = { x: margin + w * 0.58 + 8, y: 270, w: w * 0.42 - 8, h: 130 };
    status = { x: margin, y: 406, w, h: 54 };
  } else {
    const pw = Math.floor(canvasWidth * 0.56) - 34;
    plot = { x: margin + 34, y: 46, w: pw, h: Math.round(pw * W2_MAX / W1_MAX) };
    chart = { x: margin, y: plot.y + plot.h + 36, w: plot.x + plot.w - margin, h: drawHeight - plot.y - plot.h - 44 };
    info = { x: plot.x + plot.w + 14, y: 46, w: canvasWidth - margin - plot.x - plot.w - 14, h: 262 };
    status = { x: info.x, y: 316, w: info.w, h: drawHeight - 324 };
  }
  drawMap(narrow);
  drawInfo(info, ts, narrow);
  drawChart(chart, narrow);

  const [msg, col] = statusMessage();
  fill('white');
  stroke(col);
  strokeWeight(1.5);
  rect(status.x, status.y, status.w, status.h, 10);
  noStroke();
  fill(col);
  textAlign(LEFT, TOP);
  textSize(narrow ? 11 : 15);
  textWrap(WORD);
  text(msg, status.x + 10, status.y + 8, status.w - 20, status.h - 10);

  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Learning rate η: ' + nf(rateSlider.value(), 1, 2), 10, drawHeight + 56);
}

// Banded colors for the cost: light at the bottom of the surface, dark at the top
function buildHeat() {
  const L = LANDSCAPES[li], cell = 4;
  if (heat) heat.remove();
  heat = createGraphics(Math.ceil(plot.w), Math.ceil(plot.h));
  heat.noStroke();
  const jMax = Math.max(L.f(W1_MAX, W2_MAX), L.f(-W1_MAX, W2_MAX));
  const low = color('lightyellow'), mid = color('mediumaquamarine'), high = color('steelblue');
  for (let cx = 0; cx < plot.w; cx += cell) {
    for (let cy = 0; cy < plot.h; cy += cell) {
      const a = ((cx + cell / 2) / plot.w * 2 - 1) * W1_MAX, b = (1 - (cy + cell / 2) / plot.h * 2) * W2_MAX;
      const t = Math.floor(Math.pow(constrain((L.f(a, b) - L.jMin) / (jMax - L.jMin), 0, 1), L.power) * BANDS) / (BANDS - 1);
      heat.fill(t < 0.5 ? lerpColor(low, mid, t * 2) : lerpColor(mid, high, min(1, t * 2 - 1)));
      heat.rect(cx, cy, cell, cell);
    }
  }
}

// Contour map with the minima, the path so far, and an arrow for the next step
function drawMap(narrow) {
  const L = LANDSCAPES[li], p = plot, key = li + '|' + p.w + '|' + p.h;
  if (key !== heatKey) { buildHeat(); heatKey = key; }
  const gx = v => p.x + (v + W1_MAX) / (2 * W1_MAX) * p.w, gy = v => p.y + (W2_MAX - v) / (2 * W2_MAX) * p.h;
  image(heat, p.x, p.y, p.w, p.h);
  noFill();
  stroke('gray');
  strokeWeight(1);
  rect(p.x, p.y, p.w, p.h);
  noStroke();
  fill('dimgray');
  textSize(narrow ? 11 : 12);
  textAlign(CENTER, TOP);
  for (let v = -W1_MAX; v <= W1_MAX; v++) text(v, gx(v), p.y + p.h + 3);
  textAlign(RIGHT, CENTER);
  for (let v = -W2_MAX; v <= W2_MAX; v++) text(v, p.x - 4, gy(v));
  fill('black');
  textAlign(CENTER, TOP);
  text('weight w₁', p.x + p.w / 2, p.y + p.h + 16);
  push();
  translate(p.x - (narrow ? 22 : 26), p.y + p.h / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, CENTER);
  text('weight w₂', 0, 0);
  pop();

  push();
  drawingContext.beginPath();
  drawingContext.rect(p.x, p.y, p.w, p.h);
  drawingContext.clip();
  for (const m of L.minima) {                 // minima: a cross, labeled when there is more than one
    stroke('black');
    strokeWeight(2);
    line(gx(m.w[0]) - 5, gy(m.w[1]), gx(m.w[0]) + 5, gy(m.w[1]));
    line(gx(m.w[0]), gy(m.w[1]) - 5, gx(m.w[0]), gy(m.w[1]) + 5);
    noStroke();
    fill('black');
    textAlign(CENTER, TOP);
    text(L.minima.length === 1 ? 'minimum' : (m.J === L.jMin ? 'global min' : 'local min'), gx(m.w[0]), gy(m.w[1]) + 8);
  }
  const cur = path[path.length - 1];
  for (const [col, wt] of [['white', 4], ['crimson', 2]]) {      // path with a white outline
    noFill();
    stroke(col);
    strokeWeight(wt);
    beginShape();
    for (const pt of path) if (isFinite(pt.J)) vertex(gx(pt.w[0]), gy(pt.w[1]));
    endShape();
  }
  stroke('white');
  strokeWeight(1);
  fill('crimson');
  for (const pt of path) if (isFinite(pt.J)) circle(gx(pt.w[0]), gy(pt.w[1]), 6);
  if (isFinite(cur.J)) {
    if (state === 'ready' || state === 'moving') {             // arrow: the next step, -eta * gradient, at true scale
      const eta = rateSlider.value(), x0 = gx(cur.w[0]), y0 = gy(cur.w[1]);
      const x1 = gx(cur.w[0] - eta * cur.g[0]), y1 = gy(cur.w[1] - eta * cur.g[1]), ang = atan2(y1 - y0, x1 - x0);
      stroke('black');
      strokeWeight(2);
      line(x0, y0, x1, y1);
      if (dist(x0, y0, x1, y1) > 6) {
        line(x1, y1, x1 - 8 * cos(ang - 0.45), y1 - 8 * sin(ang - 0.45));
        line(x1, y1, x1 - 8 * cos(ang + 0.45), y1 - 8 * sin(ang + 0.45));
      }
    }
    stroke('white');
    strokeWeight(2);
    fill('crimson');
    circle(gx(cur.w[0]), gy(cur.w[1]), 13);
  }
  pop();
}

// Numbers for the current step
function drawInfo(r, ts, narrow) {
  const L = LANDSCAPES[li], cur = path[path.length - 1], eta = rateSlider.value();
  const f = v => nf(Math.abs(v) < 0.0005 ? 0 : v, 1, 3);        // never print -0.000
  const pair = (a, b) => '(' + f(a) + ', ' + f(b) + ')';
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(r.x, r.y, r.w, r.h, 10);
  noStroke();
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  fill('black');
  text('Step ' + (path.length - 1), r.x + 10, r.y + 7);
  textStyle(NORMAL);
  textSize(narrow ? 11 : 15);
  const lines = [[L.formula, 'dimgray'], [L.gradient, 'dimgray'],
    ['Position w = ' + pair(cur.w[0], cur.w[1]), 'crimson'],
    ['Cost J = ' + f(cur.J), 'black'],
    ['Gradient ∇J = ' + pair(cur.g[0], cur.g[1]), 'black'],
    ['Steepness |∇J| = ' + f(Math.hypot(cur.g[0], cur.g[1])), 'black'],
    ['Next step −η∇J = ' + pair(-eta * cur.g[0], -eta * cur.g[1]), 'black']];
  const lh = (r.h - 30) / lines.length;
  lines.forEach(([str, col], i) => { fill(col); text(str, r.x + 10, r.y + 28 + i * lh); });
}

// Cost at every step so far
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
  text('Cost J at each step', r.x + 10, r.y + 6);
  textStyle(NORMAL);
  const costs = path.map(pt => pt.J).filter(isFinite);
  const lo = Math.min(0, ...costs), hi = Math.max(...costs, lo + 0.001), n = Math.max(10, costs.length - 1);
  const px = r.x + 44, py = r.y + 28, pw = r.w - 58, ph = r.h - 48;
  const gx = k => px + k / n * pw, gy = v => py + ph - (v - lo) / (hi - lo) * ph;
  stroke('gray');
  line(px, py, px, py + ph);
  line(px, gy(0), px + pw, gy(0));
  noStroke();
  fill('dimgray');
  textSize(narrow ? 11 : 12);
  textAlign(RIGHT, CENTER);
  text(hi >= 100 ? round(hi) : nf(hi, 1, 2), px - 4, py + 3);
  text('0', px - 4, gy(0));
  textAlign(CENTER, TOP);
  text('0', px, py + ph + 3);
  text(n + ' steps', px + pw - 16, py + ph + 3);
  noFill();
  stroke('crimson');
  strokeWeight(2);
  beginShape();
  costs.forEach((v, k) => vertex(gx(k), gy(v)));
  endShape();
  stroke('white');
  strokeWeight(1);
  fill('crimson');
  circle(gx(costs.length - 1), gy(costs[costs.length - 1]), 8);
}

// A click on the map sets a new starting point
function mousePressed() {
  const p = plot;
  if (!p || mouseX < p.x || mouseX > p.x + p.w || mouseY < p.y || mouseY > p.y + p.h) return;
  start = [((mouseX - p.x) / p.w * 2 - 1) * W1_MAX, (1 - (mouseY - p.y) / p.h * 2) * W2_MAX];
  resetRun();
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
