// Optimization Landscape Explorer
// CANVAS_HEIGHT: 580
// Bloom L4-L5 (Analyze, Evaluate): students predict where gradient descent will stop from a
// chosen starting point, run it, and compare the outcomes of many starts with and without
// momentum to see which starts reach the global minimum.
//
// Each landscape is an explicit cost J(x) of one parameter, with derivative J'(x):
//   Convex bowl    J = 0.5 x^2                        one minimum at x = 0
//   Two valleys    J = x^4 - 3x^2 + 0.5x              global minimum near x = -1.26, local near x = 1.18
//   Many valleys   J = 0.2 x^2 - 0.7 cos(3(x - 0.4))  five minima, global near x = 0.38
//   Plateau        J = x^4/4 - 2x^3/3                 minimum at x = 2; flat inflection point at x = 0,
//                                                     where J' = 0 and J'' = 0 (the 1-D cousin of a saddle)
// Minima are located numerically at startup (sign change of J', then bisection).
// Plain gradient descent:  x = x - eta J'(x).   With momentum:  v = beta v - eta J'(x),  x = x + v.
// A run stops when |J'| < 0.002 and the last move was shorter than 0.0005, or after 600 steps.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 500;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const BETA = 0.9, SLOPE_TOL = 0.002, MOVE_TOL = 0.0005, MAX_STEPS = 600;
const LANDSCAPES = [
  { name: 'Convex bowl', formula: 'J(x) = 0.5x²', lo: -3, hi: 3, eta: 0.1, start: 2.5, flats: [],
    f: x => 0.5 * x * x, d: x => x },
  { name: 'Two valleys', formula: 'J(x) = x⁴ − 3x² + 0.5x', lo: -2, hi: 2, eta: 0.02, start: 1.95, flats: [],
    f: x => x ** 4 - 3 * x * x + 0.5 * x, d: x => 4 * x ** 3 - 6 * x + 0.5 },
  { name: 'Many valleys', formula: 'J(x) = 0.2x² − 0.7 cos(3(x − 0.4))', lo: -5, hi: 5, eta: 0.05, start: 3.3, flats: [],
    f: x => 0.2 * x * x - 0.7 * Math.cos(3 * (x - 0.4)), d: x => 0.4 * x + 2.1 * Math.sin(3 * (x - 0.4)) },
  { name: 'Plateau', formula: 'J(x) = x⁴/4 − 2x³/3', lo: -1.6, hi: 3, eta: 0.1, start: -1.4, flats: [0],
    f: x => x ** 4 / 4 - 2 * x ** 3 / 3, d: x => x ** 3 - 2 * x * x }
];

let li = 1;                 // current landscape
let startX = 1.95;
let path = [];              // positions visited in this run
let velocity = 0;
let outcome = null;         // null while the run is unfinished, else 'global', 'local', 'flat', or 'cap'
let records = [];           // finished runs on this landscape: { x0, momentum, outcome }
let running = false, lastStepFrame = 0, plot = null;
let startButton, stepButton, resetButton, landscapeSelect, momentumCheckbox;

// Minima of every landscape, and the vertical range needed to draw it
function analyzeLandscapes() {
  for (const L of LANDSCAPES) {
    const n = 4000, h = (L.hi - L.lo) / n;
    L.minima = [];
    let yLo = Infinity, yHi = -Infinity;
    for (let i = 0; i <= n; i++) {
      const x = L.lo + i * h, xNext = L.lo + (i + 1) * h;
      yLo = Math.min(yLo, L.f(x));
      yHi = Math.max(yHi, L.f(x));
      if (i < n && L.d(x) < 0 && L.d(xNext) >= 0) {       // slope turns from downhill to uphill
        let a = x, b = xNext;
        for (let k = 0; k < 50; k++) { const m = (a + b) / 2; if (L.d(m) < 0) a = m; else b = m; }
        L.minima.push({ x: (a + b) / 2, J: L.f((a + b) / 2) });
      }
    }
    L.best = L.minima.reduce((p, q) => q.J < p.J ? q : p);
    L.yLo = yLo - 0.32 * (yHi - yLo);              // room under the lowest valley for its label
    L.yHi = yHi + 0.04 * (yHi - yLo);
  }
}

function resetBall() {
  path = [startX];
  velocity = 0;
  outcome = null;
  running = false;
  updateButtons();
}

// One update of the optimizer. Without momentum the velocity is just the step -eta J'(x).
function stepOnce() {
  if (outcome) return;
  const L = LANDSCAPES[li], x = path[path.length - 1], useMomentum = momentumCheckbox.checked();
  velocity = (useMomentum ? BETA * velocity : 0) - L.eta * L.d(x);
  const next = x + velocity;
  path.push(next);
  if (Math.abs(L.d(next)) < SLOPE_TOL && Math.abs(velocity) < MOVE_TOL) {
    const near = L.minima.find(m => Math.abs(m.x - next) < 0.1);
    outcome = !near ? 'flat' : near === L.best ? 'global' : 'local';
  } else if (path.length > MAX_STEPS) outcome = 'cap';
  if (outcome) {
    running = false;
    if (!records.some(r => r.x0 === startX && r.momentum === useMomentum)) records.push({ x0: startX, momentum: useMomentum, outcome });
    updateButtons();
  }
}

function updateButtons() {
  startButton.html(running ? 'Pause' : 'Start');
  if (outcome) stepButton.attribute('disabled', ''); else stepButton.removeAttribute('disabled');
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
  // Start after a finished run repeats it from the same starting point
  startButton = makeButton('Start', 10, () => { if (outcome) resetBall(); running = !running; lastStepFrame = frameCount; updateButtons(); });
  stepButton = makeButton('Step', 66, () => { running = false; stepOnce(); updateButtons(); });
  resetButton = makeButton('Reset', 116, () => { startX = LANDSCAPES[li].start; records = []; resetBall(); });
  landscapeSelect = createSelect();
  landscapeSelect.parent(mainElement);
  landscapeSelect.position(178, drawHeight + 10);
  LANDSCAPES.forEach(L => landscapeSelect.option(L.name));
  landscapeSelect.selected(LANDSCAPES[li].name);
  landscapeSelect.style('font-size', '15px');
  landscapeSelect.changed(() => {
    li = LANDSCAPES.findIndex(L => L.name === landscapeSelect.value());
    startX = LANDSCAPES[li].start;
    records = [];
    resetBall();
  });
  momentumCheckbox = createCheckbox(' Add momentum (β = ' + BETA + ')', false);
  momentumCheckbox.parent(mainElement);
  momentumCheckbox.position(10, drawHeight + 46);
  momentumCheckbox.style('font-size', '16px');
  momentumCheckbox.changed(resetBall);

  analyzeLandscapes();
  resetBall();

  describe('A cost curve with one or more valleys and a ball that gradient descent moves downhill from a starting point chosen by ' +
    'clicking the curve. Each minimum is labeled with its cost. Panels list the position, cost, and slope at every step and say ' +
    'whether the ball reached the global minimum, a local minimum, or a flat region, with and without momentum.', LABEL);
}

function draw() {
  updateCanvasSize();
  if (running && frameCount - lastStepFrame >= (path.length > 60 ? 1 : 3)) { stepOnce(); lastStepFrame = frameCount; }

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const L = LANDSCAPES[li], narrow = canvasWidth < 600, w = canvasWidth - 2 * margin, ts = narrow ? 11 : 14;
  const f = v => nf(Math.abs(v) < 0.0005 ? 0 : v, 1, 3);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textStyle(NORMAL);
  textSize(narrow ? 18 : 24);
  text('Optimization Landscape Explorer', canvasWidth / 2, 8);
  fill('dimgray');
  textSize(narrow ? 11 : 14);
  text(L.formula + '      learning rate η = ' + L.eta, canvasWidth / 2, narrow ? 31 : 38);

  const left = narrow ? 62 : 84;
  plot = { x: margin + left, y: 60, w: w - left - 6, h: 228 };
  drawLandscape(L, narrow, f);
  drawRecords(L, narrow);

  // numbers for the current step
  const x = path[path.length - 1], k = path.length - 1, slope = L.d(x), useMomentum = momentumCheckbox.checked();
  const panelY = 366, panelH = drawHeight - panelY - 8, leftW = narrow ? w * 0.44 : w * 0.38;
  const lines = [['Position x = ' + f(x), 'crimson'], ['Cost J(x) = ' + f(L.f(x)), 'black'], ['Slope J′(x) = ' + f(slope), 'black'],
    [useMomentum ? 'Velocity v = ' + f(velocity) : 'Next step −ηJ′ = ' + f(-L.eta * slope), 'black'],
    [(narrow ? 'Above global min: ' : 'Above the global minimum: ') + f(L.f(x) - L.best.J), 'black']];
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(margin, panelY, leftW, panelH, 10);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text('Step ' + k, margin + 10, panelY + 7);
  textStyle(NORMAL);
  textSize(ts);
  lines.forEach(([str, col], i) => { fill(col); text(str, margin + 10, panelY + 28 + i * (panelH - 32) / lines.length); });

  // prediction prompt, then the result and the tally of finished runs
  const best = 'x = ' + f(L.best.x), gapHere = f(L.f(x) - L.best.J);
  let msg, col = 'black';
  if (outcome === 'global') { msg = 'Found the global minimum! J = ' + f(L.best.J) + ' at ' + best + ', after ' + k + ' steps.'; col = 'green'; }
  else if (outcome === 'local') { msg = 'Stuck in a local minimum! J = ' + f(L.f(x)) + ' at x = ' + f(x) + ', which is ' + gapHere + ' above the global minimum at ' + best + '.'; col = 'firebrick'; }
  else if (outcome === 'flat') { msg = 'Stalled on a flat region. The slope is nearly zero at x = ' + f(x) + ', but this is not a minimum: the cost is ' + gapHere + ' above the valley at ' + best + '.'; col = 'firebrick'; }
  else if (outcome === 'cap') { msg = 'Still moving after ' + MAX_STEPS + ' steps.'; col = 'darkorange'; }
  else if (k === 0) msg = 'Predict first: from x = ' + nf(startX, 1, 2) + ', where will the ball stop? Then press Start. Click the curve to try another starting point.';
  else msg = L.f(x) > L.f(path[k - 1]) ? 'Momentum is carrying the ball uphill: the cost rose on this step.' : 'Moving downhill: the cost fell on this step.';
  const tally = m => {
    const r = records.filter(q => q.momentum === m);
    return r.length ? r.filter(q => q.outcome === 'global').length + ' of ' + r.length + (r.length === 1 ? ' run' : ' runs') : 'no runs yet';
  };
  const rx = margin + leftW + 8, rw = w - leftW - 8;
  fill('white');
  stroke(col === 'black' ? 'silver' : col);
  strokeWeight(col === 'black' ? 1 : 1.5);
  rect(rx, panelY, rw, panelH, 10);
  noStroke();
  fill(col);
  textAlign(LEFT, TOP);
  textSize(narrow ? 11 : 15);
  textWrap(WORD);
  text(msg, rx + 10, panelY + 8, rw - 20, panelH - (narrow ? 50 : 44));
  fill('black');
  textSize(ts);
  text('Reached the global minimum. Plain: ' + tally(false) + '. With momentum: ' + tally(true) + '.',
    rx + 10, panelY + panelH - (narrow ? 44 : 40), rw - 20, 40);
}

// The cost curve, its minima, the path of this run, and the ball
function drawLandscape(L, narrow, f) {
  const p = plot, gx = v => p.x + (v - L.lo) / (L.hi - L.lo) * p.w, gy = v => p.y + p.h - (v - L.yLo) / (L.yHi - L.yLo) * p.h;
  fill('white');
  stroke('gray');
  strokeWeight(1);
  rect(p.x, p.y, p.w, p.h);
  noStroke();
  fill('dimgray');
  textSize(narrow ? 11 : 12);
  textAlign(CENTER, TOP);
  for (let v = Math.ceil(L.lo); v <= L.hi; v++) text(v, gx(v), p.y + p.h + 3);
  textAlign(RIGHT, CENTER);
  for (let v = Math.ceil(L.yLo); v <= L.yHi; v += (L.yHi - L.yLo > 6 ? 2 : 1)) text(v, p.x - 4, gy(v));
  fill('black');
  textAlign(RIGHT, TOP);
  text('parameter x', p.x + p.w - 4, p.y + p.h - (narrow ? 15 : 17));
  push();
  translate(p.x - (narrow ? 34 : 46), p.y + p.h / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, CENTER);
  text('cost J(x)', 0, 0);
  pop();

  push();
  drawingContext.beginPath();
  drawingContext.rect(p.x, p.y, p.w, p.h);
  drawingContext.clip();
  noFill();
  stroke('royalblue');
  strokeWeight(3);
  beginShape();
  for (let i = 0; i <= 300; i++) { const v = L.lo + i / 300 * (L.hi - L.lo); vertex(gx(v), gy(L.f(v))); }
  endShape();
  // minima and flat points, labeled with their cost
  textAlign(CENTER, TOP);
  const marks = L.minima.map(m => ({ x: m.x, label: m === L.best ? 'global' : 'local', col: m === L.best ? 'green' : 'darkorange' }))
    .concat(L.flats.map(v => ({ x: v, label: 'flat', col: 'dimgray' })));
  for (const m of marks) {
    stroke('white');
    strokeWeight(1.5);
    fill(m.col);
    triangle(gx(m.x), gy(L.f(m.x)) + 5, gx(m.x) - 6, gy(L.f(m.x)) + 15, gx(m.x) + 6, gy(L.f(m.x)) + 15);
    noStroke();
    textStyle(BOLD);
    text(m.label, gx(m.x), gy(L.f(m.x)) + 17);
    textStyle(NORMAL);
    text((narrow ? '' : 'J = ') + nf(L.f(m.x), 1, 2), gx(m.x), gy(L.f(m.x)) + (narrow ? 30 : 32));
  }
  // starting point, trail, next-step arrow, and the ball
  const x = path[path.length - 1], bx = gx(x), by = gy(L.f(x));
  stroke('crimson');
  strokeWeight(1.5);
  noFill();
  circle(gx(startX), gy(L.f(startX)), 16);
  noStroke();
  fill(220, 20, 60, 90);
  for (const v of path) circle(gx(v), gy(L.f(v)), 6);
  if (!outcome) {
    const nextV = (momentumCheckbox.checked() ? BETA * velocity : 0) - L.eta * L.d(x), ax = gx(x + nextV), dir = Math.sign(nextV);
    if (abs(ax - bx) > 5) {
      stroke('black');
      strokeWeight(2);
      line(bx, by, ax, by);
      line(ax, by, ax - 7 * dir, by - 5);
      line(ax, by, ax - 7 * dir, by + 5);
    }
  }
  stroke('white');
  strokeWeight(2);
  fill('crimson');
  circle(bx, by, 16);
  pop();
}

// One row of marks per optimizer: where each finished run started, colored by how it ended
function drawRecords(L, narrow) {
  const p = plot, gx = v => p.x + (v - L.lo) / (L.hi - L.lo) * p.w, top = p.y + p.h + 20;
  textSize(narrow ? 11 : 12);
  [['plain', false], [narrow ? 'momentum' : 'with momentum', true]].forEach(([label, m], i) => {
    const y = top + i * 16;
    fill('whitesmoke');
    stroke('gainsboro');
    strokeWeight(1);
    rect(p.x, y, p.w, 13);
    noStroke();
    fill('black');
    textAlign(RIGHT, CENTER);
    text(label, p.x - 5, y + 7);
    stroke('white');
    for (const r of records) if (r.momentum === m) { fill(r.outcome === 'global' ? 'green' : 'darkorange'); circle(gx(r.x0), y + 6.5, 10); }
  });
  noStroke();
  fill('dimgray');
  textAlign(LEFT, TOP);
  text(narrow ? 'Run starts: green reached the global minimum, orange did not.'
    : 'Starting points of finished runs: green reached the global minimum, orange stopped somewhere else.', narrow ? margin : p.x, top + 34);
}

// A click on the plot puts the ball on the curve at that x
function mousePressed() {
  const p = plot, L = LANDSCAPES[li];
  if (!p || mouseX < p.x || mouseX > p.x + p.w || mouseY < p.y || mouseY > p.y + p.h) return;
  startX = L.lo + (mouseX - p.x) / p.w * (L.hi - L.lo);
  resetBall();
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
