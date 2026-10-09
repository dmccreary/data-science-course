// Activation Function Explorer
// CANVAS_HEIGHT: 590
// Bloom L2-L3 (Understand, Apply): students choose an activation function, move the input x
// along its curve, and read the output f(x) and the gradient f'(x) to find where gradients
// vanish or die. A two-neuron network shows why a linear activation adds nothing.
//
// Functions and derivatives (every derivative is drawn from its own formula):
//   Sigmoid     f = 1 / (1 + e^-x)          f' = f (1 - f)           range (0, 1)
//   Tanh        f = tanh(x)                 f' = 1 - f^2             range (-1, 1)
//   ReLU        f = max(0, x)               f' = 1 if x > 0 else 0   range [0, inf)
//   Leaky ReLU  f = x if x > 0 else 0.1 x   f' = 1 if x > 0 else 0.1
//   Step        f = 1 if x > 0 else 0       f' = 0, undefined at 0   (the perceptron)
//   Linear      f = x                       f' = 1
// At the corner x = 0, ReLU and Leaky ReLU use the left-hand slope, as PyTorch autograd does.
// Checked against PyTorch 2.10.0: values from the nn modules named below, slopes from autograd.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 510;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let sliderLeftMargin = 130;
let defaultTextSize = 16;

const X_MAX = 5;                    // the x axis runs from -5 to 5
const Y_MIN = -1.5, Y_MAX = 2.5;    // fixed y axis, so every curve is drawn to the same scale
const LEAK = 0.1;                   // Leaky ReLU slope for x <= 0 (PyTorch's default, 0.01, is too small to see)
const FLAT = 0.05;                  // a gradient smaller than this is shaded as "vanishing"
const LAYERS = 10;                  // depth used for the gradient-through-many-layers line
const sigmoid = x => 1 / (1 + Math.exp(-x));
const FUNCS = [
  { name: 'Sigmoid', color: 'royalblue', use: 'output layer for a yes/no probability', f: sigmoid, d: x => sigmoid(x) * (1 - sigmoid(x)),
    formula: 'f(x) = 1 / (1 + e^(−x))', dform: 'f′(x) = f(x) (1 − f(x))', range: '(0, 1)', torch: 'nn.Sigmoid()' },
  { name: 'Tanh', color: 'seagreen', use: 'hidden layers, centered at 0', f: Math.tanh, d: x => 1 - Math.tanh(x) ** 2,
    formula: 'f(x) = (e^x − e^(−x)) / (e^x + e^(−x))', dform: 'f′(x) = 1 − f(x)²', range: '(−1, 1)', torch: 'nn.Tanh()' },
  { name: 'ReLU', color: 'crimson', use: 'the default for hidden layers', f: x => Math.max(0, x), d: x => (x > 0 ? 1 : 0),
    formula: 'f(x) = max(0, x)', dform: 'f′(x) = 1 if x > 0, else 0', range: '[0, ∞)', torch: 'nn.ReLU()' },
  { name: 'Leaky ReLU', color: 'darkorange', use: 'hidden layers, fixes dying ReLU', f: x => (x > 0 ? x : LEAK * x), d: x => (x > 0 ? 1 : LEAK),
    formula: 'f(x) = x if x > 0, else ' + LEAK + 'x', dform: 'f′(x) = 1 if x > 0, else ' + LEAK, range: '(−∞, ∞)', torch: 'nn.LeakyReLU(' + LEAK + ')' },
  { name: 'Step', color: 'saddlebrown', use: 'the original perceptron', f: x => (x > 0 ? 1 : 0), d: x => (x === 0 ? NaN : 0),
    formula: 'f(x) = 1 if x > 0, else 0', dform: 'f′(x) = 0, undefined at x = 0', range: '{0, 1}', torch: '(x > 0).float()' },
  { name: 'Linear', color: 'dimgray', use: 'regression output layer only', f: x => x, d: () => 1,
    formula: 'f(x) = x', dform: 'f′(x) = 1', range: '(−∞, ∞)', torch: 'nn.Identity()' }
];
// A small network with one hidden layer of two neurons and a linear output: y = f(x + 2) - f(2x - 2)
const network = (f, x) => f(x + 2) - f(2 * x - 2);

let fnSelect, cmpSelect, derivBox, xSlider;
let plotBox = null;                 // plotting rectangle of the last frame, for click-to-set-x

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  fnSelect = createSelect();
  fnSelect.parent(mainElement);
  fnSelect.position(10, drawHeight + 8);
  fnSelect.style('font-size', '15px');
  FUNCS.forEach(f => fnSelect.option(f.name));

  cmpSelect = createSelect();
  cmpSelect.parent(mainElement);
  cmpSelect.position(122, drawHeight + 8);
  cmpSelect.style('font-size', '15px');
  cmpSelect.option('Compare: none', '');
  FUNCS.forEach(f => cmpSelect.option('vs ' + f.name, f.name));

  derivBox = createCheckbox(' Show derivative', true);
  derivBox.parent(mainElement);
  derivBox.position(262, drawHeight + 8);
  derivBox.style('font-size', '16px');

  xSlider = createSlider(-X_MAX, X_MAX, 1, 0.1);
  xSlider.parent(mainElement);
  xSlider.position(sliderLeftMargin, drawHeight + 45);
  xSlider.size(canvasWidth - sliderLeftMargin - margin);

  describe('A graph of an activation function and its derivative for inputs from minus 5 to 5. A menu chooses sigmoid, ' +
    'tanh, ReLU, leaky ReLU, step, or linear, and a slider moves the input x. Panels give the formula, the output range, ' +
    'the value and gradient at x, and whether the gradient is healthy, vanishing, or zero. A small plot shows the output ' +
    'of a two-neuron network built with the chosen function.', LABEL);
}

// 'zero' = no gradient passes back, 'flat' = almost none, 'ok' = a usable gradient
function gradKind(d) {
  return isNaN(d) || d === 0 ? 'zero' : Math.abs(d) < FLAT ? 'flat' : 'ok';
}

function num(v) { return isNaN(v) ? 'undefined' : v.toFixed(4).replace('-', '−'); }

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600, w = canvasWidth - 2 * margin;
  const fn = FUNCS.find(f => f.name === fnSelect.value());
  const cmp = FUNCS.find(f => f.name === cmpSelect.value() && f !== fn) || null;
  const x = xSlider.value(), showD = derivBox.checked();
  textWrap(WORD);

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Activation Function Explorer', canvasWidth / 2, 8);

  if (narrow) {
    const bw = Math.round(w * 0.56);
    drawPlot(margin, 36, w, 200, fn, cmp, x, showD, true);
    drawInfo(margin, 242, w, 78, fn, true);
    drawReadout(margin, 326, bw, 176, fn, cmp, x, true);
    drawNetwork(margin + bw + 6, 326, w - bw - 6, 176, fn, true);
  } else {
    const lw = Math.round(w * 0.56), rx = margin + lw + 10, rw = w - lw - 10;
    drawPlot(margin, 44, lw, drawHeight - 52, fn, cmp, x, showD, false);
    drawInfo(rx, 44, rw, 104, fn, false);
    drawReadout(rx, 156, rw, 172, fn, cmp, x, false);
    drawNetwork(rx, 336, rw, drawHeight - 344, fn, false);
  }

  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Input x: ' + nf(x, 1, 1).replace('-', '−'), 10, drawHeight + 56);
}

function panel(x0, y0, w, h, bg) {
  fill(bg);
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
}

// One curve from x = -5 to 5. The line is broken wherever the function jumps or is undefined.
function plotCurve(g, gx, gy, dashed) {
  noFill();
  drawingContext.setLineDash(dashed ? [6, 5] : []);
  let prev = NaN, open = false;
  for (let i = 0; i <= 400; i++) {
    const v = -X_MAX + i * X_MAX / 200, y = g(v);
    if (open && (isNaN(y) || Math.abs(y - prev) > 0.3)) { endShape(); open = false; }
    if (!open && !isNaN(y)) { beginShape(); open = true; }
    if (open) vertex(gx(v), gy(y));
    prev = y;
  }
  if (open) endShape();
  drawingContext.setLineDash([]);
}

// The main graph: f(x) solid, f'(x) dashed, an optional comparison curve, and the point at x
function drawPlot(x0, y0, w, h, fn, cmp, x, showD, narrow) {
  panel(x0, y0, w, h, 'white');
  const px = x0 + 32, py = y0 + 12, pw = w - 46, ph = h - (narrow ? 44 : 50);
  const gx = v => px + (v + X_MAX) / (2 * X_MAX) * pw;
  const gy = v => py + (Y_MAX - v) / (Y_MAX - Y_MIN) * ph;
  const ts = narrow ? 11 : 13;
  plotBox = { px, py, pw, ph };

  // shade the inputs where the chosen function passes back no gradient (red) or almost none (orange)
  const found = { zero: false, flat: false };
  let start = -X_MAX, kind = gradKind(fn.d(-X_MAX));
  noStroke();
  for (let i = 1; i <= 200; i++) {
    const v = -X_MAX + i * X_MAX / 100, k = i === 200 ? 'end' : gradKind(fn.d(v));
    if (k === kind) continue;
    if (kind !== 'ok') {
      found[kind] = true;
      fill(kind === 'zero' ? color(220, 20, 60, 28) : color(255, 165, 0, 50));
      rect(gx(start), py, gx(v) - gx(start), ph);
    }
    start = v;
    kind = k;
  }

  // grid, axes through the origin, tick labels
  textSize(narrow ? 11 : 12);
  for (let v = -X_MAX; v <= X_MAX; v++) {
    stroke(v === 0 ? 'gray' : 'gainsboro');
    strokeWeight(1);
    line(gx(v), py, gx(v), py + ph);
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    text(String(v).replace('-', '−'), gx(v), py + ph + 4);
  }
  for (let v = Math.ceil(Y_MIN); v <= Y_MAX; v++) {
    stroke(v === 0 ? 'gray' : 'gainsboro');
    line(px, gy(v), px + pw, gy(v));
    noStroke();
    fill('dimgray');
    textAlign(RIGHT, CENTER);
    text(String(v).replace('-', '−'), px - 5, gy(v));
  }
  fill('black');
  textAlign(CENTER, TOP);
  text('input x  (click the graph or use the slider)', px + pw / 2, py + ph + (narrow ? 17 : 21));

  // curves, clipped to the plotting rectangle
  drawingContext.save();
  drawingContext.beginPath();
  drawingContext.rect(px, py, pw, ph);
  drawingContext.clip();
  if (cmp) {
    stroke(cmp.color);
    strokeWeight(1.5);
    plotCurve(cmp.f, gx, gy, false);
    if (showD) plotCurve(cmp.d, gx, gy, true);
  }
  stroke(fn.color);
  strokeWeight(2);
  if (showD) plotCurve(fn.d, gx, gy, true);
  strokeWeight(3.5);
  plotCurve(fn.f, gx, gy, false);
  drawingContext.restore();

  // the chosen input: a vertical guide, a ring on f'(x), and a dot on f(x)
  const fx = fn.f(x), dx = fn.d(x), X = gx(x);
  stroke('dimgray');
  strokeWeight(1);
  drawingContext.setLineDash([3, 3]);
  line(X, py, X, py + ph);
  drawingContext.setLineDash([]);
  if (showD && !isNaN(dx)) {
    stroke(fn.color);
    strokeWeight(2);
    fill('white');
    circle(X, gy(dx), 10);
  }
  stroke('white');
  strokeWeight(1.5);
  fill(fn.color);
  if (fx > Y_MAX || fx < Y_MIN) {           // the point is off the chart: an arrow at the edge shows which way
    const ey = fx > Y_MAX ? py : py + ph, s = fx > Y_MAX ? 1 : -1;
    triangle(X - 8, ey + 13 * s, X + 8, ey + 13 * s, X, ey);
  } else circle(X, gy(fx), 13);

  // legends sit where no curve goes: above 1 on the left, below 0 on the right
  textSize(ts);
  const rows = [[fn.color, 3, false, fn.name + ' f(x)']];
  if (showD) rows.push([fn.color, 2, true, 'derivative f′(x)']);
  if (cmp) rows.push([cmp.color, 1.5, false, 'compare: ' + cmp.name]);
  rows.forEach(([c, sw, dashed, label], i) => {
    const ly = py + 12 + i * (ts + 4);
    stroke(c);
    strokeWeight(sw);
    drawingContext.setLineDash(dashed ? [6, 5] : []);
    line(px + 8, ly, px + 34, ly);
    drawingContext.setLineDash([]);
    noStroke();
    fill('black');
    textAlign(LEFT, CENTER);
    text(label, px + 40, ly);
  });
  const shades = [];
  if (found.flat) shades.push([color(255, 165, 0, 110), 'f′(x) < ' + FLAT + ': vanishing gradient']);
  if (found.zero) shades.push([color(220, 20, 60, 70), 'f′(x) = 0: no gradient']);
  shades.forEach(([c, label], i) => {
    const ly = py + ph - 12 - (shades.length - 1 - i) * (ts + 4);
    noStroke();
    fill('black');
    textAlign(RIGHT, CENTER);
    text(label, px + pw - 8, ly);
    fill(c);
    stroke('silver');
    strokeWeight(1);
    rect(px + pw - 26 - textWidth(label), ly - 6, 13, 12);
  });
}

// Name, typical use, formula, derivative formula, range, and the PyTorch equivalent
function drawInfo(x0, y0, w, h, fn, narrow) {
  panel(x0, y0, w, h, 'white');
  const ts = narrow ? 12 : 14, lh = narrow ? 16 : 21;
  noStroke();
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 2);
  fill(fn.color);
  text(fn.name, x0 + 10, y0 + 7);
  const nameW = textWidth(fn.name);
  textStyle(NORMAL);
  textSize(ts);
  fill('dimgray');
  text(fn.use, x0 + 20 + nameW, y0 + 9);
  fill('black');
  [fn.formula, fn.dform, 'Range ' + fn.range + '      PyTorch: ' + fn.torch].forEach((s, i) => text(s, x0 + 10, y0 + 10 + lh * (i + 1)));
}

// Output and gradient at the chosen x, what that gradient means, and its effect through many layers
function drawReadout(x0, y0, w, h, fn, cmp, x, narrow) {
  panel(x0, y0, w, h, 'lightyellow');
  const ts = narrow ? 12 : 14, lh = narrow ? 16 : 21, tx = x0 + 10;
  const dx = fn.d(x), kind = gradKind(dx);
  let msg = 'Healthy gradient: the error signal passes back through this neuron.', tone = 'darkgreen';
  if (fn.name === 'Step') { msg = 'Zero gradient everywhere: gradient descent cannot train a step function.'; tone = 'firebrick'; }
  else if (fn.name === 'Linear') { msg = 'The gradient is always 1, but a straight line adds no bend, so extra layers add nothing.'; tone = 'dimgray'; }
  else if (kind === 'zero') { msg = 'Zero gradient: no error signal passes back. A neuron stuck here is a dead ReLU.'; tone = 'firebrick'; }
  else if (kind === 'flat') { msg = 'Vanishing gradient: the curve is nearly flat here, so almost no error signal passes back.'; tone = 'chocolate'; }
  else if (dx === LEAK) { msg = 'Small gradient, but never zero, so the neuron can still learn. This fixes dying ReLU.'; }

  noStroke();
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  fill('black');
  text('At x = ' + nf(x, 1, 1).replace('-', '−'), tx, y0 + 7);
  textSize(ts);
  fill(fn.color);
  let y = y0 + 9 + lh;
  text('f(x) = ' + num(fn.f(x)) + '      f′(x) = ' + num(dx), tx, y);
  textStyle(NORMAL);
  if (cmp) {
    y += lh;
    fill(cmp.color);
    text(cmp.name + ': f(x) = ' + num(cmp.f(x)) + ', f′(x) = ' + num(cmp.d(x)), tx, y);
  }
  y += lh + 2;
  fill(tone);
  text(msg, tx, y, w - 20, h - (y - y0) - lh - 6);
  // backpropagation multiplies one such factor per layer
  const chain = isNaN(dx) ? 0 : Math.pow(dx, LAYERS);
  fill('black');
  textAlign(LEFT, BOTTOM);
  text((narrow ? '' : 'Through ') + LAYERS + ' such layers: f′(x)^' + LAYERS + ' = ' +
    (chain === 0 || chain >= 0.0001 ? num(chain) : chain.toExponential(1).replace('-', '−')), tx, y0 + h - 7);
}

// Output of the two-neuron network for x from -5 to 5, scaled to fill its box
function drawNetwork(x0, y0, w, h, fn, narrow) {
  panel(x0, y0, w, h, 'white');
  const ts = narrow ? 11 : 13, capH = narrow ? 58 : 38;
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text(narrow ? 'Two-neuron network' : 'Why non-linearity matters', x0 + 10, y0 + 7);
  textStyle(NORMAL);
  textSize(ts);
  text((narrow ? '' : 'Two-neuron network: ') + 'y = f(x + 2) − f(2x − 2)', x0 + 10, y0 + ts + 13);

  const px = x0 + 10, py = y0 + 2 * ts + 20, pw = w - 20, ph = h - (py - y0) - capH - 4;
  const ys = [];
  for (let i = 0; i <= 200; i++) ys.push(network(fn.f, -X_MAX + i * X_MAX / 100));
  const lo = Math.min(...ys), hi = Math.max(...ys);
  fill('aliceblue');
  stroke('gainsboro');
  strokeWeight(1);
  rect(px, py, pw, ph);
  noFill();
  stroke(fn.color);
  strokeWeight(2.5);
  beginShape();
  ys.forEach((v, i) => vertex(px + i / 200 * pw, py + ph - 5 - (v - lo) / (hi - lo) * (ph - 10)));
  endShape();

  noStroke();
  fill(fn.name === 'Linear' ? 'firebrick' : 'black');
  textAlign(LEFT, TOP);
  // non-breaking spaces keep "4 − x" on one line
  text(fn.name === 'Linear' ? 'Linear: (x + 2) − (2x − 2) =\u00a04\u00a0−\u00a0x. The two layers collapse into one straight line.'
    : fn.name + ' bends the output. With Linear the same network is only the line y\u00a0=\u00a04\u00a0−\u00a0x.', x0 + 10, py + ph + 5, w - 20, capH);
}

function setXFromMouse() {
  const p = plotBox;
  if (!p || mouseX < p.px || mouseX > p.px + p.pw || mouseY < p.py || mouseY > p.py + p.ph) return;
  xSlider.value(Math.round(((mouseX - p.px) / p.pw * 2 - 1) * X_MAX * 10) / 10);
}

function mousePressed() { setXFromMouse(); }

function mouseDragged() { setXFromMouse(); }

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  xSlider.size(canvasWidth - sliderLeftMargin - margin);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
