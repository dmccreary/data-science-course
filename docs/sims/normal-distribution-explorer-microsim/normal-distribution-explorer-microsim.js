// Normal Distribution Explorer
// CANVAS_HEIGHT: 595
// Bloom L3 (Apply): students set the mean and the standard deviation of a normal distribution
// and use the 68-95-99.7 rule to find the intervals that hold 68%, 95%, and 99.7% of the values.
//
// Model: the normal density with mean mu and standard deviation sigma is
//   f(x) = exp(-(x - mu)^2 / (2 sigma^2)) / (sigma sqrt(2 pi)).
// The area within k standard deviations of the mean is found by Simpson's rule on f, which gives
// 68.27%, 95.45%, and 99.73% for k = 1, 2, 3 whatever mu and sigma are. Both axes are fixed, so
// a change in mu slides the curve and a change in sigma visibly widens or narrows it.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 480;
let controlHeight = 115;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;
let labelWidth = 150;       // width of the label span in each slider row
let valueWidth = 40;        // width of the value span in each slider row

const X_MAX = 200;          // the x axis runs from 0 to 200
const Y_MAX = 0.055;        // the density axis is fixed (the tallest curve, sigma = 8, peaks at 0.0499)
const PRESETS = [
  { name: 'IQ scores', mu: 100, sigma: 15, axis: 'IQ score' },
  { name: 'Test scores', mu: 75, sigma: 10, axis: 'Test score' },
  { name: 'Heights in cm', mu: 170, sigma: 8, axis: 'Height (cm)' },
  { name: 'Custom', axis: 'x' }
];

let muRow, sigmaRow, presetSelect, regionsBox, pinButton;
let pinned = null;          // { mu, sigma } of the curve kept for comparison
let hoverX = null;          // x value under the pointer, or null

function pdf(x, mu, sigma) {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

// Area under the density from a to b by Simpson's rule with 200 strips
function areaBetween(a, b, mu, sigma) {
  const n = 200, h = (b - a) / n;
  let sum = pdf(a, mu, sigma) + pdf(b, mu, sigma);
  for (let i = 1; i < n; i++) sum += (i % 2 ? 4 : 2) * pdf(a + i * h, mu, sigma);
  return sum * h / 3;
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  muRow = makeSliderRow('Mean (μ)', 0, X_MAX, 100, 1, 0);
  sigmaRow = makeSliderRow('Std. deviation (σ)', 8, 50, 15, 1, 1);
  // moving a slider by hand leaves the preset
  muRow.slider.input(() => presetSelect.selected('Custom'));
  sigmaRow.slider.input(() => presetSelect.selected('Custom'));

  presetSelect = createSelect();
  presetSelect.parent(mainElement);
  presetSelect.position(10, drawHeight + 80);
  PRESETS.forEach(p => presetSelect.option(p.name));
  presetSelect.style('font-size', '15px');
  presetSelect.changed(() => {
    const p = PRESETS.find(q => q.name === presetSelect.value());
    if (p.mu !== undefined) { muRow.slider.value(p.mu); sigmaRow.slider.value(p.sigma); }
  });

  regionsBox = createCheckbox(' 68-95-99.7 regions', true);
  regionsBox.parent(mainElement);
  regionsBox.position(135, drawHeight + 80);
  regionsBox.style('font-size', '16px');

  pinButton = createButton('Pin Curve');
  pinButton.parent(mainElement);
  pinButton.position(310, drawHeight + 80);
  pinButton.mousePressed(() => {
    pinned = pinned ? null : { mu: muRow.slider.value(), sigma: sigmaRow.slider.value() };
    pinButton.html(pinned ? 'Unpin Curve' : 'Pin Curve');
  });
  resizeSliders();

  describe('A normal distribution curve on fixed axes. Sliders set the mean and the standard deviation. Shaded bands ' +
    'mark one, two, and three standard deviations from the mean, and a table gives each interval and the area ' +
    'inside it: 68.27, 95.45, and 99.73 percent. A curve can be pinned as a dashed outline for comparison.', LABEL);
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
  muRow.slider.size(w);
  sigmaRow.slider.size(w);
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const mu = muRow.slider.value(), sigma = sigmaRow.slider.value();
  muRow.valueSpan.html(mu);
  sigmaRow.valueSpan.html(sigma);
  const narrow = canvasWidth < 600;
  const ts = narrow ? 12 : 15, lh = narrow ? 16 : 22;
  const w = canvasWidth - 2 * margin, panelH = 4 * lh + 18;

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Normal Distribution Explorer', canvasWidth / 2, 8);

  if (narrow) {
    drawPlot(margin, 40, w, drawHeight - 60 - 2 * panelH, mu, sigma, narrow);
    drawCurvePanel(margin, drawHeight - 14 - 2 * panelH, w, panelH, mu, sigma, ts, lh);
    drawRuleTable(margin, drawHeight - 8 - panelH, w, panelH, mu, sigma, ts, lh);
  } else {
    drawPlot(margin, 44, w, drawHeight - 60 - panelH, mu, sigma, narrow);
    drawCurvePanel(margin, drawHeight - 8 - panelH, w * 0.6 - 5, panelH, mu, sigma, ts, lh);
    drawRuleTable(margin + w * 0.6 + 5, drawHeight - 8 - panelH, w * 0.4 - 5, panelH, mu, sigma, ts, lh);
  }
}

// One band of the 68-95-99.7 shading: the area under the curve within k standard deviations
function shadeBand(gx, gy, mu, sigma, k) {
  const a = max(0, mu - k * sigma), b = min(X_MAX, mu + k * sigma);
  if (a >= b) return;
  beginShape();
  vertex(gx(a), gy(0));
  for (let i = 0; i <= 60; i++) { const x = a + (b - a) * i / 60; vertex(gx(x), gy(pdf(x, mu, sigma))); }
  vertex(gx(b), gy(0));
  endShape(CLOSE);
}

function drawCurve(gx, gy, mu, sigma) {
  noFill();
  beginShape();
  for (let x = 0; x <= X_MAX; x += 0.5) vertex(gx(x), gy(pdf(x, mu, sigma)));
  endShape();
}

function drawPlot(x0, y0, w, h, mu, sigma, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const px = x0 + (narrow ? 52 : 58), pw = x0 + w - 16 - px, py = y0 + 26, ph = h - 26 - 40;
  const gx = v => px + v / X_MAX * pw, gy = d => py + ph - d / Y_MAX * ph;
  const preset = PRESETS.find(q => q.name === presetSelect.value());

  // grid, tick labels, and axis titles
  textSize(12);
  for (let d = 0; d <= 0.0501; d += 0.01) {
    stroke('gainsboro');
    strokeWeight(1);
    line(px, gy(d), px + pw, gy(d));
    noStroke();
    fill('dimgray');
    textAlign(RIGHT, CENTER);
    text(nf(d, 1, 2), px - 5, gy(d));
  }
  for (let v = 0; v <= X_MAX; v += 20) {
    stroke('gainsboro');
    line(gx(v), py, gx(v), py + ph);
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    text(v, gx(v), py + ph + 5);
  }
  fill('black');
  textAlign(CENTER, TOP);
  text(preset.axis, px + pw / 2, py + ph + 21);
  push();
  translate(x0 + 11, py + ph / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, CENTER);
  text('density f(x)', 0, 0);
  pop();

  // shaded bands: three translucent layers, so the band within 1 sigma is the darkest
  if (regionsBox.checked()) {
    noStroke();
    fill(65, 105, 225, 55);
    for (const k of [3, 2, 1]) shadeBand(gx, gy, mu, sigma, k);
  }
  stroke('gray');
  strokeWeight(1.5);
  line(px, py, px, py + ph);
  line(px, py + ph, px + pw, py + ph);

  // pinned curve for comparison
  if (pinned) {
    stroke('dimgray');
    strokeWeight(2);
    drawingContext.setLineDash([6, 5]);
    drawCurve(gx, gy, pinned.mu, pinned.sigma);
    drawingContext.setLineDash([]);
    noStroke();
    fill('dimgray');
    textAlign(mu > X_MAX / 2 ? LEFT : RIGHT, TOP);
    text('dashed: pinned μ = ' + pinned.mu + ', σ = ' + pinned.sigma, mu > X_MAX / 2 ? px + 8 : px + pw - 6, py + 4);
  }

  // current curve and the line at the mean
  stroke('royalblue');
  strokeWeight(3);
  drawCurve(gx, gy, mu, sigma);
  stroke('crimson');
  strokeWeight(2);
  line(gx(mu), py + ph, gx(mu), gy(pdf(mu, mu, sigma)));

  // labels above the plot at mu and, when there is room, at each whole number of sigmas
  noStroke();
  textStyle(BOLD);
  textAlign(CENTER, BOTTOM);
  const spacing = sigma / X_MAX * pw;
  for (let k = -3; k <= 3; k++) {
    const v = mu + k * sigma;
    if (v < 0 || v > X_MAX || (k !== 0 && (!regionsBox.checked() || spacing < 23))) continue;
    fill(k === 0 ? 'crimson' : 'royalblue');
    text(k === 0 ? 'μ' : (k > 0 ? '+' : '−') + Math.abs(k) + 'σ', gx(v), py - 4);
  }
  textStyle(NORMAL);

  // pointer readout: a marker on the curve at the x value under the pointer
  hoverX = null;
  if (mouseX >= px && mouseX <= px + pw && mouseY >= py && mouseY <= py + ph) {
    hoverX = Math.round((mouseX - px) / pw * X_MAX);
    stroke('darkorange');
    strokeWeight(1.5);
    line(gx(hoverX), py + ph, gx(hoverX), gy(pdf(hoverX, mu, sigma)));
    fill('darkorange');
    stroke('white');
    circle(gx(hoverX), gy(pdf(hoverX, mu, sigma)), 10);
  }
}

// Text panel: the current parameters, the equation with the values put in, and a comparison
function drawCurvePanel(x, y, w, h, mu, sigma, ts, lh) {
  fill('lightyellow');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  let note = 'The total area is always 1, so a narrower curve is taller.';
  if (pinned) {
    const dMu = mu - pinned.mu, dSigma = sigma - pinned.sigma;
    note = 'Versus pinned: center ' + (dMu === 0 ? 'the same' : Math.abs(dMu) + ' to the ' + (dMu > 0 ? 'right' : 'left')) +
      ', ' + (dSigma === 0 ? 'same spread' : dSigma > 0 ? 'wider and flatter' : 'narrower and taller') + '.';
  }
  const z = hoverX === null ? 0 : (hoverX - mu) / sigma;
  const lines = [
    'Current curve: μ = ' + mu + ', σ = ' + sigma + ', peak f(μ) = ' + nf(pdf(mu, mu, sigma), 1, 4),
    'f(x) = exp(−(x − ' + mu + ')² / (2 × ' + sigma + '²)) / (' + sigma + ' × √(2π))',
    hoverX === null ? 'Point at the plot to read z and f(x) at any x.'
      : 'At x = ' + hoverX + ': z = ' + (z < 0 ? '−' : '+') + nf(Math.abs(z), 1, 2) + ' and f(x) = ' + nf(pdf(hoverX, mu, sigma), 1, 4),
    note];
  noStroke();
  textAlign(LEFT, TOP);
  textSize(ts);
  lines.forEach((s, i) => {
    textStyle(i === 0 ? BOLD : NORMAL);
    fill(i === 0 ? 'royalblue' : i === 2 && hoverX !== null ? 'chocolate' : 'black');
    text(s, x + 12, y + 10 + i * lh);
  });
  textStyle(NORMAL);
}

// The 68-95-99.7 table: each interval and the area under the curve inside it
function drawRuleTable(x, y, w, h, mu, sigma, ts, lh) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const cols = [x + 34, x + w * 0.56, x + w - 12];
  noStroke();
  textSize(ts);
  textStyle(BOLD);
  fill('black');
  textAlign(LEFT, TOP);
  text('Region', x + 12, y + 10);
  textAlign(CENTER, TOP);
  text('Interval', cols[1], y + 10);
  textAlign(RIGHT, TOP);
  text('Area', cols[2], y + 10);
  textStyle(NORMAL);
  for (let k = 1; k <= 3; k++) {
    const ry = y + 10 + k * lh, a = mu - k * sigma, b = mu + k * sigma;
    // swatch with the same number of translucent layers as the band on the plot
    for (let layer = 0; layer <= 3 - k; layer++) { fill(65, 105, 225, 55); rect(x + 12, ry, 15, ts); }
    fill('black');
    textAlign(LEFT, TOP);
    text('μ ± ' + k + 'σ', cols[0], ry);
    textAlign(CENTER, TOP);
    text(a + ' to ' + b, cols[1], ry);
    textAlign(RIGHT, TOP);
    text(nf(100 * areaBetween(a, b, mu, sigma), 1, 2) + '%', cols[2], ry);
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
