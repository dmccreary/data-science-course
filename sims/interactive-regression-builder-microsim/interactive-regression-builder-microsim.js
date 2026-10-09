// Interactive Regression Builder
// CANVAS_HEIGHT: 620
// Bloom L6 (Create): students build their own data set by adding, dragging, and removing points,
// and the least-squares model is refitted after every change. They read the equation, the plain
// language meaning of the slope and intercept, R squared, RMSE, and a prediction.
//
// Model (ordinary least squares, what scikit-learn's LinearRegression computes):
//   b1 = sum((x - mean x)(y - mean y)) / sum((x - mean x)^2),   b0 = mean y - b1 * mean x
//   residual e_i = y_i - (b0 + b1 x_i),   SSE = sum e_i^2,   SST = sum (y_i - mean y)^2
//   R squared = 1 - SSE / SST   (model.score, r2_score)
//   RMSE = sqrt(SSE / n)        (np.sqrt(mean_squared_error(y, y_pred)))
// The first data set is the chapter's eight study-hours points. The others come from a seeded
// generator (mulberry32), so they are the same on every load.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 540;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const MAX_POINTS = 60;
let rngState = 1;
function uniform() {        // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function gauss() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }
function sample(n, seed, make) { rngState = seed; return Array.from({ length: n }, make); }
const dollars = v => (v < 0 ? '−$' : '$') + Math.round(Math.abs(v)).toLocaleString('en-US');

// Each data set: axis ranges are [min, max, tick step]; y in the two money sets is in $1000s
const SETS = [
  { name: 'Study hours vs score', xLabel: 'Hours studied', yLabel: 'Exam score', xr: [0, 10, 2], yr: [40, 100, 10],
    xUnit: 'hour of study', yWhat: 'exam score', zero: '0 hours of study', predictAt: 4.5, yFmt: v => nf(v, 1, 1) + ' points',
    make: () => [52, 58, 65, 71, 75, 82, 87, 92].map((y, i) => ({ x: i + 1, y })) },
  { name: 'House size vs price', xLabel: 'House size (sq ft)', yLabel: 'Price ($1000s)', xr: [500, 3000, 500], yr: [0, 800, 200],
    xUnit: 'square foot', yWhat: 'price', zero: 'a house of 0 square feet', predictAt: 1500, yFmt: v => dollars(1000 * v),
    make: () => sample(20, 101, () => { const x = 800 + 1700 * uniform(); return { x, y: 50 + 0.2 * x + 70 * gauss() }; }) },
  { name: 'Car age vs value', xLabel: 'Car age (years)', yLabel: 'Value ($1000s)', xr: [0, 16, 2], yr: [0, 35, 5],
    xUnit: 'year of age', yWhat: 'value', zero: 'a new car (age 0)', predictAt: 5, yFmt: v => dollars(1000 * v),
    make: () => sample(18, 202, () => { const x = 0.5 + 13.5 * uniform(); return { x, y: Math.max(1, 28 - 1.7 * x + 2.5 * gauss()) }; }) },
  { name: 'Random (no relationship)', xLabel: 'x', yLabel: 'y', xr: [0, 100, 20], yr: [0, 100, 20],
    xUnit: 'unit of x', yWhat: 'y', zero: 'x = 0', predictAt: 50, yFmt: v => nf(v, 1, 1),
    make: () => sample(20, 311, () => ({ x: 5 + 90 * uniform(), y: 10 + 80 * uniform() })) }
];

let setSelect, clearButton, residBox, predictInput;
let cur = SETS[0];
let pts = [];               // the data: { x, y } in data units
let plot = { px: 0, py: 0, pw: 1, ph: 1 };   // scatter plot rectangle of the last frame
let dragIdx = -1;

function loadSet() {
  cur = SETS.find(s => s.name === setSelect.value());
  pts = cur.make();
  predictInput.value(cur.predictAt);
}

// Least-squares fit of the current points, or null when no line is defined
function fitModel(p) {
  const n = p.length;
  if (n < 2) return null;
  const mx = p.reduce((s, q) => s + q.x, 0) / n, my = p.reduce((s, q) => s + q.y, 0) / n;
  let sxx = 0, sxy = 0, sst = 0;
  for (const q of p) {
    sxx += (q.x - mx) * (q.x - mx);
    sxy += (q.x - mx) * (q.y - my);
    sst += (q.y - my) * (q.y - my);
  }
  if (sxx < 1e-12) return null;
  const b1 = sxy / sxx, b0 = my - b1 * mx;
  const res = p.map(q => q.y - (b0 + b1 * q.x)), sse = res.reduce((s, e) => s + e * e, 0);
  const xs = p.map(q => q.x);
  return { n, b0, b1, res, r2: sst > 0 ? 1 - sse / sst : NaN, rmse: Math.sqrt(sse / n), xLo: Math.min(...xs), xHi: Math.max(...xs) };
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  const row1 = createDiv();
  row1.parent(mainElement);
  row1.position(10, drawHeight + 8);
  row1.style('font-size', '16px');
  row1.style('white-space', 'nowrap');
  createSpan('Data set: ').parent(row1);
  setSelect = createSelect();
  setSelect.parent(row1);
  SETS.forEach(s => setSelect.option(s.name));
  setSelect.style('font-size', '15px');
  setSelect.changed(loadSet);
  clearButton = createButton('Clear All');
  clearButton.parent(row1);
  clearButton.style('margin-left', '10px');
  clearButton.mousePressed(() => { pts = []; });

  const row2 = createDiv();
  row2.parent(mainElement);
  row2.position(10, drawHeight + 43);
  row2.style('font-size', '16px');
  row2.style('white-space', 'nowrap');
  createSpan('Predict at x = ').parent(row2);
  predictInput = createInput('4.5', 'number');
  predictInput.parent(row2);
  predictInput.size(70);
  predictInput.style('font-size', '15px');
  residBox = createCheckbox(' Show residuals', true);
  residBox.parent(row2);
  residBox.style('display', 'inline-block');
  residBox.style('margin-left', '14px');
  loadSet();

  describe('A scatter plot where points can be added by clicking, moved by dragging, and removed by double-clicking. ' +
    'The least-squares line is refitted after every change. A residual plot sits under the scatter plot, and a panel ' +
    'gives the equation, the meaning of the slope and intercept, R squared, RMSE, and a prediction for a typed x value ' +
    'with a warning when it is outside the data. A menu loads four starting data sets.', LABEL);
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
  const m = fitModel(pts), xp = parseFloat(predictInput.value());
  textWrap(WORD);

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Build Your Own Regression Model', canvasWidth / 2, 8);

  if (narrow) {
    drawScatter(margin, 38, w, 210, m, xp, true);
    drawResiduals(margin, 252, w, 90, m, true);
    drawPanel(margin, 346, w, drawHeight - 354, m, xp, true);
  } else {
    const plotW = Math.round(w * 0.6);
    drawScatter(margin, 44, plotW, 318, m, xp, false);
    drawResiduals(margin, 368, plotW, drawHeight - 376, m, false);
    drawPanel(margin + plotW + 10, 44, w - plotW - 10, drawHeight - 52, m, xp, false);
  }
}

function drawScatter(x0, y0, w, h, m, xp, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const px = x0 + (narrow ? 44 : 58), py = y0 + 26, pw = w - (narrow ? 58 : 74), ph = h - (narrow ? 60 : 70);
  const [xa, xb, xStep] = cur.xr, [ya, yb, yStep] = cur.yr, ts = narrow ? 11 : 13;
  const gx = v => px + (v - xa) / (xb - xa) * pw, gy = v => py + ph - (v - ya) / (yb - ya) * ph;
  plot = { px, py, pw, ph };

  noStroke();
  fill('dimgray');
  textSize(ts);
  textAlign(CENTER, TOP);
  text('Click to add a point. Drag to move it. Double-click to remove it.', x0 + w / 2, y0 + 7);

  textSize(ts - 1);
  for (let v = xa; v <= xb + 1e-9; v += xStep) {
    stroke('gainsboro');
    strokeWeight(1);
    line(gx(v), py, gx(v), py + ph);
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    text(v, gx(v), py + ph + 4);
  }
  for (let v = ya; v <= yb + 1e-9; v += yStep) {
    stroke('gainsboro');
    strokeWeight(1);
    line(px, gy(v), px + pw, gy(v));
    noStroke();
    fill('dimgray');
    textAlign(RIGHT, CENTER);
    text(v, px - 5, gy(v));
  }
  stroke('gray');
  strokeWeight(1.5);
  line(px, py, px, py + ph);
  line(px, py + ph, px + pw, py + ph);
  noStroke();
  fill('black');
  textSize(ts);
  textAlign(CENTER, TOP);
  text(cur.xLabel + '  (x)', px + pw / 2, py + ph + (narrow ? 18 : 22));
  push();
  translate(x0 + (narrow ? 10 : 14), py + ph / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, CENTER);
  text(cur.yLabel + '  (y)', 0, 0);
  pop();

  if (m) {
    // residual segments, the fitted line, and the prediction guide, clipped to the plot area
    drawingContext.save();
    drawingContext.beginPath();
    drawingContext.rect(px, py, pw, ph);
    drawingContext.clip();
    if (residBox.checked()) {
      stroke('crimson');
      strokeWeight(2);
      pts.forEach(q => line(gx(q.x), gy(q.y), gx(q.x), gy(m.b0 + m.b1 * q.x)));
    }
    stroke('royalblue');
    strokeWeight(3);
    line(gx(xa), gy(m.b0 + m.b1 * xa), gx(xb), gy(m.b0 + m.b1 * xb));
    if (isFinite(xp)) {
      const X = gx(xp), Y = gy(m.b0 + m.b1 * xp);
      stroke('green');
      strokeWeight(1.5);
      drawingContext.setLineDash([5, 4]);
      line(X, py + ph, X, Y);
      line(px, Y, X, Y);
      drawingContext.setLineDash([]);
      stroke('white');
      fill('green');
      quad(X, Y - 9, X + 9, Y, X, Y + 9, X - 9, Y);
    }
    drawingContext.restore();
  }

  const over = dragIdx >= 0 ? dragIdx : pointAt(mouseX, mouseY);
  cursor(over >= 0 ? HAND : inPlot(mouseX, mouseY) ? CROSS : ARROW);
  pts.forEach((q, i) => {
    stroke('white');
    strokeWeight(1);
    fill(i === over ? 'gold' : 'black');
    circle(gx(q.x), gy(q.y), i === over ? 14 : (narrow ? 9 : 10));
  });
}

// Residuals against x, drawn on the same horizontal scale as the scatter plot above
function drawResiduals(x0, y0, w, h, m, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const ts = narrow ? 11 : 13, top = y0 + 24, rh = h - 34, mid = top + rh / 2;
  noStroke();
  fill('black');
  textSize(ts);
  textAlign(LEFT, TOP);
  text('Residual plot: y − ŷ at each x. Look for a band with no pattern.', x0 + 10, y0 + 6);
  if (!m) return;
  const maxR = Math.max(...m.res.map(Math.abs), 1e-9), gy = e => mid - e / (maxR * 1.15) * rh / 2;
  const gx = v => plot.px + (v - cur.xr[0]) / (cur.xr[1] - cur.xr[0]) * plot.pw;
  stroke('gray');
  strokeWeight(1);
  line(plot.px, top, plot.px, top + rh);
  drawingContext.setLineDash([5, 4]);
  line(plot.px, mid, plot.px + plot.pw, mid);
  drawingContext.setLineDash([]);
  noStroke();
  fill('dimgray');
  textSize(ts - 1);
  textAlign(RIGHT, CENTER);
  const lab = nf(maxR, 1, maxR < 10 ? 1 : 0);
  text('+' + lab, plot.px - 5, gy(maxR));
  text('0', plot.px - 5, mid);
  text('−' + lab, plot.px - 5, gy(-maxR));
  pts.forEach((q, i) => {
    stroke('crimson');
    strokeWeight(1.5);
    line(gx(q.x), mid, gx(q.x), gy(m.res[i]));
    stroke('white');
    strokeWeight(1);
    fill('crimson');
    circle(gx(q.x), gy(m.res[i]), narrow ? 7 : 9);
  });
}

// Coefficients are shown with two decimals, or three when smaller than 1
const coef = v => nf(Math.abs(v), 1, Math.abs(v) < 1 ? 3 : 2);

// Equation, interpretation, fit quality, and prediction. Each block gets a fixed number of lines.
function drawPanel(x0, y0, w, h, m, xp, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const ts = narrow ? 12 : 14, lh = ts + 4, tx = x0 + 12, tw = w - 24;
  noStroke();
  textAlign(LEFT, TOP);
  if (!m) {
    fill('black');
    textSize(ts + 1);
    text('Add at least two points with different x values to fit a line.\n\nThings to build:\n• data with R² above 0.9\n' +
      '• data with a negative slope\n• a strong pattern, then one outlier far from the line', tx, y0 + 12, tw, h - 20);
    return;
  }
  const pct = nf(100 * m.r2, 1, 1), r2Text = isNaN(m.r2) ? 'R² is undefined: every y value is the same.'
    : 'R² = ' + nf(m.r2, 1, 3) + (narrow ? ' (' + pct + '% of variation explained)' : ': the line explains ' + pct + '% of the variation in ' + cur.yWhat + '.');
  const rmseText = 'RMSE = ' + cur.yFmt(m.rmse) + (narrow ? '' : ', the typical size of a prediction error.');
  // x = 0 counts as far from the data when the gap to it is more than a quarter of the data's x range
  const farFromZero = Math.max(m.xLo, -m.xHi) > 0.25 * (m.xHi - m.xLo);
  let predText = 'Type a number in the Predict box to get a prediction.', outside = false;
  if (isFinite(xp)) {
    outside = xp < m.xLo || xp > m.xHi;
    predText = 'Prediction at x = ' + xp + ': ŷ = ' + cur.yFmt(m.b0 + m.b1 * xp) + '. ' + (outside
      ? 'Extrapolation warning: your data only cover x from ' + nf(m.xLo, 1, 1) + ' to ' + nf(m.xHi, 1, 1) + ', so this is unreliable.'
      : 'This x is inside your data range, so the prediction is an interpolation.');
  }
  // [text, lines, color, bold, text size]
  const blocks = [
    ['ŷ = ' + (m.b0 < 0 ? '−' : '') + coef(m.b0) + (m.b1 < 0 ? ' − ' : ' + ') + coef(m.b1) + 'x', 1, 'royalblue', true, ts + (narrow ? 3 : 6)],
    ['Slope: each extra ' + cur.xUnit + (m.b1 < 0 ? ' lowers' : ' raises') + ' the predicted ' + cur.yWhat + ' by ' +
      cur.yFmt(Math.abs(m.b1)) + '.', narrow ? 2 : 3, 'black'],
    ['Intercept: at ' + cur.zero + ', the predicted ' + cur.yWhat + ' is ' + cur.yFmt(m.b0) + '.' +
      (!farFromZero ? '' : narrow ? ' Caution: x = 0 is far outside your data.'
        : ' But x = 0 is far outside your data, so do not read this literally.'), narrow ? 2 : 4, 'black'],
    [narrow ? r2Text + '   ' + rmseText : r2Text, narrow ? 1 : 2, 'black', true],
    ['gauge', 1],
    ...(narrow ? [] : [[rmseText, 2, 'black']]),
    [predText, narrow ? 3 : 5, outside ? 'firebrick' : 'darkgreen'],
    ...(narrow ? [] : [['Try it: add one point far from the line and watch the slope, R², and RMSE change. Can you build ' +
      'data with R² above 0.9? With a negative slope?', 4, 'dimgray']])
  ];
  let y = y0 + (narrow ? 6 : 10);
  for (const [s, lines, col, bold, size] of blocks) {
    if (s === 'gauge') {
      // R squared as a bar from 0 to 1
      const g = constrain(m.r2 || 0, 0, 1);
      fill('whitesmoke');
      stroke('silver');
      strokeWeight(1);
      rect(tx, y, tw, 10, 3);
      noStroke();
      fill(g >= 0.7 ? 'green' : g >= 0.3 ? 'darkorange' : 'gray');
      rect(tx, y, tw * g, 10, 3);
      y += narrow ? 15 : 20;
      continue;
    }
    noStroke();
    fill(col);
    textStyle(bold ? BOLD : NORMAL);
    textSize(size || ts);
    text(s, tx, y, tw, lines * lh + 4);
    if (size && !narrow) {
      textStyle(NORMAL);
      textSize(ts);
      fill('dimgray');
      textAlign(RIGHT, TOP);
      text('n = ' + m.n, tx + tw, y + 6);
      textAlign(LEFT, TOP);
    }
    y += lines * lh + (size ? 12 : narrow ? 2 : 8);
  }
  textStyle(NORMAL);
}

function inPlot(x, y) { return x >= plot.px && x <= plot.px + plot.pw && y >= plot.py && y <= plot.py + plot.ph; }
const toPixelX = v => plot.px + (v - cur.xr[0]) / (cur.xr[1] - cur.xr[0]) * plot.pw;
const toPixelY = v => plot.py + plot.ph - (v - cur.yr[0]) / (cur.yr[1] - cur.yr[0]) * plot.ph;
function toData(x, y) {
  return { x: cur.xr[0] + constrain((x - plot.px) / plot.pw, 0, 1) * (cur.xr[1] - cur.xr[0]),
    y: cur.yr[0] + constrain((plot.py + plot.ph - y) / plot.ph, 0, 1) * (cur.yr[1] - cur.yr[0]) };
}

// Index of the point nearest the given position, or -1
function pointAt(x, y) {
  let found = -1, bestD = 12;
  pts.forEach((q, i) => {
    const d = dist(x, y, toPixelX(q.x), toPixelY(q.y));
    if (d < bestD) { bestD = d; found = i; }
  });
  return found;
}

// Press on a point to drag it. Press on empty plot space to add a point (which can be dragged at once).
function mousePressed() {
  dragIdx = -1;
  if (!inPlot(mouseX, mouseY)) return;
  dragIdx = pointAt(mouseX, mouseY);
  if (dragIdx < 0 && pts.length < MAX_POINTS) {
    pts.push(toData(mouseX, mouseY));
    dragIdx = pts.length - 1;
  }
}

function mouseDragged() {
  if (dragIdx < 0) return;
  pts[dragIdx] = toData(mouseX, mouseY);
  return false;
}

function mouseReleased() { dragIdx = -1; }

function doubleClicked() {
  const i = inPlot(mouseX, mouseY) ? pointAt(mouseX, mouseY) : -1;
  if (i >= 0) pts.splice(i, 1);
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
