// Least Squares: Fitting a Line by Shrinking the Squares
// CANVAS_HEIGHT: 655
// Bloom L2 (Understand): students move a line with slope and intercept sliders and watch each
// residual, its square, and the sum of squared errors change, to see why the least-squares line
// is the one with the smallest total square area.
//
// Model: for the line yhat = b0 + b1 x,
//   residual   e_i = y_i - yhat_i
//   SSE        = sum of e_i^2            (the total area of the squares)
// Ordinary least squares gives the line with the smallest SSE:
//   b1 = sum((x - mean x)(y - mean y)) / sum((x - mean x)^2),   b0 = mean y - b1 * mean x
// The ten (hours, score) pairs are the same as in the Regression Line Anatomy sim. Their
// least-squares line is yhat = 47.5 + 5.5x, which both sliders can reach exactly.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 540;
let controlHeight = 115;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;
let labelWidth = 105;       // width of the label span in each slider row
let valueWidth = 50;        // width of the value span in each slider row

const HOURS = [1, 2, 2, 3, 4, 5, 6, 6, 7, 8];
const SCORES = [51, 62, 56, 72, 67, 69, 82, 76, 88, 94];
const X_MAX = 10, Y_MIN = 40, Y_MAX = 110;
const START = { b1: 2, b0: 60 };            // a deliberately poor first line
const SSE_SCALE = 1200;                     // right end of the SSE meter

let slopeRow, interceptRow, bestButton, resetButton, squaresBox;
let best = { b0: 0, b1: 0, sse: 0 };        // least-squares solution, computed in setup
let revealed = false;                       // has the student asked for the best fit?
let lowest = Infinity;                      // smallest SSE reached so far

const meanOf = a => a.reduce((s, v) => s + v, 0) / a.length;
const signed = v => (v < 0 ? '−' : '+') + nf(Math.abs(v), 1, 1);

function residuals(b0, b1) { return HOURS.map((x, i) => SCORES[i] - (b0 + b1 * x)); }
function sumSquares(b0, b1) { return residuals(b0, b1).reduce((s, e) => s + e * e, 0); }

// Ordinary least squares for one predictor
function fitLine(xs, ys) {
  const mx = meanOf(xs), my = meanOf(ys);
  let sxy = 0, sxx = 0;
  for (let i = 0; i < xs.length; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) * (xs[i] - mx);
  }
  const b1 = sxy / sxx, b0 = my - b1 * mx;
  return { b0, b1, sse: sumSquares(b0, b1) };
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);
  best = fitLine(HOURS, SCORES);

  slopeRow = makeSliderRow('Slope β₁', 0, 10, START.b1, 0.1, 0);
  interceptRow = makeSliderRow('Intercept β₀', 20, 80, START.b0, 0.5, 1);
  resizeSliders();

  bestButton = createButton('Show Best Fit');
  bestButton.parent(mainElement);
  bestButton.position(10, drawHeight + 80);
  bestButton.mousePressed(() => {
    slopeRow.slider.value(best.b1);
    interceptRow.slider.value(best.b0);
    revealed = true;
  });
  resetButton = createButton('Reset');
  resetButton.parent(mainElement);
  resetButton.position(125, drawHeight + 80);
  resetButton.mousePressed(() => {
    slopeRow.slider.value(START.b1);
    interceptRow.slider.value(START.b0);
    revealed = false;
    lowest = Infinity;
  });
  squaresBox = createCheckbox(' Show squares', true);
  squaresBox.parent(mainElement);
  squaresBox.position(195, drawHeight + 80);
  squaresBox.style('font-size', '16px');

  describe('A scatter plot of exam score against hours studied for ten students with a line set by slope and intercept ' +
    'sliders. A vertical segment joins each point to the line and a square is drawn on each segment. A table lists ' +
    'every residual and its square, and a meter shows their sum, the sum of squared errors. A button moves the line ' +
    'to the least-squares solution, where the total area of the squares is smallest.', LABEL);
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
  slopeRow.slider.size(w);
  interceptRow.slider.size(w);
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const b1 = slopeRow.slider.value(), b0 = interceptRow.slider.value();
  slopeRow.valueSpan.html(nf(b1, 1, 1));
  interceptRow.valueSpan.html(nf(b0, 1, 1));
  const res = residuals(b0, b1), sse = sumSquares(b0, b1);
  lowest = Math.min(lowest, sse);
  const narrow = canvasWidth < 600, w = canvasWidth - 2 * margin;

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Least Squares: Shrink the Squares', canvasWidth / 2, 8);

  if (narrow) {
    drawPlot(margin, 38, w, 228, b0, b1, res, true);
    drawMeter(margin, 270, w, 62, sse, true);
    drawTable(margin, 336, w, drawHeight - 344, b0, b1, res, sse, true);
  } else {
    const plotW = Math.round(w * 0.6);
    drawPlot(margin, 44, plotW, 404, b0, b1, res, false);
    drawTable(margin + plotW + 10, 44, w - plotW - 10, 404, b0, b1, res, sse, false);
    drawMeter(margin, 456, w, 76, sse, false);
  }
}

function drawPlot(x0, y0, w, h, b0, b1, res, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const px = x0 + (narrow ? 42 : 54), py = y0 + 26, pw = w - (narrow ? 54 : 70), ph = h - (narrow ? 60 : 70);
  const gx = v => px + v / X_MAX * pw, gy = v => py + ph - (v - Y_MIN) / (Y_MAX - Y_MIN) * ph;
  const ts = narrow ? 11 : 13;

  noStroke();
  fill('dimgray');
  textSize(ts);
  textAlign(CENTER, TOP);
  text('Each square\'s area is one point\'s squared error (y − ŷ)².', x0 + w / 2, y0 + 7);

  // grid, ticks, axes
  textSize(ts - 1);
  for (let v = 0; v <= X_MAX; v += narrow ? 2 : 1) {
    stroke('gainsboro');
    strokeWeight(1);
    line(gx(v), py, gx(v), py + ph);
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    text(v, gx(v), py + ph + 4);
  }
  for (let v = Y_MIN; v <= Y_MAX; v += 10) {
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
  text('Hours studied (x)', px + pw / 2, py + ph + (narrow ? 18 : 22));
  push();
  translate(x0 + (narrow ? 10 : 14), py + ph / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, CENTER);
  text('Exam score (y)', 0, 0);
  pop();

  // everything that depends on the line is clipped to the plot area
  drawingContext.save();
  drawingContext.beginPath();
  drawingContext.rect(px, py, pw, ph);
  drawingContext.clip();
  HOURS.forEach((x, i) => {
    const X = gx(x), ya = gy(SCORES[i]), yp = gy(b0 + b1 * x), side = Math.abs(ya - yp);
    if (squaresBox.checked()) {
      // a point above the line gets its square on the left, a point below on the right,
      // so that the squares stay clear of an uphill line
      fill(220, 20, 60, 45);
      stroke(220, 20, 60, 150);
      strokeWeight(1);
      rect(res[i] > 0 ? X - side : X, Math.min(ya, yp), side, side);
    }
    stroke('crimson');
    strokeWeight(2.5);
    line(X, ya, X, yp);
  });
  if (revealed) {
    stroke('green');
    strokeWeight(2);
    drawingContext.setLineDash([7, 5]);
    line(gx(0), gy(best.b0), gx(X_MAX), gy(best.b0 + best.b1 * X_MAX));
    drawingContext.setLineDash([]);
  }
  stroke('royalblue');
  strokeWeight(3);
  line(gx(0), gy(b0), gx(X_MAX), gy(b0 + b1 * X_MAX));
  drawingContext.restore();

  stroke('white');
  strokeWeight(1);
  fill('black');
  HOURS.forEach((x, i) => circle(gx(x), gy(SCORES[i]), narrow ? 8 : 10));

  // legend in the lower right corner
  const lx = px + pw - (narrow ? 132 : 158), ly = py + ph - (revealed ? 36 : 18);
  stroke('royalblue');
  strokeWeight(3);
  line(lx, ly, lx + 26, ly);
  noStroke();
  fill('black');
  textSize(ts);
  textAlign(LEFT, CENTER);
  text('your line', lx + 32, ly);
  if (revealed) {
    stroke('green');
    strokeWeight(2);
    drawingContext.setLineDash([7, 5]);
    line(lx, ly + 18, lx + 26, ly + 18);
    drawingContext.setLineDash([]);
    noStroke();
    fill('black');
    text('least-squares line', lx + 32, ly + 18);
  }
}

// One row per data point: the residual and its square. The last column adds up to SSE.
function drawTable(x0, y0, w, h, b0, b1, res, sse, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const ts = narrow ? 11 : 14, headY = y0 + (narrow ? 26 : 40), sumH = narrow ? 18 : 30;
  const rowH = (y0 + h - headY - sumH - 6) / (HOURS.length + 1);
  const colX = k => x0 + 6 + (k + 1) * (w - 18) / 5;           // right edge of column k

  noStroke();
  fill('royalblue');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 2);
  text('Your line:  ŷ = ' + nf(b0, 1, 1) + ' + ' + nf(b1, 1, 1) + 'x', x0 + 10, y0 + (narrow ? 7 : 10));
  textSize(ts);
  textAlign(RIGHT, CENTER);
  fill('dimgray');
  ['x', 'y', 'ŷ', 'y − ŷ', '(y − ŷ)²'].forEach((s, k) => text(s, colX(k), headY + rowH / 2));
  textStyle(NORMAL);
  stroke('silver');
  strokeWeight(1);
  line(x0 + 8, headY + rowH, x0 + w - 8, headY + rowH);

  HOURS.forEach((x, i) => {
    const cy = headY + (i + 1.5) * rowH + 1, yh = b0 + b1 * x;
    noStroke();
    fill('black');
    text(x, colX(0), cy);
    text(SCORES[i], colX(1), cy);
    fill('royalblue');
    text(nf(yh, 1, 1), colX(2), cy);
    fill('black');
    text(signed(res[i]), colX(3), cy);
    fill('crimson');
    text(nf(res[i] * res[i], 1, 2), colX(4), cy);
  });

  const sy = y0 + h - sumH - 4;
  stroke('gray');
  strokeWeight(1.5);
  line(x0 + 8, sy, x0 + w - 8, sy);
  noStroke();
  textStyle(BOLD);
  textSize(ts + 1);
  fill('black');
  textAlign(LEFT, CENTER);
  text('SSE = sum of last column', x0 + 10, sy + sumH / 2 + 1);
  fill('crimson');
  textAlign(RIGHT, CENTER);
  text(nf(sse, 1, 2), colX(4), sy + sumH / 2 + 1);
  textStyle(NORMAL);
}

// SSE as a number and as a bar that turns from red to green as the line improves
function drawMeter(x0, y0, w, h, sse, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const ts = narrow ? 11 : 13, ratio = sse / best.sse, t = constrain((ratio - 1) / 4, 0, 1);
  const col = t < 0.5 ? lerpColor(color('green'), color('darkorange'), t * 2) : lerpColor(color('darkorange'), color('red'), t * 2 - 1);

  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textSize(ts);
  text('Sum of squared errors', x0 + 10, y0 + (narrow ? 5 : 8));
  fill(col);
  textStyle(BOLD);
  textSize(narrow ? 18 : 26);
  text('SSE = ' + nf(sse, 1, 2), x0 + 10, y0 + (narrow ? 20 : 27));
  textStyle(NORMAL);

  const bx = x0 + (narrow ? 152 : 236), bw = x0 + w - 20 - bx, by = y0 + (narrow ? 16 : 20), bh = narrow ? 13 : 16;
  const sx = v => bx + Math.min(v, SSE_SCALE) / SSE_SCALE * bw;
  fill('whitesmoke');
  stroke('silver');
  strokeWeight(1);
  rect(bx, by, bw, bh);
  noStroke();
  fill(col);
  rect(bx, by, sx(sse) - bx, bh);
  textSize(11);
  for (let v = 0; v <= SSE_SCALE; v += narrow ? 400 : 200) {
    stroke('gray');
    strokeWeight(1);
    line(sx(v), by + bh, sx(v), by + bh + 3);
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    text(v, sx(v), by + bh + 4);
  }
  // marker: the minimum once it has been revealed, otherwise the best SSE reached so far
  const mark = revealed ? best.sse : lowest, mx = sx(mark), markCol = revealed ? 'green' : 'black';
  stroke(markCol);
  strokeWeight(2);
  line(mx, by - 3, mx, by + bh + 3);
  noStroke();
  fill(markCol);
  const flip = mx > bx + bw - 100;                             // keep the label inside the panel
  textAlign(flip ? RIGHT : LEFT, BOTTOM);
  text(revealed ? 'minimum ' + nf(best.sse, 1, 2) : 'lowest so far', mx + (flip ? -4 : 4), by - 1);

  let msg;
  if (ratio <= 1.0005) msg = 'This is the least-squares line. No other line has a smaller SSE.';
  else if (ratio <= 1.05) msg = 'Very close: within 5% of the smallest possible SSE.';
  else if (revealed) msg = 'Your SSE is ' + nf(100 * (ratio - 1), 1, 0) + '% above the minimum of ' + nf(best.sse, 1, 2) + '.';
  else msg = 'Lowest SSE so far: ' + nf(lowest, 1, 2) + '. Shrink the squares with the sliders.';
  fill(ratio <= 1.05 ? 'green' : 'black');
  textSize(ts);
  textAlign(LEFT, BOTTOM);
  text(msg, narrow ? x0 + 10 : bx, y0 + h - (narrow ? 3 : 5));
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
