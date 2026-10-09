// Metrics Comparison MicroSim
// CANVAS_HEIGHT: 580
// Bloom L3-L4 (Apply, Analyze): students drag data points or add outliers, then compare how MAE,
// MSE, RMSE, and R-squared respond to the same set of prediction errors.
//
// Model: 10 points from y = 2x + 1 plus normal noise (SD 2.4) from a seeded generator. The line is
// the least-squares fit to the points on screen and is refit after every change:
//   slope = Sxy / Sxx,  intercept = mean(y) - slope * mean(x),  residual e = y - prediction.
// Metrics (the scikit-learn definitions, computed from the residuals that are drawn):
//   MAE = mean |e|,  MSE = mean e^2,  RMSE = sqrt(MSE),  R^2 = 1 - sum e^2 / sum (y - mean y)^2,
//   adjusted R^2 = 1 - (1 - R^2)(n - 1) / (n - p - 1) with p = 1 predictor.
// Each square is drawn with its side equal to the residual's length on screen, so its area is
// proportional to e^2: MAE averages the segment lengths and MSE averages the square areas.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 500;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const SEED = 64, NOISE_SD = 2.4, X_MAX = 11, Y_MAX = 30;
const SMALL = 2, LARGE = 5, MAX_OUTLIERS = 3;      // residual size classes, outlier limit

let rngState = 1;
let pts = [];                 // { x, y, pred, e }
let fit = {};                 // fitted line and metrics for the points on screen
let base = {};                // MAE and RMSE of the default points
let outliers = 0, selected = -1, dragIndex = -1, dragDX = 0, dragDY = 0;
let box = {}, panel = {}, plot = {};       // pixel rectangles, set by layout()
let outlierButton, resetButton, absCheckbox, squareCheckbox;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }

function resetData() {
  rngState = SEED;
  pts = [];
  for (let x = 1; x <= 10; x++) pts.push({ x, y: 2 * x + 1 + NOISE_SD * stdNormal() });
  computeFit();
  base = { mae: fit.mae, rmse: fit.rmse };
  outliers = 0;
  selected = -1;
  outlierButton.removeAttribute('disabled');
}

// A new point 10 to 13 units above or below the current line, on a side where it fits in the plot
function addOutlier() {
  if (outliers >= MAX_OUTLIERS) return;
  const x = 1 + 9 * uniform(), lineY = fit.intercept + fit.slope * x, off = 10 + 3 * uniform();
  const canUp = lineY + off <= Y_MAX - 1, canDown = lineY - off >= 1;
  const up = canUp && (!canDown || uniform() < 0.5);
  pts.push({ x, y: constrain(lineY + (up ? off : -off), 0.3, Y_MAX - 0.3) });
  selected = pts.length - 1;
  outliers++;
  if (outliers >= MAX_OUTLIERS) outlierButton.attribute('disabled', '');
  computeFit();
}

// Least-squares line through the current points, then every metric from its residuals
function computeFit() {
  const n = pts.length;
  const mx = pts.reduce((s, p) => s + p.x, 0) / n, my = pts.reduce((s, p) => s + p.y, 0) / n;
  let sxx = 0, sxy = 0, sst = 0, sae = 0, sse = 0, worst = 0;
  for (const p of pts) {
    sxx += (p.x - mx) ** 2;
    sxy += (p.x - mx) * (p.y - my);
    sst += (p.y - my) ** 2;
  }
  const slope = sxx > 1e-9 ? sxy / sxx : 0, intercept = my - slope * mx;
  pts.forEach((p, i) => {
    p.pred = intercept + slope * p.x;
    p.e = p.y - p.pred;
    sae += Math.abs(p.e);
    sse += p.e * p.e;
    if (Math.abs(p.e) > Math.abs(pts[worst].e)) worst = i;
  });
  const mse = sse / n, r2 = sst > 1e-9 ? 1 - sse / sst : NaN;
  fit = { n, slope, intercept, sae, sse, worst, mae: sae / n, mse, rmse: Math.sqrt(mse), r2,
    adjR2: 1 - (1 - r2) * (n - 1) / (n - 1 - 1) };
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  outlierButton = createButton('Add Outlier');
  outlierButton.parent(mainElement);
  outlierButton.position(10, drawHeight + 10);
  outlierButton.mousePressed(addOutlier);

  resetButton = createButton('Reset to Default');
  resetButton.parent(mainElement);
  resetButton.position(105, drawHeight + 10);
  resetButton.mousePressed(resetData);

  absCheckbox = createCheckbox(' Show absolute residuals', true);
  squareCheckbox = createCheckbox(' Show squared residuals', true);
  [absCheckbox, squareCheckbox].forEach(c => {
    c.parent(mainElement);
    c.style('font-size', '15px');
  });
  placeCheckboxes();

  resetData();

  describe('A scatter plot of ten draggable points with a least-squares line. Each residual is drawn as a colored ' +
    'vertical segment and as a square. A panel reports R squared, adjusted R squared, MSE, RMSE, and MAE, with bars ' +
    'comparing MAE and RMSE to their starting values. Buttons add an outlier or reset the points.', LABEL);
}

function placeCheckboxes() {
  absCheckbox.position(10, drawHeight + 45);
  squareCheckbox.position(10 + Math.min(230, (canvasWidth - 20) / 2), drawHeight + 45);
}

function layout(narrow) {
  const w = canvasWidth - 2 * margin;
  if (narrow) {
    box = { x: margin, y: 36, w, h: 222 };
    panel = { x: margin, y: 264, w, h: drawHeight - 272 };
  } else {
    box = { x: margin, y: 42, w: Math.round(w * 0.57), h: drawHeight - 52 };
    panel = { x: box.x + box.w + 10, y: 42, w: w - box.w - 10, h: drawHeight - 52 };
  }
  plot = { x: box.x + 36, y: box.y + 48, w: box.w - 50, h: box.h - 48 - (narrow ? 38 : 46) };
}
function gx(v) { return plot.x + v / X_MAX * plot.w; }
function gy(v) { return plot.y + plot.h - v / Y_MAX * plot.h; }
function sizeColor(e) {
  const a = Math.abs(e);
  return a < SMALL ? [34, 139, 34] : a <= LARGE ? [255, 140, 0] : [220, 20, 60];
}
function signed(v) { return (v < 0 ? '−' : '+') + nf(Math.abs(v), 1, 2); }

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600;
  layout(narrow);
  if (dragIndex < 0) {                      // the last point under the mouse stays selected
    const h = pointAt(mouseX, mouseY);
    if (h >= 0) selected = h;
    cursor(h >= 0 ? 'grab' : ARROW);
  }

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Metrics Comparison: MAE, MSE, RMSE, and R²', canvasWidth / 2, 8);

  drawPlot(narrow);
  drawMetrics(narrow);

  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(narrow ? 14 : defaultTextSize);
  text('Outliers added: ' + outliers + ' of ' + MAX_OUTLIERS, 235, drawHeight + 22);
}

function drawPlot(narrow) {
  const ts = narrow ? 11 : 13;
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(box.x, box.y, box.w, box.h, 10);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text('Data and least-squares line (drag any point)', box.x + 10, box.y + 7);
  textStyle(NORMAL);
  textSize(ts);

  // legend for the residual size classes
  textAlign(LEFT, CENTER);
  let lx = box.x + 10;
  text('Residual size:', lx, box.y + 34);
  lx += textWidth('Residual size:') + 8;
  for (const [label, e] of [['under ' + SMALL, 0], [SMALL + ' to ' + LARGE, SMALL], ['over ' + LARGE, LARGE + 1]]) {
    fill(...sizeColor(e));
    rect(lx, box.y + 28, 12, 12);
    fill('black');
    text(label, lx + 16, box.y + 34);
    lx += textWidth(label) + 28;
  }

  // grid, ticks, and axes
  for (let v = 0; v <= Y_MAX; v += narrow ? 10 : 5) {
    stroke('gainsboro');
    strokeWeight(1);
    line(plot.x, gy(v), plot.x + plot.w, gy(v));
    noStroke();
    fill('dimgray');
    textAlign(RIGHT, CENTER);
    text(v, plot.x - 4, gy(v));
  }
  textAlign(CENTER, TOP);
  for (let v = 0; v <= 10; v += 2) text(v, gx(v), plot.y + plot.h + 3);
  fill('black');
  textAlign(LEFT, CENTER);
  text('y', box.x + 8, plot.y + plot.h / 2);
  textAlign(RIGHT, TOP);
  text('x', plot.x + plot.w, plot.y + plot.h + 3);
  stroke('gray');
  strokeWeight(1.5);
  line(plot.x, plot.y, plot.x, plot.y + plot.h);
  line(plot.x, plot.y + plot.h, plot.x + plot.w, plot.y + plot.h);

  push();
  drawingContext.beginPath();
  drawingContext.rect(plot.x, plot.y, plot.w, plot.h);
  drawingContext.clip();
  // squared residuals: a square on each residual segment, toward the side with more room
  if (squareCheckbox.checked()) {
    for (const p of pts) {
      const side = Math.abs(gy(p.y) - gy(p.pred)), c = sizeColor(p.e);
      const left = gx(p.x) + side > plot.x + plot.w ? gx(p.x) - side : gx(p.x);
      fill(c[0], c[1], c[2], 45);
      stroke(c[0], c[1], c[2], 160);
      strokeWeight(1);
      rect(left, Math.min(gy(p.y), gy(p.pred)), side, side);
    }
  }
  stroke('black');
  strokeWeight(2.5);
  line(gx(0), gy(fit.intercept), gx(X_MAX), gy(fit.intercept + fit.slope * X_MAX));
  // absolute residuals: the vertical distance from each point to the line
  if (absCheckbox.checked()) {
    strokeWeight(4);
    for (const p of pts) {
      stroke(...sizeColor(p.e));
      line(gx(p.x), gy(p.y), gx(p.x), gy(p.pred));
    }
  }
  pts.forEach((p, i) => {
    stroke(i === selected ? 'black' : 'white');
    strokeWeight(i === selected ? 2.5 : 1);
    fill('royalblue');
    circle(gx(p.x), gy(p.y), narrow ? 11 : 13);
  });
  pop();

  // readout for the selected point
  const p = pts[selected];
  noStroke();
  fill(p ? 'black' : 'dimgray');
  textAlign(LEFT, BOTTOM);
  text(p ? 'Selected: (' + nf(p.x, 1, 1) + ', ' + nf(p.y, 1, 2) + ')   predicted ' + nf(p.pred, 1, 2) + '   e = ' + signed(p.e) +
    '   e² = ' + nf(p.e * p.e, 1, 2) : 'Hover over a point to read its residual e = actual − predicted.', box.x + 10, box.y + box.h - 6);
}

function drawMetrics(narrow) {
  const ts = narrow ? 12 : 15, lh = ts + (narrow ? 5 : 8), x = panel.x + 10, w = panel.w - 20;
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(panel.x, panel.y, panel.w, panel.h, 10);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text('Error metrics for ' + fit.n + ' points', x, panel.y + 7);
  textStyle(NORMAL);
  textSize(ts);

  let y = panel.y + 9 + lh;
  const rows = [
    'Line: ŷ = ' + nf(fit.slope, 1, 2) + 'x ' + (fit.intercept < 0 ? '− ' : '+ ') + nf(Math.abs(fit.intercept), 1, 2),
    Number.isFinite(fit.r2) ? 'R² = ' + nf(fit.r2, 1, 3) + '     adjusted R² = ' + nf(fit.adjR2, 1, 3) : 'R² is undefined: every y is the same',
    'MSE = ' + nf(fit.mse, 1, 2) + ' (squared y units)'
  ];
  if (!narrow) {                // room to show each calculation: total of the errors divided by n
    rows[2] = 'MAE = ' + nf(fit.sae, 1, 2) + ' ÷ ' + fit.n + ' = ' + nf(fit.mae, 1, 2);
    rows.push('MSE = ' + nf(fit.sse, 1, 2) + ' ÷ ' + fit.n + ' = ' + nf(fit.mse, 1, 2) + ' (squared units)',
      'RMSE = √' + nf(fit.mse, 1, 2) + ' = ' + nf(fit.rmse, 1, 2));
  }
  for (const row of rows) {
    text(row, x, y);
    y += lh;
  }

  // MAE and RMSE share the units of y, so they can be compared on one scale
  y += narrow ? 0 : 6;
  textStyle(BOLD);
  text('Typical error, in y units', x, y);
  textStyle(NORMAL);
  y += lh + 2;
  const scaleMax = Math.max(6, 2 * Math.ceil(fit.rmse * 1.1 / 2));
  drawBar('MAE', fit.mae, base.mae, 'teal', x, y, w, scaleMax, narrow);
  drawBar('RMSE', fit.rmse, base.rmse, 'darkorchid', x, y + (narrow ? 21 : 26), w, scaleMax, narrow);
  y += narrow ? 44 : 56;
  fill('black');
  text('RMSE ÷ MAE = ' + nf(fit.rmse / fit.mae, 1, 2), x, y);
  fill('dimgray');
  textAlign(RIGHT, TOP);
  text('black mark = at reset', x + w, y);
  textAlign(LEFT, TOP);
  y += lh + (narrow ? 0 : 6);

  // how much of each total comes from the single largest residual
  const big = Math.abs(pts[fit.worst].e);
  fill('black');
  textWrap(WORD);
  const share = 'The largest residual is ' + nf(big, 1, 2) + ' units from the line. It is ' + Math.round(100 * big / fit.sae) +
    '% of the absolute-error total (averaged by MAE) but ' + Math.round(100 * big * big / fit.sse) +
    '% of the squared-error total (averaged by MSE and RMSE).';
  text(share, x, y, w, panel.y + panel.h - y - 4);
  if (narrow) return;

  const growMae = fit.mae / base.mae, growRmse = fit.rmse / base.rmse;
  const note = Math.abs(growMae - 1) < 0.005 && Math.abs(growRmse - 1) < 0.005 ?
    'Drag one point far from the line, or press Add Outlier, and watch which bar grows faster.' :
    growRmse > growMae * 1.05 ? 'RMSE has grown faster than MAE. Squaring gives the largest misses the most weight.' :
      growRmse < growMae * 0.95 ? 'MAE has changed more than RMSE: the errors are now closer to each other in size.' :
        'MAE and RMSE have changed by about the same factor, so no single error dominates more than before.';
  fill('darkslateblue');
  text(note, x, panel.y + panel.h - 78, w, 74);
}

// One horizontal bar with a mark at its value for the default points and its growth factor
function drawBar(label, value, baseValue, col, x, y, w, scaleMax, narrow) {
  const labelW = narrow ? 42 : 50, textW = narrow ? 86 : 104, barW = w - labelW - textW, bx = x + labelW, bh = narrow ? 14 : 16;
  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  text(label, x, y + bh / 2);
  fill('gainsboro');
  rect(bx, y, barW, bh);
  fill(col);
  rect(bx, y, barW * Math.min(value / scaleMax, 1), bh);
  stroke('black');
  strokeWeight(2);
  const mark = bx + barW * baseValue / scaleMax;
  line(mark, y - 3, mark, y + bh + 3);
  noStroke();
  fill('black');
  text(nf(value, 1, 2), bx + barW + 6, y + bh / 2);
  fill(col);
  textAlign(RIGHT, CENTER);
  text('×' + nf(value / baseValue, 1, 2), x + w, y + bh / 2);
  textAlign(LEFT, TOP);
}

function pointAt(mx, my) {
  let best = -1, bestD = 14;
  pts.forEach((p, i) => {
    const d = dist(mx, my, gx(p.x), gy(p.y));
    if (d < bestD) { bestD = d; best = i; }
  });
  return best;
}

function mousePressed() {
  dragIndex = pointAt(mouseX, mouseY);
  if (dragIndex < 0) return;
  selected = dragIndex;
  dragDX = gx(pts[dragIndex].x) - mouseX;
  dragDY = gy(pts[dragIndex].y) - mouseY;
}

function mouseDragged() {
  if (dragIndex < 0) return;
  const p = pts[dragIndex];
  p.x = constrain((mouseX + dragDX - plot.x) / plot.w * X_MAX, 0.2, X_MAX - 0.2);
  p.y = constrain((plot.y + plot.h - mouseY - dragDY) / plot.h * Y_MAX, 0.3, Y_MAX - 0.3);
  computeFit();
  return false;
}

function mouseReleased() { dragIndex = -1; }

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  placeCheckboxes();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
