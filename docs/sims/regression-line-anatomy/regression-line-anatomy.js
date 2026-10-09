// Regression Line Anatomy
// CANVAS_HEIGHT: 585
// Bloom L1 (Remember): students identify the five parts of a fitted regression line (intercept,
// slope, predicted value, actual value, residual) by hovering over or clicking numbered callouts,
// then hide the names and test themselves.
//
// Model: ordinary least squares on ten (hours studied, exam score) pairs.
//   slope      b1 = sum((x - mean x)(y - mean y)) / sum((x - mean x)^2)
//   intercept  b0 = mean y - b1 * mean x
//   predicted  yhat = b0 + b1 x          residual = y - yhat
// The ten scores were chosen so that the least-squares line is exactly yhat = 47.5 + 5.5x, the
// model the chapter interprets. Every number on screen is computed from the data below.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 540;
let controlHeight = 45;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const HOURS = [1, 2, 2, 3, 4, 5, 6, 6, 7, 8];
const SCORES = [51, 62, 56, 72, 67, 69, 82, 76, 88, 94];
const X_MAX = 10, Y_MIN = 40, Y_MAX = 100;
const RUN_FROM = 6, RUN_TO = 8;            // the slope triangle spans these x values

const PARTS = [
  { name: 'Intercept', sym: 'β₀', tag: 'Intercept β₀', color: 'royalblue', short: 'where the line crosses the y-axis',
    notes: ['Where the line crosses the y-axis.', 'The predicted y when x = 0.', 'In the equation it is the constant term.'] },
  { name: 'Slope', sym: 'β₁', tag: 'Slope β₁', color: 'crimson', short: 'rise over run',
    notes: ['Rise over run: the change in y for each one-unit increase in x.',
      'A positive slope runs uphill, a negative slope runs downhill.', 'In the equation it is the number that multiplies x.'] },
  { name: 'Predicted value', sym: 'ŷ', tag: 'Predicted ŷ', color: 'green', short: 'the model\'s answer, always on the line',
    notes: ['The value the model predicts for a given x.', 'It always falls exactly on the line.', 'Formula: ŷ = β₀ + β₁x.'] },
  { name: 'Actual value', sym: 'y', tag: 'Actual y', color: 'darkorange', short: 'a real observed data point',
    notes: ['A real observed value: one data point.', 'It is usually not exactly on the line.',
      'Every dot in the scatter plot is an actual value.'] },
  { name: 'Residual', sym: 'y − ŷ', tag: 'Residual', color: 'purple', short: 'actual minus predicted',
    notes: ['The vertical distance from the actual value to the predicted value.', 'Residual = y − ŷ: what the model got wrong.',
      'Positive when the point is above the line, negative when it is below.'] }
];

let residBox, namesBox;
let fit = { b0: 0, b1: 0 };
let feat = 0;               // index of the featured data point (largest residual)
let locked = -1;            // part kept selected by a click, or -1
let sel = -1;               // part shown this frame (hovered, else locked)
let hits = [];              // hover and click rectangles of the last frame: { i, x, y, w, h }

const meanOf = a => a.reduce((s, v) => s + v, 0) / a.length;
function predict(x) { return fit.b0 + fit.b1 * x; }

// Ordinary least squares for one predictor
function fitLine(xs, ys) {
  const mx = meanOf(xs), my = meanOf(ys);
  let sxy = 0, sxx = 0;
  for (let i = 0; i < xs.length; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) * (xs[i] - mx);
  }
  const b1 = sxy / sxx;
  return { b0: my - b1 * mx, b1 };
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  fit = fitLine(HOURS, SCORES);
  HOURS.forEach((x, i) => {
    if (Math.abs(SCORES[i] - predict(x)) > Math.abs(SCORES[feat] - predict(HOURS[feat]))) feat = i;
  });

  residBox = createCheckbox(' Show all residuals', false);
  residBox.parent(mainElement);
  residBox.position(10, drawHeight + 10);
  residBox.style('font-size', '16px');
  namesBox = createCheckbox(' Show names', true);
  namesBox.parent(mainElement);
  namesBox.position(200, drawHeight + 10);
  namesBox.style('font-size', '16px');

  describe('A scatter plot of exam score against hours studied for ten students with its least-squares regression line. ' +
    'Five numbered callouts mark the intercept, the slope triangle, a predicted value, an actual value, and the residual ' +
    'between them. Hovering over or clicking a callout explains that part and shows its value. The regression equation ' +
    'is shown with the same colors. Checkboxes show every residual and hide the names for a self-test.', LABEL);
}

// Color of part i: full strength when nothing or this part is selected, faded otherwise.
// The residual joins the actual and predicted values, so those two stay lit with it.
function tone(i, alpha) {
  const c = color(PARTS[i].color);
  const lit = sel < 0 || sel === i || (sel === 4 && (i === 2 || i === 3));
  c.setAlpha(lit ? (alpha || 255) : (alpha ? 12 : 55));
  return c;
}

function badge(i, x, y, r, solid) {
  stroke('white');
  strokeWeight(1.5);
  fill(solid ? PARTS[i].color : tone(i));
  circle(x, y, 2 * r);
  noStroke();
  fill('white');
  textAlign(CENTER, CENTER);
  textStyle(BOLD);
  textSize(r + 3);
  text(i + 1, x, y + 1);
  textStyle(NORMAL);
}

function partAt(x, y) {
  for (const h of hits) if (x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h) return h.i;
  return -1;
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
  const hov = partAt(mouseX, mouseY);
  sel = hov >= 0 ? hov : locked;
  cursor(hov >= 0 ? HAND : ARROW);
  hits = [];
  textWrap(WORD);

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Regression Line Anatomy', canvasWidth / 2, 8);

  if (narrow) {
    drawPlot(margin, 38, w, 230, true);
    drawEquation(margin, 272, w, 68, true);
    drawPanel(margin, 344, w, drawHeight - 352, true);
  } else {
    const plotW = Math.round(w * 0.6);
    drawPlot(margin, 44, plotW, 396, false);
    drawEquation(margin, 448, plotW, 84, false);
    drawPanel(margin + plotW + 10, 44, w - plotW - 10, drawHeight - 52, false);
  }
}

function drawPlot(x0, y0, w, h, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const px = x0 + (narrow ? 42 : 54), py = y0 + 12, pw = w - (narrow ? 54 : 70), ph = h - (narrow ? 46 : 56);
  const gx = v => px + v / X_MAX * pw, gy = v => py + ph - (v - Y_MIN) / (Y_MAX - Y_MIN) * ph;
  const ts = narrow ? 11 : 13, r = narrow ? 10 : 12;

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

  // regression line, clipped to the plot area
  drawingContext.save();
  drawingContext.beginPath();
  drawingContext.rect(px, py, pw, ph);
  drawingContext.clip();
  stroke('black');
  strokeWeight(2.5);
  line(gx(0), gy(predict(0)), gx(X_MAX), gy(predict(X_MAX)));
  drawingContext.restore();

  // slope triangle: run along x, then rise up to the line
  const ax = gx(RUN_FROM), bx = gx(RUN_TO), ay = gy(predict(RUN_FROM)), cy = gy(predict(RUN_TO));
  noStroke();
  fill(tone(1, 45));
  triangle(ax, ay, bx, ay, bx, cy);
  stroke(tone(1));
  strokeWeight(2.5);
  line(ax, ay, bx, ay);
  line(bx, ay, bx, cy);
  strokeWeight(1);
  noFill();
  rect(bx - 8, ay - 8, 8, 8);
  noStroke();
  fill(tone(1));
  textSize(ts);
  textAlign(CENTER, TOP);
  text('run = ' + (RUN_TO - RUN_FROM), gx(RUN_FROM + 1.25), ay + 4);
  textAlign(LEFT, CENTER);
  text('rise = ' + nf(predict(RUN_TO) - predict(RUN_FROM), 1, 1), bx + 6, (ay + cy) / 2);

  // residuals of every point (optional), then the data points
  const xf = gx(HOURS[feat]), ya = gy(SCORES[feat]), yp = gy(predict(HOURS[feat]));
  HOURS.forEach((x, i) => {
    if (residBox.checked() && i !== feat) {
      stroke(tone(4, 170));
      strokeWeight(2);
      line(gx(x), gy(SCORES[i]), gx(x), gy(predict(x)));
    }
  });
  stroke('white');
  strokeWeight(1);
  fill('steelblue');
  HOURS.forEach((x, i) => { if (i !== feat) circle(gx(x), gy(SCORES[i]), narrow ? 9 : 10); });

  // featured point: residual segment, predicted value on the line, actual value; and the intercept
  stroke(tone(4));
  strokeWeight(4);
  line(xf, ya, xf, yp);
  stroke('white');
  strokeWeight(1.5);
  fill(tone(2));
  circle(xf, yp, 13);
  fill(tone(3));
  circle(xf, ya, 13);
  fill(tone(0));
  circle(gx(0), gy(fit.b0), 13);

  // numbered callouts: [part, anchor x, anchor y, badge x, badge y, side of the name]
  const calls = [
    [0, gx(0), gy(fit.b0), gx(0) + 30, Math.min(gy(fit.b0) + 16, py + ph - r - 2), 1],
    [1, ax + r + 2, ay, ax + r + 2, ay + (narrow ? 32 : 38), 1],
    [2, xf, yp, xf + 30, yp + 24, 1],
    [3, xf, ya, xf - 24, ya - 20, -1],
    [4, xf, (ya + yp) / 2, xf - 30, (ya + yp) / 2, -1]
  ];
  for (const [i, anchorX, anchorY, bx2, by2, side] of calls) {
    stroke(tone(i));
    strokeWeight(1.5);
    line(anchorX, anchorY, bx2, by2);
    badge(i, bx2, by2, r);
    let tagW = 0;
    if (namesBox.checked()) {
      noStroke();
      fill(tone(i));
      textStyle(BOLD);
      textSize(ts);
      textAlign(side > 0 ? LEFT : RIGHT, CENTER);
      text(PARTS[i].tag, bx2 + side * (r + 4), by2);
      tagW = textWidth(PARTS[i].tag) + 6;
      textStyle(NORMAL);
    }
    hits.push({ i, x: side > 0 ? bx2 - r : bx2 - r - tagW, y: by2 - r, w: 2 * r + tagW, h: 2 * r });
    hits.push({ i, x: anchorX - 9, y: anchorY - 9, w: 18, h: 18 });
  }
  hits.push({ i: 1, x: ax, y: cy, w: bx - ax, h: ay - cy });
}

// The equation with this data's numbers, colored like the parts in the plot
function drawEquation(x0, y0, w, h, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const big = narrow ? 22 : 28, ts = narrow ? 11 : 13, gap = big * 0.45;
  noStroke();
  fill('dimgray');
  textSize(ts);
  textAlign(CENTER, TOP);
  text('Regression equation  ŷ = β₀ + β₁x  for these ten students', x0 + w / 2, y0 + 6);

  // [text, part index or -1, label underneath]
  const toks = [['ŷ', 2, 'predicted'], ['=', -1, ''], [nf(fit.b0, 1, 1), 0, 'intercept'], ['+', -1, ''],
    [nf(fit.b1, 1, 1), 1, 'slope'], ['x', -1, 'hours']];
  textSize(big);
  textStyle(BOLD);
  let total = -gap * 1.7;
  for (const t of toks) total += textWidth(t[0]) + gap;
  let x = x0 + w / 2 - total / 2;
  const by = y0 + (narrow ? 20 : 26);
  for (const [s, i, label] of toks) {
    if (s === 'x') x -= gap * 0.7;                          // 5.5x reads as one term
    textSize(big);
    textStyle(BOLD);
    const tw = textWidth(s);
    noStroke();
    fill(i >= 0 ? tone(i) : 'black');
    textAlign(LEFT, TOP);
    text(s, x, by);
    textStyle(NORMAL);
    textSize(ts);
    textAlign(s === 'x' ? LEFT : CENTER, TOP);              // the x label starts under x so it clears "slope"
    if (label && (i < 0 || namesBox.checked() || sel === i)) text(label, s === 'x' ? x : x + tw / 2, by + big + 2);
    if (i >= 0) hits.push({ i, x: x - 5, y: by - 2, w: tw + 10, h: big + ts + 6 });
    x += tw + gap;
  }
}

const signed = v => (v < 0 ? '−' : '+') + nf(Math.abs(v), 1, 1);

// The value of each part in this data, computed from the fit and the featured point
function valueText(i) {
  const x = HOURS[feat], y = SCORES[feat], yh = predict(x), b0 = nf(fit.b0, 1, 1), b1 = nf(fit.b1, 1, 1);
  const rise = predict(RUN_TO) - predict(RUN_FROM);
  return [
    'β₀ = ' + b0 + '. A student who studies 0 hours is predicted to score ' + b0 + ' points.',
    'β₁ = rise ÷ run = ' + nf(rise, 1, 1) + ' ÷ ' + (RUN_TO - RUN_FROM) + ' = ' + b1 + ' points for each extra hour studied.',
    'for x = ' + x + ' hours, ŷ = ' + b0 + ' + ' + b1 + ' × ' + x + ' = ' + nf(yh, 1, 1) + ' points.',
    'the student who studied ' + x + ' hours scored y = ' + y + ' points.',
    'y − ŷ = ' + y + ' − ' + nf(yh, 1, 1) + ' = ' + signed(y - yh) + ' points. The model predicted too ' +
      (y > yh ? 'low.' : 'high.')
  ][i];
}

// List of the five parts and the explanation of the selected one
function drawPanel(x0, y0, w, h, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const ts = narrow ? 12 : 15, lh = ts + 5, names = namesBox.checked();
  const rowH = narrow ? 28 : 44, cw = (w - 12) / 5;
  PARTS.forEach((p, i) => {
    const rx = narrow ? x0 + 6 + i * cw : x0 + 6, ry = narrow ? y0 + 5 : y0 + 6 + i * rowH;
    const rw = narrow ? cw - 2 : w - 12, show = names || sel === i;
    if (sel === i) {
      const c = color(p.color);
      c.setAlpha(45);
      noStroke();
      fill(c);
      rect(rx, ry, rw, rowH - 2, 7);
    }
    badge(i, rx + (narrow ? 12 : 18), ry + rowH / 2 - 1, narrow ? 10 : 12, true);
    noStroke();
    fill('black');
    textAlign(LEFT, narrow ? CENTER : TOP);
    textStyle(BOLD);
    textSize(narrow ? 11 : 15);
    if (narrow) text(show ? p.name.split(' ')[0] : '?', rx + 25, ry + rowH / 2);
    else text(show ? p.name + ' (' + p.sym + ')' : '?', rx + 38, ry + 4);
    textStyle(NORMAL);
    if (!narrow && show) {
      fill('dimgray');
      textSize(12.5);
      text(p.short, rx + 38, ry + 23);
    }
    hits.push({ i, x: rx, y: ry, w: rw, h: rowH - 2 });
  });

  const tx = x0 + 12, tw = w - 24, dy = narrow ? y0 + 38 : y0 + 12 + 5 * rowH;
  noStroke();
  textAlign(LEFT, TOP);
  textSize(ts);
  if (sel < 0) {
    fill('dimgray');
    text(names ? 'Hover over a numbered part to read about it. Click it to keep it selected, and click again to release it.\n\n' +
      'Then uncheck Show names, name each number from memory, and click to check.'
      : 'Self-test: name each numbered part, then hover over or click its number to check your answer.',
      tx, dy + 4, tw, y0 + h - dy - 8);
    return;
  }
  const p = PARTS[sel], boxH = (narrow ? 2 : 3) * lh + 8;
  fill(p.color);
  textStyle(BOLD);
  textSize(ts + 2);
  text((sel + 1) + '. ' + p.name + ' (' + p.sym + ')', tx, dy + 2);
  textStyle(NORMAL);
  fill('black');
  textSize(ts);
  text(p.notes.map(s => '• ' + s).join('\n'), tx, dy + lh + 8, tw, y0 + h - boxH - 10 - (dy + lh + 8));
  fill('lightyellow');
  stroke('silver');
  strokeWeight(1);
  rect(x0 + 6, y0 + h - boxH - 6, w - 12, boxH, 6);
  noStroke();
  fill('black');
  text('In this data: ' + valueText(sel), tx, y0 + h - boxH - 2, tw, boxH);
}

function mousePressed() {
  if (mouseY < 0 || mouseY > drawHeight || mouseX < 0 || mouseX > canvasWidth) return;
  const p = partAt(mouseX, mouseY);
  locked = p === locked ? -1 : p;
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
