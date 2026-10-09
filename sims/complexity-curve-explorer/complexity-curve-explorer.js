// Complexity Curve Explorer
// CANVAS_HEIGHT: 615
// Bloom L3 and L5 (Apply, Evaluate): students set the degree of a polynomial model, read its
// training and test error, and judge which degree is best before checking with Find Best Degree.
//
// Model: 30 points from y = 0.12 (x - 1)(x - 5)(x - 9) + 10 plus normal noise (SD 1.2), from a
// seeded generator. 20 are training points and 10 are test points. For every degree from 1 to 15
// the least-squares polynomial is fit to the training points only. The fit is built from
// polynomials that are orthogonal on the training x values (Forsythe's recurrence). It is the
// same curve np.polyfit finds, but it stays accurate at degree 15, where solving the normal
// equations directly loses most of its digits. MSE = mean of (y - prediction)^2, computed on the
// training points and again on the test points.
// Zones: a degree is in the sweet spot when its test MSE is within 25% of the lowest test MSE.
// Degrees below the simplest sweet-spot degree are underfitting; the remaining degrees are overfitting.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 500;
let controlHeight = 115;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let sliderLeftMargin = 190;
let defaultTextSize = 16;

const N = 30, N_TRAIN = 20, MAX_DEG = 15, NOISE_SD = 1.2, SWEET_TOL = 1.25;
const truth = x => 0.12 * (x - 1) * (x - 5) * (x - 9) + 10;
const ZONES = {
  under: { name: 'Underfitting', curve: 'darkorange', ink: 'chocolate', fill: [255, 165, 0, 45] },
  good: { name: 'Good fit', curve: 'forestgreen', ink: 'darkgreen', fill: [60, 179, 113, 80] },
  over: { name: 'Overfitting', curve: 'crimson', ink: 'firebrick', fill: [220, 20, 60, 35] },
  hidden: { name: 'Test error is hidden', curve: 'slategray', ink: 'black' }
};

let rngState = 1, seed = 40;
let xs = [], ys = [], train = [], test = [];
let alpha = [], beta = [], coef = [];       // recurrence terms and coefficients of the orthogonal fit
let mseTrain = [], mseTest = [];            // index = polynomial degree
let best = 1, simplest = 1, zoneOf = [];     // lowest-test-MSE degree, simplest sweet-spot degree, zone per degree
let degreeSlider, bestButton, dataButton, trainCheckbox, testCheckbox;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }

// The points are independent draws, so the first 20 make a random training set
function makeData() {
  rngState = seed;
  xs = []; ys = [];
  for (let i = 0; i < N; i++) {
    xs.push(10 * uniform());
    ys.push(truth(xs[i]) + NOISE_SD * stdNormal());
  }
  train = xs.map((_, i) => i).slice(0, N_TRAIN);
  test = xs.map((_, i) => i).slice(N_TRAIN);
  fitAll();
}

// Least-squares fits of every degree at once. With t = (x - 5) / 5:
//   p_0 = 1,  p_(k+1) = (t - a_k) p_k - b_k p_(k-1),
//   a_k = <t p_k, p_k> / <p_k, p_k>,  b_k = <p_k, p_k> / <p_(k-1), p_(k-1)>,  c_k = <y, p_k> / <p_k, p_k>,
// where <u, v> sums over the training points. The degree-d fit is c_0 p_0 + ... + c_d p_d.
function fitAll() {
  const t = train.map(i => (xs[i] - 5) / 5), y = train.map(i => ys[i]);
  let pPrev = t.map(() => 0), p = t.map(() => 1), normPrev = 1;
  for (let k = 0; k <= MAX_DEG; k++) {
    const norm = p.reduce((s, v) => s + v * v, 0);
    coef[k] = p.reduce((s, v, i) => s + v * y[i], 0) / norm;
    alpha[k] = p.reduce((s, v, i) => s + t[i] * v * v, 0) / norm;
    beta[k] = k === 0 ? 0 : norm / normPrev;
    const next = p.map((v, i) => (t[i] - alpha[k]) * v - beta[k] * pPrev[i]);
    pPrev = p; p = next; normPrev = norm;
  }
  const mse = rows => {
    const sum = new Array(MAX_DEG + 1).fill(0);
    for (const i of rows) predictAll(xs[i]).forEach((f, d) => { sum[d] += (ys[i] - f) ** 2; });
    return sum.map(v => v / rows.length);
  };
  mseTrain = mse(train);
  mseTest = mse(test);
  best = 1;
  for (let d = 2; d <= MAX_DEG; d++) if (mseTest[d] < mseTest[best]) best = d;
  const isGood = d => mseTest[d] <= SWEET_TOL * mseTest[best];
  simplest = 1;
  while (!isGood(simplest)) simplest++;
  for (let d = 1; d <= MAX_DEG; d++) zoneOf[d] = isGood(d) ? 'good' : d < simplest ? 'under' : 'over';
}

// Predictions of the fits of degree 0, 1, ..., MAX_DEG at one x value
function predictAll(x) {
  const t = (x - 5) / 5, out = [];
  let pPrev = 0, p = 1, sum = 0;
  for (let k = 0; k <= MAX_DEG; k++) {
    sum += coef[k] * p;
    out.push(sum);
    const next = (t - alpha[k]) * p - beta[k] * pPrev;
    pPrev = p; p = next;
  }
  return out;
}

function formatMSE(v) { return v < 100 ? nf(v, 1, 2) : v < 1e6 ? Math.round(v).toLocaleString('en-US') : 'over 1,000,000'; }

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  degreeSlider = createSlider(1, MAX_DEG, 1, 1);
  degreeSlider.parent(mainElement);
  degreeSlider.position(sliderLeftMargin, drawHeight + 8);
  degreeSlider.size(canvasWidth - sliderLeftMargin - margin);

  bestButton = createButton('Find Best Degree');
  bestButton.parent(mainElement);
  bestButton.position(10, drawHeight + 45);
  bestButton.mousePressed(() => { testCheckbox.checked(true); degreeSlider.value(best); });

  dataButton = createButton('New Data');
  dataButton.parent(mainElement);
  dataButton.position(140, drawHeight + 45);
  dataButton.mousePressed(() => { seed++; makeData(); });

  trainCheckbox = createCheckbox(' Show training error', true);
  testCheckbox = createCheckbox(' Show test error', true);
  [trainCheckbox, testCheckbox].forEach((box, i) => {
    box.parent(mainElement);
    box.position(10 + i * 190, drawHeight + 82);
    box.style('font-size', '16px');
  });

  makeData();

  describe('A scatter plot of twenty training points and ten test points with a polynomial curve whose degree is set by ' +
    'a slider. Below it, a chart of training and test mean squared error for degrees 1 to 15 with shaded underfitting, ' +
    'sweet spot, and overfitting zones. A panel reports both errors, their gap, and a verdict on the fit.', LABEL);
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600;
  const w = canvasWidth - 2 * margin;
  const d = degreeSlider.value(), showTest = testCheckbox.checked();
  const zone = showTest ? zoneOf[d] : 'hidden';

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Complexity Curve Explorer', canvasWidth / 2, 8);

  if (narrow) {
    drawData(margin, 36, w, 160, d, zone, narrow);
    drawVerdict(margin, 202, w, 98, d, zone, narrow);
    drawChart(margin, 306, w, drawHeight - 314, d, narrow);
  } else {
    const dataW = w * 0.6;
    drawData(margin, 42, dataW, 232, d, zone, narrow);
    drawVerdict(margin + dataW + 10, 42, w - dataW - 10, 232, d, zone, narrow);
    drawChart(margin, 282, w, drawHeight - 290, d, narrow);
  }

  // control label
  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Polynomial degree: ' + d, 10, drawHeight + 18);
}

function drawPanelBox(x, y, w, h, title, ts) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text(title, x + 10, y + 7);
  textStyle(NORMAL);
  textSize(ts);
}

// A row of legend entries [label, color, 'line' | 'dot' | 'box'], right-aligned at xRight
function drawLegend(xRight, y, items, ts) {
  textSize(ts);
  textAlign(LEFT, CENTER);
  let x = xRight - items.reduce((s, it) => s + textWidth(it[0]) + 28, 0) + 10;
  for (const [label, col, kind] of items) {
    stroke(col);
    strokeWeight(kind === 'line' ? 3 : 1);
    fill(col);
    if (kind === 'line') line(x, y, x + 14, y);
    else if (kind === 'dot') circle(x + 7, y, 8);
    else rect(x + 1, y - 6, 12, 12);
    noStroke();
    fill('black');
    text(label, x + 18, y);
    x += textWidth(label) + 28;
  }
}

// The data and the fitted curve of the chosen degree
function drawData(x0, y0, w, h, d, zone, narrow) {
  const ts = narrow ? 11 : 13;
  drawPanelBox(x0, y0, w, h, 'Degree ' + d + (narrow ? ' fit' : ' polynomial fit'), ts);
  drawLegend(x0 + w - 8, y0 + 14, [['training (20)', 'royalblue', 'dot'], ['test (10)', 'darkorange', 'box']], ts);

  const px = x0 + 34, py = y0 + 28, pw = w - 48, ph = h - 50;
  const yLo = Math.floor(Math.min(...ys) / 5) * 5, yHi = Math.ceil(Math.max(...ys) / 5) * 5;
  const gx = v => px + v / 10 * pw, gy = v => py + ph - (constrain(v, yLo - 50, yHi + 50) - yLo) / (yHi - yLo) * ph;
  textSize(narrow ? 11 : 12);
  for (let v = yLo; v <= yHi; v += 5) {
    stroke('gainsboro');
    strokeWeight(1);
    line(px, gy(v), px + pw, gy(v));
    noStroke();
    fill('dimgray');
    textAlign(RIGHT, CENTER);
    text(v, px - 4, gy(v));
  }
  textAlign(CENTER, TOP);
  for (let v = 0; v <= 10; v += 2) text(v, gx(v), py + ph + 3);
  stroke('gray');
  strokeWeight(1.5);
  line(px, py, px, py + ph);
  line(px, py + ph, px + pw, py + ph);

  // the curve leaves the plot when a high-degree fit swings far above or below the data
  push();
  drawingContext.beginPath();
  drawingContext.rect(px, py, pw, ph);
  drawingContext.clip();
  noFill();
  stroke(ZONES[zone].curve);
  strokeWeight(3);
  beginShape();
  for (let i = 0; i <= 200; i++) vertex(gx(i / 20), gy(predictAll(i / 20)[d]));
  endShape();
  pop();

  stroke('white');
  strokeWeight(1);
  fill('royalblue');
  for (const i of train) circle(gx(xs[i]), gy(ys[i]), narrow ? 7 : 9);
  fill('darkorange');
  for (const i of test) square(gx(xs[i]) - 4, gy(ys[i]) - 4, 8);
}

// Errors of the chosen degree and a verdict on the fit
function drawVerdict(x0, y0, w, h, d, zone, narrow) {
  const ts = narrow ? 12 : 14, lh = ts + 6, z = ZONES[zone], hidden = zone === 'hidden';
  drawPanelBox(x0, y0, w, h, 'Degree ' + d + ' polynomial (' + (d + 1) + ' coefficients)', ts);
  const gap = mseTest[d] - mseTrain[d];
  let y = y0 + 9 + lh;
  const parts = [['Training MSE: ' + formatMSE(mseTrain[d]), 'mediumblue'],
    ['Test MSE: ' + (hidden ? 'hidden' : formatMSE(mseTest[d])), 'chocolate'],
    ['Gap (test − train): ' + (hidden ? 'hidden' : formatMSE(gap)), 'black']];
  if (narrow) {                       // one line: train, test, gap
    let x = x0 + 10;
    for (const [str, col] of [[parts[0][0].replace('Training', 'Train'), parts[0][1]], parts[1], [parts[2][0].replace(' (test − train)', ''), 'black']]) {
      fill(col);
      text(str, x, y);
      x += textWidth(str) + 14;
    }
    y += lh;
  } else {
    for (const [str, col] of parts) {
      fill(col);
      text(str, x0 + 10, y);
      y += lh;
    }
    y += 6;
  }
  const ratio = mseTest[d] / mseTest[best];
  const msg = {
    under: 'A degree ' + d + ' curve is too stiff to follow the pattern. Its training error is ' + nf(mseTrain[d] / mseTrain[simplest], 1, 1) +
      ' times and its test error ' + nf(mseTest[d] / mseTest[simplest], 1, 1) + ' times the error at degree ' + simplest + '.',
    good: (d === best ? 'Test error is at its lowest here.' : 'Test error is within 25% of its lowest value (reached at degree ' + best + ').') +
      // a test error far below the training error means the 10 test points are not typical
      (mseTest[d] < 0.5 * mseTrain[d] ? ' But it is far below the training error, so these test points are unusually easy. Try New Data.' :
        d > simplest ? ' Degree ' + simplest + ' does about as well with fewer coefficients, so it is the safer choice.' :
          ' No simpler curve tests this well.'),
    over: 'The extra flexibility is spent on noise in the training points. Training error keeps falling, but test error is ' +
      (ratio < 10 ? nf(ratio, 1, 1) : ratio < 1000 ? Math.round(ratio) : 'over 1,000') + ' times its lowest value.',
    hidden: 'Training error never rises as the degree goes up, so on its own it always favors the most complex model. ' +
      'Check Show test error to judge the fit.'
  }[zone];
  fill(z.ink);
  textWrap(WORD);
  if (narrow) {                       // heading and message share one wrapped paragraph
    text(z.name + ': ' + msg, x0 + 10, y, w - 20, y0 + h - y - 2);
    return;
  }
  textStyle(BOLD);
  textSize(ts + 2);
  text(z.name, x0 + 10, y);
  textStyle(NORMAL);
  textSize(ts);
  fill('black');
  text(msg, x0 + 10, y + lh + 4, w - 20, y0 + h - y - lh - 6);
}

// Training and test MSE for every degree, with the three zones shaded
function drawChart(x0, y0, w, h, d, narrow) {
  const ts = narrow ? 11 : 13, showTrain = trainCheckbox.checked(), showTest = testCheckbox.checked();
  drawPanelBox(x0, y0, w, h, 'Error vs. model complexity', ts);
  const legend = [[narrow ? 'Train' : 'Training MSE', 'royalblue', 'line'], [narrow ? 'Test' : 'Test MSE', 'darkorange', 'line']];
  if (showTest) {
    legend.push([narrow ? 'Underfit' : 'Underfitting', color(...ZONES.under.fill), 'box'], ['Sweet spot', color(...ZONES.good.fill), 'box'],
      [narrow ? 'Overfit' : 'Overfitting', color(...ZONES.over.fill), 'box']);
  }
  drawLegend(x0 + w - 8, y0 + (narrow ? 32 : 15), legend, ts);

  const px = x0 + (narrow ? 44 : 56), py = y0 + (narrow ? 46 : 32), pw = x0 + w - px - 14, ph = y0 + h - py - (narrow ? 30 : 36);
  const band = pw / MAX_DEG, gx = deg => px + (deg - 0.5) * band;
  // the axis is tall enough for the degree 1 errors; larger errors leave through the top
  const yTop = 1.25 * Math.max(mseTrain[1], mseTest[1]);
  const gy = v => py + ph - Math.min(v, 2 * yTop) / yTop * ph;
  const tick = [0.5, 1, 2, 2.5, 5, 10, 20, 50].find(t => yTop / t <= 5) || 100;

  noStroke();
  for (let deg = 1; deg <= MAX_DEG && showTest; deg++) {
    fill(...ZONES[zoneOf[deg]].fill);
    rect(px + (deg - 1) * band, py, band + 0.5, ph);
  }
  textSize(narrow ? 11 : 12);
  for (let v = 0; v <= yTop; v += tick) {
    stroke('gainsboro');
    strokeWeight(1);
    line(px, gy(v), px + pw, gy(v));
    noStroke();
    fill('dimgray');
    textAlign(RIGHT, CENTER);
    text(v, px - 4, gy(v));
  }
  textAlign(CENTER, TOP);
  for (let deg = 1; deg <= MAX_DEG; deg++) {
    noStroke();
    fill(deg === d ? 'black' : 'dimgray');
    textStyle(deg === d ? BOLD : NORMAL);
    if (!narrow || deg % 2 === 1 || deg === d) text(deg, gx(deg), py + ph + 3);
  }
  textStyle(NORMAL);
  fill('black');
  text('polynomial degree (model complexity)', px + pw / 2, py + ph + (narrow ? 15 : 18));
  push();
  translate(x0 + 12, py + ph / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, CENTER);
  text('MSE', 0, 0);
  pop();
  stroke('gray');
  strokeWeight(1.5);
  line(px, py, px, py + ph);
  line(px, py + ph, px + pw, py + ph);

  // marker for the chosen degree
  stroke('black');
  strokeWeight(1.5);
  drawingContext.setLineDash([5, 4]);
  line(gx(d), py, gx(d), py + ph);
  drawingContext.setLineDash([]);

  push();
  drawingContext.beginPath();
  drawingContext.rect(px, py - 1, pw, ph + 8);
  drawingContext.clip();
  for (const [show, mse, col] of [[showTrain, mseTrain, 'royalblue'], [showTest, mseTest, 'darkorange']]) {
    if (!show) continue;
    noFill();
    stroke(col);
    strokeWeight(2.5);
    beginShape();
    for (let deg = 1; deg <= MAX_DEG; deg++) vertex(gx(deg), gy(mse[deg]));
    endShape();
    stroke('white');
    strokeWeight(1);
    fill(col);
    for (let deg = 1; deg <= MAX_DEG; deg++) circle(gx(deg), gy(mse[deg]), deg === d ? 13 : 7);
  }
  pop();
  // an arrow at the top marks each test error that is off the chart
  noStroke();
  fill('darkorange');
  for (let deg = 1; deg <= MAX_DEG && showTest; deg++) {
    if (mseTest[deg] > yTop) triangle(gx(deg) - 5, py + 8, gx(deg) + 5, py + 8, gx(deg), py);
  }
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  degreeSlider.size(canvasWidth - sliderLeftMargin - margin);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
