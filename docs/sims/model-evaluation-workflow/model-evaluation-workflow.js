// Model Evaluation Workflow
// CANVAS_HEIGHT: 650
// Bloom L3-L4 (Apply, Analyze): students step through the ten-step evaluation pipeline on a real
// data set, see which numbers exist at each step, and compare the honest order of operations with a
// leaky one in which the test data is used to choose the model.
//
// Model: 100 points with x uniform on 0 to 10 and y = 0.12 (x - 1)(x - 5)(x - 9) + 10 + N(0, 1.2), from a
// seeded generator. The first 80 are training rows and the last 20 are test rows. Polynomials of degree
// 1 to 5 (the chapter's param_grid) are fit by least squares using polynomials that are orthogonal on
// the x values (Forsythe's recurrence), which gives the same fit as LinearRegression on
// PolynomialFeatures. R^2 = 1 - sum (y - prediction)^2 / sum (y - mean y)^2 on the rows being scored.
// CV: 5 folds of 16 consecutive training rows (KFold without shuffling), mean and np.std of the fold R^2.
// Honest choice: highest CV mean. Leaky choice: highest test R^2, which can never be below the honest
// model's test R^2 because the honest model is one of the five candidates.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 570;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const N = 100, N_TRAIN = 80, K = 5, MAX_DEG = 5, NOISE_SD = 1.2;
const LANES = [['Full Dataset', 'whitesmoke'], ['Training Data', 'honeydew'], ['Validation / CV', 'lightyellow'], ['Test Data', 'mistyrose']];
const COLORS = { gray: ['gainsboro', 'dimgray'], blue: ['lightskyblue', 'steelblue'], green: ['palegreen', 'forestgreen'],
  yellow: ['khaki', 'darkgoldenrod'], red: ['lightpink', 'crimson'], purple: ['plum', 'purple'] };
// lane -1 spans every lane
const STEPS = [
  { name: 'Load Complete Dataset', label: 'Load Complete\nDataset', lane: 0, color: 'gray', hover: 'All of the data, before any splits.' },
  { name: 'Split into Train and Test', label: 'Split into\nTrain and Test', lane: 0, color: 'blue', hover: 'Typically an 80/20 split. The test data is locked away.' },
  { name: 'Train Initial Model', label: 'Train Initial\nModel', lane: 1, color: 'green', hover: 'Fit the model on the training data only.' },
  { name: 'Cross-Validate', label: 'Cross-Validate', lane: 1, color: 'green', hover: 'Get a reliable performance estimate using K-fold CV.' },
  { name: 'Try Different Models?', label: 'Try Different\nModels?', lane: 2, color: 'yellow', hover: 'Compare polynomial degrees, feature sets, and algorithms.' },
  { name: 'Hyperparameter Tuning', label: 'Hyperparameter\nTuning', lane: 1, color: 'green', hover: 'Use GridSearchCV or similar to find the best settings.' },
  { name: 'Select Best Model', label: 'Select Best\nModel', lane: 2, color: 'yellow', hover: 'Choose by validation or CV performance, not training performance.' },
  { name: 'Final Evaluation', label: 'Final\nEvaluation', lane: 3, color: 'red', hover: 'Only now touch the test data. This is your honest grade.' },
  { name: 'Residual Analysis', label: 'Residual\nAnalysis', lane: 3, color: 'red', hover: 'Check for patterns and validate assumptions.' },
  { name: 'Report Results', label: 'Report Results', lane: -1, color: 'purple', hover: 'Report the test metrics together with their uncertainty.' }
];

let rngState = 1, seed = 17;
let xs = [], ys = [], train = [], test = [];
let trainR2 = [], testR2 = [], cvMean = [], cvStd = [], firstFolds = [];   // index = degree; fold scores of degree 1
let bestCV = 1, bestTest = 1, stats = [];           // stats[d] = test RMSE, residual mean, residual SD
let step = 1;                                       // 1 to 10
let chart = {}, detail = {}, boxes = [];
let prevButton, nextButton, dataButton, leakCheckbox;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }
function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }
function f3(v) { return (v < 0 ? '−' : '') + Math.abs(v).toFixed(3); }
function f2(v) { return (v < 0 ? '−' : '') + Math.abs(v).toFixed(2); }

// Least-squares polynomial fits of degree 0 to MAX_DEG to the listed rows. With t = (x - 5) / 5:
//   p_0 = 1,  p_(k+1) = (t - a_k) p_k - b_k p_(k-1),  c_k = <y, p_k> / <p_k, p_k>,
//   a_k = <t p_k, p_k> / <p_k, p_k>,  b_k = <p_k, p_k> / <p_(k-1), p_(k-1)>.
// Returns a function that gives the prediction of every degree at one x.
function fitPoly(rows) {
  const t = rows.map(i => (xs[i] - 5) / 5), y = rows.map(i => ys[i]), alpha = [], beta = [], coef = [];
  let pPrev = t.map(() => 0), p = t.map(() => 1), normPrev = 1;
  for (let k = 0; k <= MAX_DEG; k++) {
    const norm = p.reduce((s, v) => s + v * v, 0);
    coef[k] = p.reduce((s, v, i) => s + v * y[i], 0) / norm;
    alpha[k] = p.reduce((s, v, i) => s + t[i] * v * v, 0) / norm;
    beta[k] = k === 0 ? 0 : norm / normPrev;
    const next = p.map((v, i) => (t[i] - alpha[k]) * v - beta[k] * pPrev[i]);
    pPrev = p; p = next; normPrev = norm;
  }
  return x => {
    const tx = (x - 5) / 5, out = [];
    let a = 0, b = 1, sum = 0;
    for (let k = 0; k <= MAX_DEG; k++) {
      sum += coef[k] * b;
      out.push(sum);
      const next = (tx - alpha[k]) * b - beta[k] * a;
      a = b; b = next;
    }
    return out;
  };
}

function r2All(predict, rows) {
  const my = mean(rows.map(i => ys[i])), sse = new Array(MAX_DEG + 1).fill(0);
  let sst = 0;
  for (const i of rows) {
    sst += (ys[i] - my) ** 2;
    predict(xs[i]).forEach((f, d) => { sse[d] += (ys[i] - f) ** 2; });
  }
  return sse.map(v => 1 - v / sst);
}

// Run the whole pipeline once. The steps only decide which of these numbers are shown.
function runPipeline() {
  rngState = seed;
  xs = []; ys = [];
  for (let i = 0; i < N; i++) {
    xs.push(10 * uniform());
    ys.push(0.12 * (xs[i] - 1) * (xs[i] - 5) * (xs[i] - 9) + 10 + NOISE_SD * stdNormal());
  }
  train = xs.map((_, i) => i).slice(0, N_TRAIN);
  test = xs.map((_, i) => i).slice(N_TRAIN);
  const model = fitPoly(train), size = N_TRAIN / K, folds = [];
  trainR2 = r2All(model, train);
  testR2 = r2All(model, test);
  for (let k = 0; k < K; k++) {
    const held = train.slice(k * size, (k + 1) * size), rest = train.filter(i => i < k * size || i >= (k + 1) * size);
    folds.push(r2All(fitPoly(rest), held));
  }
  firstFolds = folds.map(f => f[1]);
  bestCV = 1; bestTest = 1;
  for (let d = 1; d <= MAX_DEG; d++) {
    const scores = folds.map(f => f[d]);
    cvMean[d] = mean(scores);
    cvStd[d] = Math.sqrt(mean(scores.map(v => (v - cvMean[d]) ** 2)));
    if (cvMean[d] > cvMean[bestCV]) bestCV = d;
    if (testR2[d] > testR2[bestTest]) bestTest = d;
    const res = test.map(i => ys[i] - model(xs[i])[d]), m = mean(res);      // residual = actual - predicted
    stats[d] = { rmse: Math.sqrt(mean(res.map(v => v * v))), mean: m,
      sd: Math.sqrt(res.reduce((s, v) => s + (v - m) ** 2, 0) / (res.length - 1)) };
  }
}

// What happens at the current step on this data set
function stepText(leak) {
  const nTest = N - N_TRAIN, pick = leak ? bestTest : bestCV, s = stats[pick];
  switch (step) {
    case 1: return 'This data set has ' + N + ' rows, each with one feature x and a target y. Nothing has been fit yet.';
    case 2: return N_TRAIN + ' rows go to training and ' + nTest + ' to the test set. Split first: every later step that learns from data must see only the training rows.';
    case 3: return 'A straight line (degree 1) is fit to the ' + N_TRAIN + ' training rows. Training R² = ' + f3(trainR2[1]) +
      '. It is graded on rows the model has already seen, so it is too optimistic to trust.';
    case 4: return 'Each fold of ' + N_TRAIN / K + ' training rows is scored by a model fit to the other ' + (N_TRAIN - N_TRAIN / K) + '. Fold R²: ' +
      firstFolds.map(f2).join(', ') + '. Mean ' + f3(cvMean[1]) + ' ± ' + f3(cvStd[1]) + '.';
    case 5: return leak ? 'LEAK: the degree 1 model is scored on the test rows (R² = ' + f3(testR2[1]) + ') to decide what to try next. The test set has now influenced a choice.' :
      'One model is not a comparison. Loop back to steps 3 and 4 with polynomial degrees 2 to ' + MAX_DEG + ', using the training rows only.';
    case 6: return (leak ? 'LEAK: every candidate is also scored on the test rows, before any choice has been made. ' : '') +
      'GridSearchCV runs the loop: ' + MAX_DEG + ' degrees × ' + K + ' folds = ' + MAX_DEG * K + ' fits, all inside the training rows. The table shows each CV mean.';
    case 7: return leak ? 'LEAK: degree ' + bestTest + ' is selected because it has the highest test R² (' + f3(testR2[bestTest]) + '). ' +
        (bestTest === bestCV ? 'CV selects the same degree on this sample, but the test score is no longer an independent check.' : 'CV would have selected degree ' + bestCV + '.') :
      'Degree ' + bestCV + ' has the highest CV mean (' + f3(cvMean[bestCV]) + ' ± ' + f3(cvStd[bestCV]) + '), so it is selected. Training R² is highest for degree ' +
        MAX_DEG + ' (' + f3(trainR2[MAX_DEG]) + '), as it always is for the most flexible model.';
    case 8: return leak ? 'Degree ' + pick + ' scores R² = ' + f3(testR2[pick]) + ' on the test rows, but that score was the highest of ' + MAX_DEG +
        ' and was used to choose the model. It is biased upward, and no unseen data is left to check it.' :
      'The test set is unlocked once. Degree ' + pick + ', fit to all ' + N_TRAIN + ' training rows, scores R² = ' + f3(testR2[pick]) + ' and RMSE = ' + f2(s.rmse) +
        ' on the ' + nTest + ' test rows. The number can be trusted because the test rows played no part in any choice.';
    case 9: return 'Test residuals (actual − predicted) for degree ' + pick + ': mean ' + f2(s.mean) + ', which should be near 0, and standard deviation ' + f2(s.sd) +
      '. A residual plot should show no curve, funnel, or clusters.';
    default: return leak ? 'Leaky report: degree ' + bestTest + ', test R² = ' + f3(testR2[bestTest]) + '. The honest workflow reports degree ' + bestCV + ', test R² = ' +
        f3(testR2[bestCV]) + '. ' + (f3(testR2[bestTest]) === f3(testR2[bestCV]) ? 'The scores match on this sample, but the leaky one was picked as the best of ' + MAX_DEG + ' looks at the test rows, so it is not an independent check.' :
          'The highest of ' + MAX_DEG + ' test scores can only match or beat the honest number, so it overstates how the model will do on new data.') :
      'Report: polynomial degree ' + pick + '. CV R² = ' + f3(cvMean[pick]) + ' ± ' + f3(cvStd[pick]) + ' on training data. Test R² = ' + f3(testR2[pick]) +
        ', RMSE = ' + f2(s.rmse) + ' on ' + nTest + ' unseen rows. Do not go back and tune after this.';
  }
}

function setStep(n) {
  step = constrain(n, 1, STEPS.length);
  const setEnabled = (b, on) => { if (on) b.removeAttribute('disabled'); else b.attribute('disabled', ''); };
  setEnabled(prevButton, step > 1);
  setEnabled(nextButton, step < STEPS.length);
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  prevButton = createButton('Previous');
  prevButton.parent(mainElement);
  prevButton.position(10, drawHeight + 10);
  prevButton.mousePressed(() => setStep(step - 1));
  nextButton = createButton('Next');
  nextButton.parent(mainElement);
  nextButton.position(85, drawHeight + 10);
  nextButton.mousePressed(() => setStep(step + 1));
  dataButton = createButton('New Data');
  dataButton.parent(mainElement);
  dataButton.position(138, drawHeight + 10);
  dataButton.mousePressed(() => { seed++; runPipeline(); });
  leakCheckbox = createCheckbox(' Use the test data to choose the model (a leak)', false);
  leakCheckbox.parent(mainElement);
  leakCheckbox.position(10, drawHeight + 45);
  leakCheckbox.style('font-size', '15px');

  runPipeline();
  setStep(1);

  describe('A flowchart of the ten-step model evaluation workflow in four swimlanes: full dataset, training data, ' +
    'validation and cross-validation, and test data. Previous and Next buttons move through the steps. A panel shows what ' +
    'each step computes on a real data set, and a table of scores fills in as the steps are completed. A checkbox switches ' +
    'to a leaky workflow in which the test data is used to choose the model.', LABEL);
}

function layout(narrow) {
  const w = canvasWidth - 2 * margin, top = narrow ? 36 : 42;
  chart = narrow ? { x: margin, y: top, w, h: 352 } : { x: margin, y: top, w: Math.round(w * 0.58), h: drawHeight - top - 10 };
  detail = narrow ? { x: margin, y: top + 358, w, h: drawHeight - top - 366 } : { x: chart.x + chart.w + 10, y: top, w: w - chart.w - 10, h: chart.h };
  const headH = narrow ? 20 : 26, laneW = chart.w / 4, pitch = (chart.h - headH - 4) / STEPS.length, bh = pitch - (narrow ? 6 : 14);
  boxes = STEPS.map((s, i) => {
    const cy = chart.y + headH + 2 + pitch * (i + 0.5), lane = Math.max(s.lane, 0), span = s.lane < 0 ? 4 : 1;
    return { x: chart.x + lane * laneW + 3, y: cy - bh / 2, w: span * laneW - 6, h: bh, cx: chart.x + (lane + span / 2) * laneW, cy };
  });
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600, leak = leakCheckbox.checked();
  layout(narrow);
  cursor(boxes.some(b => inBox(b)) ? HAND : ARROW);

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Model Evaluation Workflow', canvasWidth / 2, 8);

  drawChart(narrow, leak);
  drawDetail(narrow, leak);

  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(narrow ? 14 : defaultTextSize);
  text('Step ' + step + ' of ' + STEPS.length + (narrow ? '' : ': ' + STEPS[step - 1].name), 225, drawHeight + 22);
}

function inBox(b) { return mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h; }

function arrowLine(pts, col, dashed) {
  stroke(col);
  strokeWeight(1.5);
  noFill();
  if (dashed) drawingContext.setLineDash([5, 4]);
  for (let i = 0; i < pts.length - 1; i++) line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]);
  drawingContext.setLineDash([]);
  const [x1, y1] = pts[pts.length - 2], [x2, y2] = pts[pts.length - 1], a = atan2(y2 - y1, x2 - x1);
  fill(col);
  noStroke();
  triangle(x2, y2, x2 - 7 * cos(a - 0.45), y2 - 7 * sin(a - 0.45), x2 - 7 * cos(a + 0.45), y2 - 7 * sin(a + 0.45));
}

// A padlock. Open means the shackle has swung clear of the body.
function drawLock(x, y, s, open, col) {
  noFill();
  stroke(col);
  strokeWeight(2);
  arc(x + (open ? s * 0.8 : 0), y - s * 0.2, s * 0.9, s * 1.2, PI, TWO_PI);
  noStroke();
  fill(col);
  rect(x - s * 0.7, y - s * 0.2, s * 1.4, s, 2);
}

function drawWarning(x, y, s) {
  fill('gold');
  stroke('crimson');
  strokeWeight(1.5);
  triangle(x - s, y + s * 0.8, x + s, y + s * 0.8, x, y - s);
  noStroke();
  fill('crimson');
  textAlign(CENTER, CENTER);
  textStyle(BOLD);
  textSize(s * 1.3);
  text('!', x, y + s * 0.2);
  textStyle(NORMAL);
}

function drawChart(narrow, leak) {
  const laneW = chart.w / 4, headH = narrow ? 20 : 26, ts = narrow ? 11 : 13;
  const leaked = leak && step >= 5, unlocked = step >= 8;
  textSize(ts);
  LANES.forEach(([name, tint], k) => {
    fill(tint);
    stroke('silver');
    strokeWeight(1);
    rect(chart.x + k * laneW, chart.y, laneW, chart.h);
    noStroke();
    fill('black');
    textStyle(BOLD);
    textAlign(CENTER, CENTER);
    text(name, chart.x + (k + 0.5) * laneW - (k === 3 ? 9 : 0), chart.y + headH / 2);
    textStyle(NORMAL);
  });
  drawLock(chart.x + chart.w - (narrow ? 13 : 18), chart.y + headH / 2 + 1, narrow ? 6 : 7, leaked || unlocked, leaked ? 'crimson' : unlocked ? 'forestgreen' : 'gray');

  // the point of no return sits between Select Best Model and Final Evaluation
  const barrierY = (boxes[6].y + boxes[6].h + boxes[7].y) / 2;
  stroke('firebrick');
  strokeWeight(2.5);
  drawingContext.setLineDash([8, 5]);
  line(chart.x + 4, barrierY, chart.x + chart.w - 4, barrierY);
  drawingContext.setLineDash([]);
  noStroke();
  fill('firebrick');
  textAlign(LEFT, BOTTOM);
  text('Point of no return' + (narrow ? '' : ': choices are final'), chart.x + 8, barrierY - 3);

  // held-out test rows: from the split straight to the final evaluation
  const heldX = chart.x + 3.75 * laneW, b2 = boxes[1], b8 = boxes[7];
  arrowLine([[b2.x + b2.w, b2.cy], [heldX, b2.cy], [heldX, b8.y]], leaked ? 'crimson' : 'gray', true);
  noStroke();
  fill('dimgray');
  textAlign(CENTER, BOTTOM);
  text('20% held out', chart.x + 2.9 * laneW, b2.cy - 3);

  // main flow: sideways out of one step, then down into the next
  for (let i = 0; i < STEPS.length - 1; i++) {
    const a = boxes[i], b = boxes[i + 1], bx = STEPS[i + 1].lane < 0 ? a.cx : b.cx;
    if (Math.abs(bx - a.cx) < 1) arrowLine([[a.cx, a.y + a.h], [bx, b.y]], 'dimgray');
    else arrowLine([[bx > a.cx ? a.x + a.w : a.x, a.cy + 4], [bx, a.cy + 4], [bx, b.y]], 'dimgray');
  }
  // iteration loop from the decision back to Train Initial Model
  const b3 = boxes[2], b5 = boxes[4], loopX = b5.cx + b5.w * 0.32;
  arrowLine([[loopX, b5.y], [loopX, b3.cy], [b3.x + b3.w, b3.cy]], 'darkgoldenrod');
  noStroke();
  fill('darkgoldenrod');
  textAlign(CENTER, BOTTOM);
  text(narrow ? 'loop' : 'yes: loop back', b5.cx, b3.cy - 3);

  // leak: test rows flow into the two steps where a choice is made
  for (const i of [4, 6]) {
    if (!leak || step < i + 1) continue;
    const b = boxes[i];
    arrowLine([[heldX, b.cy - 6], [b.x + b.w, b.cy - 6]], 'crimson', true);
    drawWarning(heldX - laneW * 0.25, b.cy - 7, 9);
  }

  STEPS.forEach((s, i) => {
    const b = boxes[i], [tint, edge] = COLORS[s.color], current = i === step - 1;
    fill(tint);
    stroke(current ? 'black' : edge);
    strokeWeight(current ? 3 : inBox(b) ? 2.5 : 1.5);
    if (i === 4) {                      // decision: a box with pointed ends
      beginShape();
      for (const [px, py] of [[b.x + 8, b.y], [b.x + b.w - 8, b.y], [b.x + b.w, b.cy], [b.x + b.w - 8, b.y + b.h], [b.x + 8, b.y + b.h], [b.x, b.cy]]) vertex(px, py);
      endShape(CLOSE);
    } else rect(b.x, b.y, b.w, b.h, i === 0 || i === 9 ? b.h / 2 : 6);
    noStroke();
    fill('black');
    textAlign(CENTER, CENTER);
    textStyle(current ? BOLD : NORMAL);
    textSize(ts);
    textLeading(ts + 1);
    text(s.label, b.cx + 2, b.cy);
    textStyle(NORMAL);
    // numbered badge: it turns into a green check mark when the step is done
    const done = i < step - 1, bx = b.x + 1, by = b.y + 1;
    fill(done ? 'forestgreen' : current ? 'black' : 'white');
    stroke(done ? 'forestgreen' : current ? 'black' : edge);
    strokeWeight(1);
    circle(bx, by, narrow ? 15 : 17);
    if (done) {
      stroke('white');
      strokeWeight(2);
      line(bx - 4, by, bx - 1, by + 3);
      line(bx - 1, by + 3, bx + 4, by - 3);
    } else {
      noStroke();
      fill(current ? 'white' : 'black');
      textSize(11);
      text(i + 1, bx, by + 1);
    }
  });
}

// The step's description, what it computes on this data set, and the scores known so far
function drawDetail(narrow, leak) {
  const s = STEPS[step - 1], ts = narrow ? 12 : 15, x = detail.x + 10, w = detail.w - 20;
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(detail.x, detail.y, detail.w, detail.h, 10);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text('Step ' + step + ': ' + s.name, x, detail.y + 7);
  textStyle(NORMAL);
  textSize(ts);
  textLeading(ts * 1.25);
  textWrap(WORD);
  const rows = narrow ? ['Degree', 'Train R²', 'CV mean', 'Test R²'] : ['Degree', 'Train R²', 'CV mean', 'CV std', 'Test R²'];
  const rh = narrow ? 14 : 22, tableY = narrow ? detail.y + detail.h - rows.length * rh - 5 : detail.y + 268;
  const y = detail.y + (narrow ? 26 : 34), bodyY = y + (textWidth(s.hover) > w ? 2 : 1) * ts * 1.25 + (narrow ? 1 : 10);
  text(s.hover, x, y, w, bodyY - y);
  fill(leak && [5, 6, 7, 8, 10].includes(step) ? 'firebrick' : 'black');
  text(stepText(leak), x, bodyY, w, tableY - bodyY - 2);

  // table: a score appears at the step that computes it
  const pick = leak ? bestTest : bestCV, labelW = narrow ? 62 : 78, cw = (w - labelW) / MAX_DEG, small = narrow ? 11 : 13;
  textSize(small);
  if (step >= 7) {
    fill(leak ? 'mistyrose' : 'palegreen');
    rect(x + labelW + (pick - 1) * cw, tableY, cw, rows.length * rh, 4);
  }
  rows.forEach((label, r) => {
    const mid = tableY + (r + 0.5) * rh;
    noStroke();
    fill('dimgray');
    textAlign(LEFT, CENTER);
    textStyle(NORMAL);
    text(label, x, mid);
    for (let d = 1; d <= MAX_DEG; d++) {
      const cx = x + labelW + (d - 0.5) * cw, fitted = step >= 6 || (d === 1 && step >= 3), scored = step >= 6 || (d === 1 && step >= 4);
      const peeked = leak && (step >= 6 || (d === 1 && step >= 5));
      let cell = '–';
      if (label === 'Degree') cell = String(d);
      else if (label === 'Train R²' && fitted) cell = f3(trainR2[d]);
      else if (label === 'CV mean' && scored) cell = f3(cvMean[d]);
      else if (label === 'CV std' && scored) cell = f3(cvStd[d]);
      else if (label === 'Test R²' && (peeked || (step >= 8 && d === pick))) cell = f3(testR2[d]);
      else if (label === 'Test R²') cell = '';
      noStroke();
      fill(label === 'Test R²' && leak ? 'crimson' : 'black');
      textAlign(CENTER, CENTER);
      textStyle(label === 'Degree' ? BOLD : NORMAL);
      text(cell, cx, mid);
      if (cell === '') drawLock(cx, mid, narrow ? 4.5 : 5.5, false, 'gray');
    }
  });
  textStyle(NORMAL);
  if (narrow) return;
  // legend for the lines in the flowchart
  let ly = tableY + rows.length * rh + 22;
  for (const [col, dashed, label] of [['gray', true, 'test rows, held out until step 8'], ['darkgoldenrod', false, 'loop back to try another model'],
    ['crimson', true, 'leak: test rows used in a choice']]) {
    stroke(col);
    strokeWeight(2);
    if (dashed) drawingContext.setLineDash([5, 4]);
    line(x, ly, x + 26, ly);
    drawingContext.setLineDash([]);
    noStroke();
    fill('dimgray');
    textAlign(LEFT, CENTER);
    text(label, x + 34, ly);
    ly += 19;
  }
  text('Click any step to jump to it.', x, ly + 4);
}

function mousePressed() {
  const i = boxes.findIndex(b => inBox(b));
  if (i >= 0) setStep(i + 1);
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
