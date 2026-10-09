// Model Selection Dashboard
// CANVAS_HEIGHT: 715
// Bloom L4-L5 (Analyze, Evaluate): students train polynomial models of different degrees, compare
// their cross-validation scores, declare a winner, and only then see the test scores and a verdict.
//
// Model: 75 points with x uniform on 0 to 10 and y = f(x) + normal noise, from a seeded generator.
// The first 60 are training points and the last 15 are test points (an 80/20 split of independent
// draws). Polynomials of degree 1 to 10 are fit by least squares to training points only. The fit is
// built from polynomials that are orthogonal on the x values used (Forsythe's recurrence), which gives
// the same curve as LinearRegression on PolynomialFeatures but stays accurate at degree 10.
//   R^2 = 1 - sum (y - prediction)^2 / sum (y - mean y)^2, on whichever points are being scored.
//   Train R^2: fit on all 60 training points, scored on the same 60.
//   CV R^2: 5 folds of 12 consecutive training points (KFold without shuffling). Each fold is scored
//           by a model fit to the other 48. The table shows the mean and standard deviation (np.std).
//   Test R^2: the 60-point fit scored on the 15 test points. It can be negative.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 600;
let controlHeight = 115;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let sliderLeftMargin = 185;
let defaultTextSize = 16;

const N = 75, N_TRAIN = 60, K = 5, MAX_DEG = 10, GAP = 0.20;
const DATASETS = [
  { name: 'Linear', f: x => 2 + 1.5 * x, sd: 1.5 },
  { name: 'Quadratic', f: x => 0.35 * (x - 5) * (x - 5) + 2, sd: 1.5 },
  { name: 'Sine wave', f: x => 6 + 4 * Math.sin(x), sd: 1.2 },
  { name: 'Noisy line', f: x => 2 + 1.5 * x, sd: 5 }
];

let rngState = 1, seed = 6;
let xs = [], ys = [], train = [], test = [];
let model = null;                                   // predictions of every degree at one x, fit on all training points
let trainR2 = [], testR2 = [], cvMean = [], cvStd = [];     // index = polynomial degree
let added = [], declared = false, pick = 0;         // degrees in the comparison, and the declared winner
let dataBox = {}, tableBox = {}, verdictBox = {};
let dataSelect, dataButton, degreeSlider, addButton, declareButton;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }
function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }

// Least-squares polynomial fits of degree 0 to MAX_DEG to the listed rows. With t = (x - 5) / 5:
//   p_0 = 1,  p_(k+1) = (t - a_k) p_k - b_k p_(k-1),
//   a_k = <t p_k, p_k> / <p_k, p_k>,  b_k = <p_k, p_k> / <p_(k-1), p_(k-1)>,  c_k = <y, p_k> / <p_k, p_k>.
// The degree-d fit is c_0 p_0 + ... + c_d p_d. Returns a function giving all the fits at one x.
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

// R-squared of every degree on the listed rows
function r2All(predict, rows) {
  const my = mean(rows.map(i => ys[i])), sse = new Array(MAX_DEG + 1).fill(0);
  let sst = 0;
  for (const i of rows) {
    sst += (ys[i] - my) ** 2;
    predict(xs[i]).forEach((f, d) => { sse[d] += (ys[i] - f) ** 2; });
  }
  return sse.map(v => 1 - v / sst);
}

function makeData() {
  const set = DATASETS.find(s => s.name === dataSelect.value());
  rngState = seed;
  xs = []; ys = [];
  for (let i = 0; i < N; i++) {
    xs.push(10 * uniform());
    ys.push(set.f(xs[i]) + set.sd * stdNormal());
  }
  train = xs.map((_, i) => i).slice(0, N_TRAIN);
  test = xs.map((_, i) => i).slice(N_TRAIN);
  model = fitPoly(train);
  trainR2 = r2All(model, train);
  testR2 = r2All(model, test);
  const size = N_TRAIN / K, folds = [];
  for (let k = 0; k < K; k++) {             // hold out one block, fit on the other four
    const held = train.slice(k * size, (k + 1) * size), rest = train.filter(i => i < k * size || i >= (k + 1) * size);
    folds.push(r2All(fitPoly(rest), held));
  }
  for (let d = 0; d <= MAX_DEG; d++) {
    const scores = folds.map(f => f[d]);
    cvMean[d] = mean(scores);
    cvStd[d] = Math.sqrt(mean(scores.map(v => (v - cvMean[d]) ** 2)));
  }
  added = [1, MAX_DEG];                     // the comparison starts with the simplest and the most flexible model
  declared = false;
  updateControls();
}

function addModel() {
  const d = degreeSlider.value();
  if (!declared && !added.includes(d)) added.push(d);
  updateControls();
}

// The winner is the degree on the slider. Declaring unlocks the test set and ends the comparison.
function declareWinner() {
  if (declared) return;
  addModel();
  pick = degreeSlider.value();
  declared = true;
  updateControls();
}

function updateControls() {
  const d = degreeSlider.value();
  declareButton.html('Declare Degree ' + d + ' the Winner');
  const setEnabled = (b, on) => { if (on) b.removeAttribute('disabled'); else b.attribute('disabled', ''); };
  setEnabled(addButton, !declared && !added.includes(d));
  setEnabled(declareButton, !declared);
}

function fmt(v) {
  if (v < -99) return 'below −99';
  return (v < 0 ? '−' : '') + (Math.abs(v) >= 10 ? Math.abs(v).toFixed(1) : Math.abs(v).toFixed(3));
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  dataSelect = createSelect();
  dataSelect.parent(mainElement);
  dataSelect.position(85, drawHeight + 10);
  DATASETS.forEach(s => dataSelect.option(s.name));
  dataSelect.selected('Quadratic');
  dataSelect.changed(makeData);

  dataButton = createButton('New Data');
  dataButton.parent(mainElement);
  dataButton.position(190, drawHeight + 10);
  dataButton.mousePressed(() => { seed++; makeData(); });

  degreeSlider = createSlider(1, MAX_DEG, 2, 1);
  degreeSlider.parent(mainElement);
  degreeSlider.position(sliderLeftMargin, drawHeight + 43);
  degreeSlider.size(canvasWidth - sliderLeftMargin - margin);
  degreeSlider.input(updateControls);

  addButton = createButton('Add to Comparison');
  addButton.parent(mainElement);
  addButton.position(10, drawHeight + 80);
  addButton.mousePressed(addModel);

  declareButton = createButton('Declare Degree 2 the Winner');
  declareButton.parent(mainElement);
  declareButton.position(150, drawHeight + 80);
  declareButton.mousePressed(declareWinner);

  makeData();

  describe('A model selection dashboard. A scatter plot shows 60 training points and a polynomial fit whose degree is set ' +
    'by a slider. A table lists each model added to the comparison with its training R squared, a bar for its 5-fold ' +
    'cross-validation R squared with standard deviation, and a test R squared that stays locked until a winner is declared. ' +
    'A verdict panel then compares the declared winner with the model that has the best cross-validation score.', LABEL);
}

function layout(narrow) {
  const w = canvasWidth - 2 * margin;
  if (narrow) {
    dataBox = { x: margin, y: 36, w, h: 150 };
    tableBox = { x: margin, y: 192, w, h: 230 };
    verdictBox = { x: margin, y: 428, w, h: drawHeight - 436 };
  } else {
    const leftW = Math.round(w * 0.42), topH = drawHeight - 190;
    dataBox = { x: margin, y: 42, w: leftW, h: topH };
    tableBox = { x: margin + leftW + 10, y: 42, w: w - leftW - 10, h: topH };
    verdictBox = { x: margin, y: 50 + topH, w, h: 130 };
  }
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600, d = degreeSlider.value();
  layout(narrow);

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Model Selection Dashboard', canvasWidth / 2, 8);

  drawData(dataBox, d, narrow);
  drawTable(tableBox, d, narrow);
  drawVerdict(verdictBox, narrow);

  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Data set:', 10, drawHeight + 21);
  text('Polynomial degree: ' + d, 10, drawHeight + 53);
}

function panelBox(r, title, ts) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(r.x, r.y, r.w, r.h, 10);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text(title, r.x + 10, r.y + 7);
  textStyle(NORMAL);
  textSize(ts);
}

// A padlock: closed and gray while the test set is locked
function drawLock(x, y, s) {
  noFill();
  stroke('gray');
  strokeWeight(1.5);
  arc(x, y - s * 0.2, s * 0.9, s * 1.1, PI, TWO_PI);
  noStroke();
  fill('gray');
  rect(x - s * 0.7, y - s * 0.2, s * 1.4, s, 2);
}

function drawStar(x, y, r) {
  fill('gold');
  stroke('darkgoldenrod');
  strokeWeight(1);
  beginShape();
  for (let i = 0; i < 10; i++) {
    const a = -HALF_PI + i * PI / 5, rr = i % 2 === 0 ? r : r * 0.45;
    vertex(x + rr * cos(a), y + rr * sin(a));
  }
  endShape(CLOSE);
}

// Training points, the fit of the degree on the slider, and (after the declaration) the test points
function drawData(r, d, narrow) {
  const ts = narrow ? 11 : 13;
  panelBox(r, 'Degree ' + d + ' fit to the training data', ts);
  text('Train R² ' + fmt(trainR2[d]) + '    CV R² ' + fmt(cvMean[d]) + ' ± ' + fmt(cvStd[d]) +
    (declared ? '    Test R² ' + fmt(testR2[d]) : ''), r.x + 10, r.y + 26);

  const px = r.x + 32, py = r.y + 46, pw = r.w - 44, ph = r.h - 46 - (narrow ? 34 : 40);
  const yLo = Math.floor(Math.min(...ys) - 1), yHi = Math.ceil(Math.max(...ys) + 1);
  const gx = v => px + v / 10 * pw, gy = v => py + ph - (constrain(v, yLo - 50, yHi + 50) - yLo) / (yHi - yLo) * ph;
  const step = yHi - yLo > 24 ? 10 : 5;
  for (let v = Math.ceil(yLo / step) * step; v <= yHi; v += step) {
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

  push();
  drawingContext.beginPath();
  drawingContext.rect(px, py, pw, ph);
  drawingContext.clip();
  stroke('white');
  strokeWeight(0.5);
  fill(65, 105, 225, 200);
  for (const i of train) circle(gx(xs[i]), gy(ys[i]), narrow ? 6 : 8);
  fill('darkorange');
  for (const i of test) if (declared) square(gx(xs[i]) - 4, gy(ys[i]) - 4, 8);
  noFill();
  stroke('black');
  strokeWeight(2.5);
  beginShape();
  for (let i = 0; i <= 200; i++) vertex(gx(i / 20), gy(model(i / 20)[d]));
  endShape();
  pop();

  // legend: the test points are hidden until the winner is declared
  const ly = r.y + r.h - 12;
  noStroke();
  fill('royalblue');
  circle(r.x + 16, ly, 8);
  fill('darkorange');
  square(r.x + 122, ly - 4, 8);
  fill('black');
  textAlign(LEFT, CENTER);
  text('training (' + N_TRAIN + ')', r.x + 24, ly);
  text('test (' + (N - N_TRAIN) + ')' + (declared ? ' unlocked' : ' locked'), r.x + 136, ly);
  if (!declared) drawLock(r.x + (narrow ? 222 : 240), ly, 8);
}

// One row per model in the comparison, sorted by degree
function drawTable(r, d, narrow) {
  const ts = narrow ? 11 : 13, rh = narrow ? 15 : 26, x0 = r.x + 10, w = r.w - 20;
  panelBox(r, 'Model comparison: 5-fold CV on the ' + N_TRAIN + ' training points', ts);
  const cTrain = x0 + (narrow ? 46 : 72), cBar = cTrain + (narrow ? 58 : 66), cTest = x0 + w - (narrow ? 60 : 66);
  const barW = cTest - cBar - (narrow ? 92 : 102), top = r.y + (narrow ? 42 : 52);
  const sorted = [...added].sort((a, b) => a - b);
  const best = sorted.reduce((b, k) => cvMean[k] > cvMean[b] ? k : b);
  const near = k => cvMean[k] >= cvMean[best] - cvStd[best];

  fill('dimgray');
  textAlign(LEFT, BOTTOM);
  text('Model', x0 + 4, top - 3);
  text('Train R²', cTrain, top - 3);
  text('CV R² (0 to 1)', cBar, top - 3);
  text('mean ± std', cBar + barW + 6, top - 3);
  textAlign(RIGHT, BOTTOM);
  text('Test R²', x0 + w - 18, top - 3);
  stroke('silver');
  strokeWeight(1);
  line(x0, top, x0 + w, top);

  sorted.forEach((k, row) => {
    const y = top + 2 + row * rh, mid = y + rh / 2;
    if (declared ? k === pick : k === d) {              // your pick, or before that the degree on the slider
      noStroke();
      fill('lightskyblue');
      rect(x0, y, w, rh - 1, 4);
    }
    noStroke();
    fill('black');
    textAlign(LEFT, CENTER);
    text((narrow ? 'Deg ' : 'Degree ') + k, x0 + 4, mid);
    text(fmt(trainR2[k]), cTrain, mid);
    if (trainR2[k] > GAP && trainR2[k] - cvMean[k] > GAP) {       // fits the training data, does much worse in CV
      fill('darkorange');
      triangle(cTrain + textWidth('0.000') + 3, mid + 4, cTrain + textWidth('0.000') + 13, mid + 4, cTrain + textWidth('0.000') + 8, mid - 5);
    }
    // CV bar with a whisker of one standard deviation either side of the mean
    const bh = rh - (narrow ? 6 : 12), u = v => cBar + constrain(v, 0, 1) * barW;
    fill('gainsboro');
    rect(cBar, mid - bh / 2, barW, bh);
    fill(!declared ? 'steelblue' : k === best ? 'forestgreen' : near(k) ? 'goldenrod' : 'indianred');
    rect(cBar, mid - bh / 2, u(cvMean[k]) - cBar, bh);
    stroke('black');
    strokeWeight(1.5);
    line(u(cvMean[k] - cvStd[k]), mid, u(cvMean[k] + cvStd[k]), mid);
    line(u(cvMean[k] - cvStd[k]), mid - 3, u(cvMean[k] - cvStd[k]), mid + 3);
    line(u(cvMean[k] + cvStd[k]), mid - 3, u(cvMean[k] + cvStd[k]), mid + 3);
    noStroke();
    fill('black');
    text(fmt(cvMean[k]) + ' ± ' + fmt(cvStd[k]), cBar + barW + 6, mid);
    if (declared) {
      textAlign(RIGHT, CENTER);
      text(fmt(testR2[k]), x0 + w - 18, mid);
      if (k === best) drawStar(x0 + w - 8, mid, narrow ? 6 : 7);
    } else drawLock(x0 + w - 38, mid, narrow ? 6 : 8);
  });

  // legend
  let ly = r.y + r.h - (narrow ? 26 : 34);
  fill('darkorange');
  noStroke();
  triangle(x0 + 2, ly + 5, x0 + 12, ly + 5, x0 + 7, ly - 4);
  fill('black');
  textAlign(LEFT, CENTER);
  text('Train R² over ' + GAP.toFixed(2) + ' and more than ' + GAP.toFixed(2) + ' above CV mean (overfitting)', x0 + 17, ly);
  ly += narrow ? 14 : 18;
  let lx = x0;
  const items = declared ? [['best CV mean', 'forestgreen'], ['within 1 std', 'goldenrod'], ['lower', 'indianred'], ['your pick', 'lightskyblue']] :
    [['row for the degree on the slider', 'lightskyblue']];
  for (const [label, col] of items) {
    fill(col);
    rect(lx + 1, ly - 5, 11, 11);
    fill('black');
    text(label, lx + 16, ly);
    lx += textWidth(label) + 26;
  }
  if (declared) {
    drawStar(lx + 7, ly, 6);
    noStroke();
    fill('black');
    text(narrow ? 'winner' : 'CV winner', lx + 17, ly);
  }
}

// Instructions before the declaration, feedback on the pick after it
function drawVerdict(r, narrow) {
  const ts = narrow ? 12 : 14;
  panelBox(r, declared ? 'Verdict on your pick: degree ' + pick : 'Your task: choose the model that will predict new data best', ts);
  let s = 'Move the slider to train a model and press Add to Comparison for each candidate. Then put your choice on the slider ' +
    'and declare it the winner. Judge by the CV mean and its standard deviation, not by Train R², which rises with every added ' +
    'degree. The ' + (N - N_TRAIN) + ' test points stay locked until you declare, and after that the comparison is closed.';
  if (declared) {
    const sorted = [...added].sort((a, b) => a - b), d = pick;
    const best = sorted.reduce((b, k) => cvMean[k] > cvMean[b] ? k : b);
    const near = k => cvMean[k] >= cvMean[best] - cvStd[best];
    const simplest = sorted.find(near);                 // the simplest model within one std of the best
    let overall = 1;
    for (let k = 2; k <= MAX_DEG; k++) if (cvMean[k] > cvMean[overall]) overall = k;
    if (d === best) {
      s = 'Your pick has the highest CV mean in your comparison: ' + fmt(cvMean[d]) + ' ± ' + fmt(cvStd[d]) + '.';
      if (simplest < d) s += ' Degree ' + simplest + ' is simpler and within one standard deviation of it (' + fmt(cvMean[simplest]) + '), so it would also be a sound choice.';
    } else if (near(d) && d < best) {
      s = 'Degree ' + best + ' has the highest CV mean (' + fmt(cvMean[best]) + ' ± ' + fmt(cvStd[best]) + '), but your pick is within one standard ' +
        'deviation of it (' + fmt(cvMean[d]) + ') and is simpler. That is a sound choice.';
    } else {
      s = 'Degree ' + best + ' beats your pick on CV mean: ' + fmt(cvMean[best]) + ' against ' + fmt(cvMean[d]) + '.';
      if (near(d)) s += ' The gap is within one standard deviation, but your pick is the more complex model, so degree ' + best + ' is the better choice.';
      else s += ' The gap is more than one standard deviation (' + fmt(cvStd[best]) + ').';
      if (!near(d) && sorted.every(k => trainR2[k] <= trainR2[d])) s += ' Your pick has the highest Train R², but Train R² rises with every added degree, so it cannot choose a model.';
    }
    if (!added.includes(overall)) s += ' You did not try degree ' + overall + ', whose CV mean is higher still (' + fmt(cvMean[overall]) + ').';
    s += ' Test set unlocked: your pick scores R² = ' + fmt(testR2[d]) + (d === best ? '' : ' and the CV winner scores ' + fmt(testR2[best])) +
      ' on ' + (N - N_TRAIN) + ' points held out from the start.' +
      (testR2[d] < 0 ? ' A negative R² means these points were predicted worse than their own mean would predict them.' : '') +
      ' The test scores have now been seen, so they cannot be used to choose again. Press New Data for a new problem.';
  }
  textWrap(WORD);
  text(s, r.x + 10, r.y + (narrow ? 26 : 30), r.w - 20, r.h - (narrow ? 28 : 32));
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
