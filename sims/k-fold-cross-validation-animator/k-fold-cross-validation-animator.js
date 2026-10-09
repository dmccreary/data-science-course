// K-Fold Cross-Validation Animator
// CANVAS_HEIGHT: 620
// Bloom L2-L3 (Understand, Apply): students step through the K rounds of cross-validation, see
// which rows are held out in each round, and use the fold scores to judge how far a single
// train-test split could be from the average. Rounds advance with Next Fold; Start plays them.
//
// Model: 50 synthetic houses, price ($1000s) = 50 + 15 * size (100 sq ft) + noise, from a seeded
// generator. The rows are cut into K folds the way scikit-learn's KFold does: contiguous blocks
// in row order, with the first n % K folds one row larger. In round r a least-squares line is fit
// on every row outside fold r and scored on fold r with R^2 = 1 - SS_res / SS_tot, where SS_tot
// uses the mean of the test fold (the r2_score definition). The summary reports the mean, the
// standard deviation with ddof = 0 (what cv_scores.std() returns), the minimum, and the maximum.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 540;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const N = 50, DATA_SEED = 24, STEP_FRAMES = 70;
let rngState = 1;
let xs = [], ys = [];        // size and price of each row, in file order
let order = [];              // order[j] = row at position j (file order until Shuffle Rows is pressed)
let foldOf = [];             // foldOf[j] = fold that holds position j
let rounds = [];             // per fold: { b0, b1, r2, train: [rows], test: [rows] }
let K = 5, step = 1, viewRound = 0, running = false, shuffleCount = 0, lastStepFrame = 0;
let mat = null;              // geometry of the fold picture, used for clicks
let kSelect, nextButton, startButton, resetButton, shuffleButton;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }

function makeData() {
  rngState = DATA_SEED;
  for (let i = 0; i < N; i++) {
    xs[i] = Math.round((8 + 22 * uniform()) * 10) / 10;
    ys[i] = Math.round(50 + 15 * xs[i] + 55 * stdNormal());
  }
}

// Least-squares line through the given rows: returns [intercept, slope]
function fitLine(rows) {
  const n = rows.length;
  const mx = rows.reduce((s, r) => s + xs[r], 0) / n, my = rows.reduce((s, r) => s + ys[r], 0) / n;
  let sxy = 0, sxx = 0;
  for (const r of rows) { sxy += (xs[r] - mx) * (ys[r] - my); sxx += (xs[r] - mx) ** 2; }
  return [my - sxy / sxx * mx, sxy / sxx];
}

// R^2 of the line y = b0 + b1 x, measured on the given rows
function rSquared(rows, b0, b1) {
  const my = rows.reduce((s, r) => s + ys[r], 0) / rows.length;
  let ssRes = 0, ssTot = 0;
  for (const r of rows) { ssRes += (ys[r] - (b0 + b1 * xs[r])) ** 2; ssTot += (ys[r] - my) ** 2; }
  return 1 - ssRes / ssTot;
}

// Assign rows to folds and run all K rounds. The picture reveals them one at a time.
function runCrossValidation() {
  order = Array.from({ length: N }, (_, i) => i);
  if (shuffleCount > 0) {                     // seeded Fisher-Yates shuffle, like KFold(shuffle=True)
    rngState = 1000 + shuffleCount;
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(uniform() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
  }
  foldOf = [];
  for (let f = 0; f < K; f++) {
    const size = Math.floor(N / K) + (f < N % K ? 1 : 0);
    for (let i = 0; i < size; i++) foldOf.push(f);
  }
  rounds = [];
  for (let f = 0; f < K; f++) {
    const test = order.filter((_, j) => foldOf[j] === f), train = order.filter((_, j) => foldOf[j] !== f);
    const [b0, b1] = fitLine(train);
    rounds.push({ b0, b1, r2: rSquared(test, b0, b1), train, test });
  }
}

// Mean, standard deviation (ddof = 0), minimum, and maximum of the scores
function summarize(scores) {
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const sd = Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / scores.length);
  return { mean, sd, min: Math.min(...scores), max: Math.max(...scores) };
}

function goTo(newStep, keepRunning) {
  step = newStep;
  viewRound = step - 1;
  running = !!keepRunning && step < K;
  lastStepFrame = frameCount;
  startButton.html(running ? 'Pause' : 'Start');
  if (step >= K) nextButton.attribute('disabled', ''); else nextButton.removeAttribute('disabled');
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  const makeButton = (label, x, y, action) => {
    const b = createButton(label);
    b.parent(mainElement);
    b.position(x, drawHeight + y);
    b.mousePressed(action);
    return b;
  };
  nextButton = makeButton('Next Fold', 10, 45, () => { if (step < K) goTo(step + 1, false); });
  startButton = makeButton('Start', 98, 45, () => goTo(running ? step : (step >= K ? 1 : step), !running));
  resetButton = makeButton('Reset', 158, 45, () => { shuffleCount = 0; runCrossValidation(); goTo(1, false); });
  shuffleButton = makeButton('Shuffle Rows', 250, 8, () => { shuffleCount++; runCrossValidation(); });

  const row = createDiv();
  row.parent(mainElement);
  row.position(10, drawHeight + 8);
  row.style('font-size', '16px');
  createSpan('Number of folds (K): ').parent(row);
  kSelect = createSelect();
  kSelect.parent(row);
  ['3', '5', '10'].forEach(o => kSelect.option(o));
  kSelect.selected('5');
  kSelect.style('font-size', '15px');
  kSelect.changed(() => { K = int(kSelect.value()); runCrossValidation(); goTo(1, false); });

  makeData();
  runCrossValidation();
  goTo(1, false);

  describe('A grid with one row per round of K-fold cross-validation. In each row fifty data rows are colored blue for the ' +
    'held-out test fold and green for the training folds, with the R squared score of that round beside it. A scatter plot ' +
    'shows the line fit in the selected round, and a panel reports the mean, standard deviation, minimum, and maximum score.', LABEL);
}

function draw() {
  updateCanvasSize();
  if (running && frameCount - lastStepFrame >= STEP_FRAMES) goTo(step + 1, true);

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600;
  const w = canvasWidth - 2 * margin;
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('K-Fold Cross-Validation', canvasWidth / 2, 8);
  fill('dimgray');
  textSize(narrow ? 12 : 14);
  const cv = shuffleCount ? 'KFold(' + K + ', shuffle=True, random_state=' + shuffleCount + ')' : K;
  text((narrow ? 'cv=' + cv : 'cross_val_score(model, X, y, cv=' + cv + ", scoring='r2')") +
    (shuffleCount ? '' : '   (rows in file order)'), canvasWidth / 2, narrow ? 33 : 38);

  const top = narrow ? 68 : 78, areaH = narrow ? 165 : 200;
  drawFolds(margin, top, w, areaH, narrow);
  const lowerY = top + areaH + 26;
  if (narrow) {
    drawFit(margin, lowerY, w, 150, narrow);
    drawResults(margin, lowerY + 156, w, drawHeight - lowerY - 164, narrow);
  } else {
    const fitW = w * 0.52;
    drawFit(margin, lowerY, fitW, drawHeight - lowerY - 8, narrow);
    drawResults(margin + fitW + 10, lowerY, w - fitW - 10, drawHeight - lowerY - 8, narrow);
  }

  // control label
  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Round ' + step + ' of ' + K, 225, drawHeight + 57);
}

// One row per round: the 50 data rows colored by their role in that round, then the round's score
function drawFolds(x0, top, w, areaH, narrow) {
  const labelW = narrow ? 52 : 74, scoreW = narrow ? 100 : 205, gap = narrow ? 2 : 4;
  const stripX = x0 + labelW, stripW = w - labelW - scoreW;
  const cw = (stripW - (K - 1) * gap) / N;
  const rowH = min(62, areaH / K), ch = rowH - (narrow ? 4 : 6);
  const posX = j => stripX + j * cw + foldOf[j] * gap;
  const scoreX = stripX + stripW + 10, barW = narrow ? 44 : 110;
  const ts = narrow ? 11 : 13;
  mat = { top, rowH, x0, x1: x0 + w };

  // column headers
  noStroke();
  fill('black');
  textSize(ts);
  textAlign(CENTER, BOTTOM);
  for (let f = 0; f < K; f++) {
    const first = foldOf.indexOf(f), size = foldOf.lastIndexOf(f) - first + 1;
    text((narrow ? '' : 'Fold ') + (f + 1), posX(first) + size * cw / 2, top - 3);
  }
  textAlign(RIGHT, BOTTOM);
  if (narrow) text('Fold', stripX - 6, top - 3);
  textAlign(LEFT, BOTTOM);
  text(narrow ? 'Test R²' : 'Test score (R²)', scoreX, top - 3);

  for (let r = 0; r < K; r++) {
    const y = top + r * rowH, done = r < step;
    if (r === viewRound) {                    // highlight the round shown in the scatter plot
      fill('lemonchiffon');
      stroke('goldenrod');
      strokeWeight(1.5);
      rect(x0 - 5, y - 3, w + 10, ch + 6, 5);
    }
    noStroke();
    fill(done ? 'black' : 'gray');
    textSize(ts);
    textAlign(LEFT, CENTER);
    text('Round ' + (r + 1), x0, y + ch / 2);
    for (let j = 0; j < N; j++) {
      const isTest = foldOf[j] === r;
      fill(isTest ? color(65, 105, 225, done ? 255 : 70) : color(46, 139, 87, done ? 255 : 70));
      rect(posX(j), y, cw - (narrow ? 0.7 : 1.2), ch);
    }
    // score bar on a 0 to 1 scale (a negative R² shows as an empty bar)
    fill('white');
    stroke('silver');
    strokeWeight(1);
    rect(scoreX, y + ch / 2 - 6, barW, 12);
    noStroke();
    if (done) {
      fill('royalblue');
      rect(scoreX, y + ch / 2 - 6, barW * constrain(rounds[r].r2, 0, 1), 12);
      fill('black');
      text((narrow ? '' : 'R² = ') + nf(rounds[r].r2, 1, 2), scoreX + barW + 6, y + ch / 2);
    } else {
      fill('gray');
      text(narrow ? '?' : 'not run yet', scoreX + barW + 6, y + ch / 2);
    }
  }

  // mean of the rounds run so far
  const s = summarize(rounds.slice(0, step).map(r => r.r2));
  const mx = scoreX + barW * constrain(s.mean, 0, 1), bottom = top + K * rowH;
  stroke('crimson');
  strokeWeight(2);
  drawingContext.setLineDash([5, 3]);
  line(mx, top, mx, bottom);
  drawingContext.setLineDash([]);
  noStroke();
  fill('crimson');
  textAlign(CENTER, TOP);
  text('mean ' + nf(s.mean, 1, 2), constrain(mx, scoreX + 30, x0 + w - 30), bottom + 1);

  // legend
  const ly = top + areaH + 13;
  textAlign(LEFT, CENTER);
  fill('royalblue');
  rect(stripX, ly - 6, 12, 12);
  fill('black');
  text(narrow ? 'Test fold' : 'Test fold (held out, then scored)', stripX + 17, ly);
  const lx = stripX + (narrow ? 80 : 240);
  fill('seagreen');
  rect(lx, ly - 6, 12, 12);
  fill('black');
  text(narrow ? 'Training folds' : 'Training folds (used to fit the line)', lx + 17, ly);
}

// Scatter plot of the round being viewed: training rows, test rows, and the line fit on the training rows
function drawFit(x0, y0, w, h, narrow) {
  const rd = rounds[viewRound], ts = narrow ? 12 : 14;
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text('Round ' + (viewRound + 1) + ': fit on ' + rd.train.length + ' rows, test on ' + rd.test.length, x0 + 10, y0 + 7);
  textStyle(NORMAL);

  const px = x0 + (narrow ? 44 : 56), py = y0 + 30, pw = w - (narrow ? 56 : 72), ph = h - (narrow ? 62 : 70);
  const yMax = Math.ceil(Math.max(...ys) / 100) * 100;
  const gx = v => px + (v - 5) / 28 * pw, gy = v => py + ph - v / yMax * ph;
  textSize(narrow ? 11 : 12);
  for (let v = 0; v <= yMax; v += 200) {
    stroke('gainsboro');
    line(px, gy(v), px + pw, gy(v));
    noStroke();
    fill('dimgray');
    textAlign(RIGHT, CENTER);
    text(v, px - 4, gy(v));
  }
  for (let v = 10; v <= 30; v += 10) {
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    text(v, gx(v), py + ph + 3);
  }
  stroke('gray');
  strokeWeight(1.5);
  line(px, py, px, py + ph);
  line(px, py + ph, px + pw, py + ph);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  text('size (100 sq ft)', px + pw / 2, py + ph + 16);
  push();
  translate(x0 + 11, py + ph / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, CENTER);
  text('price ($1000s)', 0, 0);
  pop();

  // training rows, the fitted line, then the test rows with their errors
  noStroke();
  fill(46, 139, 87, 190);
  for (const r of rd.train) circle(gx(xs[r]), gy(ys[r]), narrow ? 6 : 7);
  stroke('black');
  strokeWeight(2);
  line(gx(6), gy(rd.b0 + rd.b1 * 6), gx(32), gy(rd.b0 + rd.b1 * 32));
  for (const r of rd.test) {
    stroke('royalblue');
    strokeWeight(1.5);
    line(gx(xs[r]), gy(ys[r]), gx(xs[r]), gy(rd.b0 + rd.b1 * xs[r]));
    stroke('white');
    strokeWeight(1);
    fill('royalblue');
    square(gx(xs[r]) - 4.5, gy(ys[r]) - 4.5, 9);
  }
  noStroke();
  textAlign(LEFT, TOP);
  textSize(ts);
  fill('black');
  text('price = ' + nf(rd.b0, 1, 1) + ' + ' + nf(rd.b1, 1, 2) + ' × size', px + 8, py + 2);
  fill('mediumblue');
  text('R² on the blue rows = ' + nf(rd.r2, 1, 2), px + 8, py + ts + 6);
}

// Summary of the fold scores revealed so far
function drawResults(x0, y0, w, h, narrow) {
  const ts = narrow ? 12 : 15, lh = ts + 6;
  const s = summarize(rounds.slice(0, step).map(r => r.r2));
  const tested = rounds.slice(0, step).reduce((a, r) => a + r.test.length, 0);
  fill(step === K ? 'honeydew' : 'lightyellow');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text('Scores after ' + step + ' of ' + K + ' rounds', x0 + 10, y0 + 7);
  textStyle(NORMAL);
  textSize(ts);
  let y = y0 + 9 + lh;
  fill('crimson');
  text('Mean R² = ' + nf(s.mean, 1, 2), x0 + 10, y);
  fill('black');
  if (narrow) {
    text('Std dev ' + nf(s.sd, 1, 2) + '   Min ' + nf(s.min, 1, 2) + '   Max ' + nf(s.max, 1, 2), x0 + 130, y);
  } else {
    y += lh;
    text('Std dev = ' + nf(s.sd, 1, 2) + '    Min = ' + nf(s.min, 1, 2) + '    Max = ' + nf(s.max, 1, 2), x0 + 10, y);
  }
  y += lh + 2;
  let msg;
  if (step < K) {
    msg = tested + ' of the 50 rows have been test rows so far. Round 1 alone is an ordinary train-test split: it reports R² = ' +
      nf(rounds[0].r2, 1, 2) + '. Press Next Fold to hold out a different fold.';
  } else {
    msg = 'Every row has now been a test row exactly once. A single train-test split would have reported just one of these ' +
      'scores, anywhere from ' + nf(s.min, 1, 2) + ' to ' + nf(s.max, 1, 2) + '. The mean of all ' + K + ' is a steadier estimate.';
    if (s.sd > 0.15) msg += ' With only ' + rounds[K - 1].test.length + ' test rows per fold, the scores vary widely.';
  }
  textWrap(WORD);
  text(msg, x0 + 10, y, w - 20, y0 + h - y - 4);
}

// Click a finished round to see its fit in the scatter plot
function mousePressed() {
  if (!mat || mouseX < mat.x0 || mouseX > mat.x1) return;
  const r = Math.floor((mouseY - mat.top) / mat.rowH);
  if (r >= 0 && r < step) viewRound = r;
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
