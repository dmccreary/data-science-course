// Scikit-learn Workflow
// CANVAS_HEIGHT: 560
// Bloom L3 (Apply): students step through the six-step scikit-learn pattern (import, prepare
// data, create, fit, predict, evaluate) on the chapter's study-hours data, choose the value to
// predict for, and switch on the most common mistake (a 1-D X) to see where the script fails.
//
// Every result is computed here the way scikit-learn computes it:
//   fit      ordinary least squares: coef_ = Sxy / Sxx, intercept_ = mean y - coef_ * mean x
//   predict  intercept_ + coef_ * x
//   score    R squared = 1 - SSE / SST   (for a regressor, score is R squared, not accuracy)
// Checked against scikit-learn 1.8.0 on the same data: coef_ = [5.71428571],
// intercept_ = 47.0357, score = 0.99704, predict for 4.5, 9, 10 hours = 72.75, 98.46, 104.18.
// The ValueError wording for a 1-D pandas Series is the message in that version's source.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 480;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let sliderLeftMargin = 190;
let defaultTextSize = 16;

const HOURS = [1, 2, 3, 4, 5, 6, 7, 8];
const SCORES = [52, 58, 65, 71, 75, 82, 87, 92];
const STEPS = [
  { label: 'IMPORT', short: 'Import', color: 'royalblue' },
  { label: 'PREPARE DATA', short: 'Data', color: 'seagreen' },
  { label: 'CREATE MODEL', short: 'Create', color: 'darkorange' },
  { label: 'FIT', short: 'Fit', color: 'purple' },
  { label: 'PREDICT', short: 'Predict', color: 'crimson' },
  { label: 'EVALUATE', short: 'Evaluate', color: 'teal' }
];

let prevButton, nextButton, mistakeBox, hoursSlider;
let cur = 0;                // index of the selected step
let model = { b0: 0, b1: 0, r2: 0 };
let hits = [];              // clickable rectangles of the last frame: { i, x, y, w, h }

// What LinearRegression().fit(X, y) and .score(X, y) compute for one feature
function fitModel(xs, ys) {
  const n = xs.length, mx = xs.reduce((s, v) => s + v, 0) / n, my = ys.reduce((s, v) => s + v, 0) / n;
  let sxy = 0, sxx = 0, sst = 0, sse = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sst += (ys[i] - my) * (ys[i] - my);
  }
  const b1 = sxy / sxx, b0 = my - b1 * mx;
  for (let i = 0; i < n; i++) sse += (ys[i] - (b0 + b1 * xs[i])) ** 2;
  return { b0, b1, r2: 1 - sse / sst };
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);
  model = fitModel(HOURS, SCORES);

  prevButton = createButton('Previous');
  prevButton.parent(mainElement);
  prevButton.position(10, drawHeight + 8);
  prevButton.mousePressed(() => { cur = max(0, cur - 1); });
  nextButton = createButton('Next');
  nextButton.parent(mainElement);
  nextButton.position(88, drawHeight + 8);
  nextButton.mousePressed(() => { cur = min(STEPS.length - 1, cur + 1); });
  mistakeBox = createCheckbox(' Mistake: single brackets on X', false);
  mistakeBox.parent(mainElement);
  mistakeBox.position(148, drawHeight + 8);
  mistakeBox.style('font-size', '16px');

  hoursSlider = createSlider(0, 12, 4.5, 0.5);
  hoursSlider.parent(mainElement);
  hoursSlider.position(sliderLeftMargin, drawHeight + 45);
  hoursSlider.size(canvasWidth - sliderLeftMargin - margin);

  describe('A six-step flowchart of the scikit-learn workflow: import, prepare data, create model, fit, predict, and ' +
    'evaluate. Below it a Python script highlights the lines of the selected step, a panel explains the step and ' +
    'shows its real result for eight students, and a small plot shows the data, the fitted line, and the prediction. ' +
    'A slider sets the study hours to predict for and a checkbox introduces the one-dimensional X mistake.', LABEL);
}

// The script, as [step index, line of code]
function codeLines(h, bad) {
  return [
    [0, 'from sklearn.linear_model import LinearRegression'],
    [1, bad ? 'X = df[\'study_hours\']' : 'X = df[[\'study_hours\']]'],
    [1, 'y = df[\'exam_scores\']'],
    [2, 'model = LinearRegression()'],
    [3, 'model.fit(X, y)'],
    [4, 'X_new = pd.DataFrame({\'study_hours\': [' + h + ']})'],
    [4, 'y_pred = model.predict(X_new)'],
    [5, 'r2 = model.score(X, y)']
  ];
}

// Explanation and result of step i. bad = X was made with single brackets.
function stepInfo(i, h, bad) {
  const b0 = nf(model.b0, 1, 4), b1 = nf(model.b1, 1, 4), yp = model.b0 + model.b1 * h;
  if (bad && i > 3) {
    return { error: true, explain: 'This line is never reached. The script stopped with an error at step 4, so there is no trained model to use.',
      results: ['Not run.', 'Fix step 2 first: X = df[[\'study_hours\']]'] };
  }
  const outside = h < HOURS[0] || h > HOURS[HOURS.length - 1];
  return [
    { explain: 'Import the model class you need. Nothing is computed yet: this line only makes the LinearRegression class available.',
      results: ['LinearRegression is scikit-learn\'s ordinary least squares model.'] },
    bad ? { error: true, explain: 'Single brackets pull out one column as a 1-D Series. scikit-learn needs a 2-D X: one row per sample and one column per feature.',
      results: ['X.shape → (8,)     1-D: fit will reject it', 'y.shape → (8,)     1-D: correct for y'] }
      : { explain: 'Split the DataFrame into the features X and the target y. Double brackets keep X two-dimensional: one row per student, one column per feature.',
        results: ['X.shape → (8, 1)     2-D: 8 rows, 1 column', 'y.shape → (8,)        1-D: 8 values'] },
    { explain: 'Create an untrained model object. Swap in another scikit-learn model class here and every later step stays the same.',
      results: ['model → LinearRegression()', 'It has no coef_ or intercept_ yet, because it has not seen any data.'] },
    bad ? { error: true, explain: 'fit checks X before any calculation, finds that it is 1-D, and raises an error. The script stops here.',
      results: ['ValueError: Expected a 2-dimensional container but got <class \'pandas.core.series.Series\'> instead. ...',
        'Fix: X = df[[\'study_hours\']]'] }
      : { explain: 'fit(X, y) trains the model. It finds the least-squares slope and intercept and stores them in the model. Learned attributes end with an underscore.',
        results: ['model.coef_ → [' + b1 + ']     the slope', 'model.intercept_ → ' + b0,
          'so  ŷ = ' + nf(model.b0, 1, 2) + ' + ' + nf(model.b1, 1, 2) + 'x   (values rounded)'] },
    { explain: 'predict applies the learned equation to new data. X_new must be 2-D with the same column as X. Move the slider to change it.',
      results: ['y_pred → [' + nf(yp, 1, 2) + ']', '= ' + b0 + ' + ' + b1 + ' × ' + h,
        outside ? 'Extrapolation: ' + h + ' hours is outside the training data (1 to 8 hours).' + (yp > 100 ? ' A score above 100 is impossible.' : '')
          : 'Inside the training data (1 to 8 hours).'], warn: outside },
    { explain: 'score predicts for X and compares the predictions with y. For a regression model it returns R², the share of the variation in y that the model explains. It is not accuracy.',
      results: ['r2 → ' + nf(model.r2, 1, 4), 'The line explains ' + nf(100 * model.r2, 1, 1) + '% of the variation in exam scores.',
        'Scored here on the same data it was trained on.'] }
  ][i];
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
  const h = hoursSlider.value(), bad = mistakeBox.checked(), info = stepInfo(cur, h, bad);
  if (cur === 0) prevButton.attribute('disabled', ''); else prevButton.removeAttribute('disabled');
  if (cur === STEPS.length - 1) nextButton.attribute('disabled', ''); else nextButton.removeAttribute('disabled');
  cursor(hits.some(r => mouseX >= r.x && mouseX <= r.x + r.w && mouseY >= r.y && mouseY <= r.y + r.h) ? HAND : ARROW);
  hits = [];
  textWrap(WORD);

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('The Scikit-learn Workflow', canvasWidth / 2, 8);

  const fitted = cur >= 3 && !bad;
  if (narrow) {
    drawFlow(margin, 38, w, 44, bad, true);
    drawCode(margin, 88, w, 160, codeLines(h, bad), bad, true);
    drawText(margin, 252, w, 96, 'Step ' + (cur + 1) + ': ' + STEPS[cur].label, STEPS[cur].color, info.explain, 'black', true);
    const half = info.error ? w : Math.round(w * 0.52);
    drawText(margin, 352, half, 120, 'Result', info.error ? 'firebrick' : 'black', info.results.join('\n'), info.error || info.warn ? 'firebrick' : 'black', true);
    if (!info.error) drawPlot(margin + half + 6, 352, w - half - 6, 120, fitted, h, true);
  } else {
    const leftW = Math.round(w * 0.52), rx = margin + leftW + 10, rw = w - leftW - 10;
    drawFlow(margin, 44, w, 56, bad, false);
    drawCode(margin, 108, leftW, 236, codeLines(h, bad), bad, false);
    drawText(margin, 350, leftW, 122, 'Step ' + (cur + 1) + ' of 6: ' + STEPS[cur].label, STEPS[cur].color, info.explain, 'black', false);
    drawText(rx, 108, rw, 138, 'Result of this step', info.error ? 'firebrick' : 'black', info.results.join('\n'), info.error || info.warn ? 'firebrick' : 'black', false);
    drawPlot(rx, 252, rw, 220, fitted, h, false, bad && cur >= 3);
  }

  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Hours for X_new: ' + nf(h, 1, 1), 10, drawHeight + 56);
}

// The six steps as a row of boxes. After a failed fit the later steps are grayed out.
function drawFlow(x0, y0, w, h, bad, narrow) {
  const gap = narrow ? 6 : 16, bw = (w - 5 * gap) / 6;
  STEPS.forEach((s, i) => {
    const x = x0 + i * (bw + gap), dead = bad && cur >= 3 && i > 3, c = color(dead ? (i === cur ? 'gray' : 'silver') : s.color);
    stroke(c);
    strokeWeight(i === cur ? 3 : 1.5);
    if (i === cur) fill(c);
    else if (i < cur) { const pale = color(s.color); pale.setAlpha(45); fill(pale); } else fill('white');
    rect(x, y0, bw, h, 8);
    noStroke();
    if (i < 5) {
      fill('dimgray');
      triangle(x + bw + gap * 0.2, y0 + h / 2 - 5, x + bw + gap * 0.2, y0 + h / 2 + 5, x + bw + gap * 0.85, y0 + h / 2);
    }
    fill(i === cur ? 'white' : dead ? 'gray' : 'black');
    textAlign(CENTER, CENTER);
    textStyle(BOLD);
    textSize(narrow ? 13 : 16);
    text(bad && cur >= 3 && i === 3 ? '4 ✗' : i + 1, x + bw / 2, y0 + h * 0.3);
    textSize(narrow ? 11 : 13);
    text(narrow ? s.short : s.label, x + bw / 2, y0 + h * 0.7);
    textStyle(NORMAL);
    hits.push({ i, x, y: y0, w: bw, h });
  });
}

// The whole script. Lines of the selected step are highlighted, later lines are gray.
function drawCode(x0, y0, w, h, lines, bad, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const ts = narrow ? 12 : 14, headH = narrow ? 22 : 30, pitch = (h - headH - 6) / lines.length;
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text('Python script', x0 + 10, y0 + (narrow ? 5 : 8));
  textStyle(NORMAL);
  fill('dimgray');
  textSize(narrow ? 11 : 12);
  textAlign(RIGHT, TOP);
  text('df: 8 students, study_hours 1 to 8', x0 + w - 10, y0 + (narrow ? 7 : 10));

  lines.forEach(([step, s], k) => {
    const y = y0 + headH + k * pitch, failed = bad && cur >= 3 && step === 3, dead = bad && cur >= 3 && step > 3;
    if (step === cur) {
      const tint = color(failed ? 'red' : STEPS[step].color);
      tint.setAlpha(40);
      noStroke();
      fill(tint);
      rect(x0 + 4, y + 1, w - 8, pitch - 2, 5);
    }
    if (k === 0 || lines[k - 1][0] !== step) {               // step number beside the first line of each step
      noStroke();
      fill(dead || step > cur ? 'silver' : STEPS[step].color);
      circle(x0 + 18, y + pitch / 2, narrow ? 15 : 19);
      fill('white');
      textAlign(CENTER, CENTER);
      textStyle(BOLD);
      textSize(narrow ? 11 : 12);
      text(step + 1, x0 + 18, y + pitch / 2 + 1);
    }
    noStroke();
    fill(failed || (bad && k === 1) ? 'red' : dead || step > cur ? 'gray' : 'black');   // the 1-D X line is red too
    textStyle(step === cur ? BOLD : NORMAL);
    textSize(ts);
    textAlign(LEFT, CENTER);
    text(s, x0 + 34, y + pitch / 2 + 1);
    if (failed) {
      textAlign(RIGHT, CENTER);
      text('✗ ValueError', x0 + w - 10, y + pitch / 2 + 1);
    }
    textStyle(NORMAL);
    hits.push({ i: step, x: x0 + 4, y, w: w - 8, h: pitch });
  });
}

// A titled panel with one block of wrapped text
function drawText(x0, y0, w, h, title, titleColor, body, bodyColor, narrow) {
  fill(bodyColor === 'firebrick' ? 'mistyrose' : 'white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const ts = narrow ? 12 : 14;
  noStroke();
  fill(titleColor);
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text(title, x0 + 10, y0 + 7);
  textStyle(NORMAL);
  fill(bodyColor);
  textSize(ts);
  text(body, x0 + 10, y0 + ts + 14, w - 20, h - ts - 18);
}

// The eight students, the fitted line once fit has run, and the prediction from step 5 on
function drawPlot(x0, y0, w, h, fitted, hours, narrow, failed) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
  const px = x0 + 36, py = y0 + (narrow ? 10 : 26), pw = w - 50, ph = h - (narrow ? 30 : 62);
  const gx = v => px + v / 12 * pw, gy = v => py + ph - (v - 40) / 90 * ph;
  noStroke();
  textSize(narrow ? 11 : 12);
  if (!narrow) {
    fill('black');
    textStyle(BOLD);
    textAlign(LEFT, TOP);
    text(fitted ? 'Data and fitted line' : failed ? 'Data (fit failed, so there is no model)' : 'Data (model not fitted yet)', x0 + 10, y0 + 7);
    textStyle(NORMAL);
    textAlign(CENTER, TOP);
    text('study hours', px + pw / 2, py + ph + 18);
  }
  for (const v of [0, 4, 8, 12]) {
    stroke('gainsboro');
    strokeWeight(1);
    line(gx(v), py, gx(v), py + ph);
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    text(v, gx(v), py + ph + 3);
  }
  for (const v of [40, 70, 100, 130]) {
    stroke('gainsboro');
    line(px, gy(v), px + pw, gy(v));
    noStroke();
    fill('dimgray');
    textAlign(RIGHT, CENTER);
    text(v, px - 4, gy(v));
  }
  stroke('gray');
  line(px, py, px, py + ph);
  line(px, py + ph, px + pw, py + ph);
  // the highest possible exam score
  stroke('darkgray');
  drawingContext.setLineDash([4, 4]);
  line(px, gy(100), px + pw, gy(100));
  drawingContext.setLineDash([]);
  if (!narrow) {
    noStroke();
    fill('dimgray');
    textAlign(LEFT, BOTTOM);
    text('highest possible score', px + 4, gy(100) - 1);
  }

  if (fitted) {
    stroke(STEPS[3].color);
    strokeWeight(2.5);
    line(gx(0), gy(model.b0), gx(12), gy(model.b0 + 12 * model.b1));
  }
  stroke('white');
  strokeWeight(1);
  fill('black');
  HOURS.forEach((x, i) => circle(gx(x), gy(SCORES[i]), narrow ? 7 : 9));
  if (fitted && cur >= 4) {
    const X = gx(hours), Y = gy(model.b0 + model.b1 * hours);
    stroke(STEPS[4].color);
    strokeWeight(1.5);
    drawingContext.setLineDash([4, 3]);
    line(X, py + ph, X, Y);
    drawingContext.setLineDash([]);
    stroke('white');
    fill(STEPS[4].color);
    quad(X, Y - 8, X + 8, Y, X, Y + 8, X - 8, Y);
    noStroke();
    textStyle(BOLD);
    // the label goes above the line on the right half of the plot and below the line on the left half
    textAlign(hours >= 6 ? RIGHT : LEFT, CENTER);
    text('ŷ = ' + nf(model.b0 + model.b1 * hours, 1, 2), X + (hours >= 6 ? -11 : 11), Y + (hours >= 6 ? -9 : 10));
    textStyle(NORMAL);
  }
}

function mousePressed() {
  for (const r of hits) {
    if (mouseX >= r.x && mouseX <= r.x + r.w && mouseY >= r.y && mouseY <= r.y + r.h) { cur = r.i; return; }
  }
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  hoursSlider.size(canvasWidth - sliderLeftMargin - margin);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
