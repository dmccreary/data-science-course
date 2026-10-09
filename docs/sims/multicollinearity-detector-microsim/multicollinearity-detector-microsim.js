// Multicollinearity Detector MicroSim
// CANVAS_HEIGHT: 630
// Bloom L4-L5 (Analyze, Evaluate): students raise the correlation between two features, read the
// correlation matrix and the VIF of each feature, judge whether the coefficients can be trusted,
// and test the remedy of removing one of the correlated features.
//
// Data: 60 simulated houses from a seeded generator. Rooms is built to correlate with square feet:
//   z_rooms = rho * z_sqft + sqrt(1 - rho^2) * z_other,   age is independent.
//   price ($1000s) = 60 + 0.08 sqft + 6 rooms - 0.8 age + normal noise (SD 25)
// Model: least squares solved inside the sketch (normal equations on centered columns, Gauss-Jordan).
//   VIF_j = 1 / (1 - R^2_j), where R^2_j comes from regressing feature j on the other features
//   SE(b_j) = sqrt(s^2 * [(X'X)^-1]_jj),  s^2 = SSE / (n - p - 1),  95% interval = b_j +/- t * SE
// Coefficients are reported per STEP units of each feature so that all three share one $1000s axis.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 550;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;
let sliderLeftMargin = 255;

const N = 60, NOISE_SD = 25, RUNS = 100;
const NAMES = ['Square feet', 'Rooms', 'Age'];
const MEAN = [1800, 7, 30], SD = [500, 2, 15], LOW = [600, 2, 0];
const INTERCEPT = 60, BETA = [0.08, 6, -0.8];          // the true model, price in $1000s
const STEP = [500, 2, 15], STEP_LABEL = ['+500 sq ft', '+2 rooms', '+15 years'];
const TRUE = BETA.map((b, j) => b * STEP[j]);          // true price change for one step: 40, 12, -12
const RANGE = [[400, 3300], [1, 13], [0, 75]];         // plot ranges of the scatter cells
const T95 = { 56: 2.0032, 57: 2.0025 };                // t critical values for n - p - 1 degrees of freedom
const COEF_MIN = -40, COEF_MAX = 100, VIF_MAX = 20;

let rngState = 1, seed = 18, lastKey = '';
let data = {}, an = {}, runs = null;   // current sample, its analysis, coefficients of the 100 resamples
let rhoSlider, roomsBox, sampleButton, runButton;

function uniform() {                   // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }

function makeSample(sampleSeed, rho) {
  rngState = sampleSeed;
  const d = { x: [[], [], []], y: [] };
  for (let i = 0; i < N; i++) {
    const z1 = stdNormal(), z2 = stdNormal(), z3 = stdNormal();
    const z = [z1, rho * z1 + Math.sqrt(1 - rho * rho) * z2, z3];
    const v = z.map((q, j) => Math.max(LOW[j], Math.round(MEAN[j] + SD[j] * q)));
    v.forEach((q, j) => d.x[j].push(q));
    d.y.push(INTERCEPT + v.reduce((s, q, j) => s + BETA[j] * q, 0) + NOISE_SD * stdNormal());
  }
  return d;
}

// Least squares with an intercept. The columns are centered, then [X'X | X'y | I] is reduced by
// Gauss-Jordan elimination with partial pivoting, which gives the coefficients and (X'X)^-1.
function ols(xs, y) {
  const n = y.length, p = xs.length, mean = a => a.reduce((s, v) => s + v, 0) / n;
  const mx = xs.map(mean), my = mean(y);
  const A = xs.map((cj, j) => {
    const row = xs.map((ck, k) => cj.reduce((s, v, i) => s + (v - mx[j]) * (ck[i] - mx[k]), 0));
    row.push(cj.reduce((s, v, i) => s + (v - mx[j]) * (y[i] - my), 0));
    return row.concat(xs.map((c, k) => (k === j ? 1 : 0)));
  });
  for (let c = 0; c < p; c++) {
    let piv = c;
    for (let r = c + 1; r < p; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]];
    const d = A[c][c];
    A[c] = A[c].map(v => v / d);
    for (let r = 0; r < p; r++) {
      if (r === c) continue;
      const f = A[r][c];
      A[r] = A[r].map((v, k) => v - f * A[c][k]);
    }
  }
  const b = A.map(row => row[p]), b0 = my - b.reduce((s, v, j) => s + v * mx[j], 0);
  let sse = 0, sst = 0;
  y.forEach((v, i) => {
    sse += (v - b0 - b.reduce((s, bj, j) => s + bj * xs[j][i], 0)) ** 2;
    sst += (v - my) ** 2;
  });
  const r2 = 1 - sse / sst;
  return { b0, b, sse, r2, adjR2: 1 - (1 - r2) * (n - 1) / (n - p - 1), inv: A.map(row => row.slice(p + 1)) };
}

function corr(a, b) {
  const n = a.length, ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
  return sab / Math.sqrt(saa * sbb);
}

// Fit price on the chosen features. Results are indexed by feature (0 sqft, 1 rooms, 2 age).
function analyze(d, useRooms) {
  const idx = useRooms ? [0, 1, 2] : [0, 2], p = idx.length;
  const xs = idx.map(j => d.x[j].map(v => v / STEP[j]));
  const f = ols(xs, d.y), s2 = f.sse / (N - p - 1);
  const out = { idx, b0: f.b0, r2: f.r2, adjR2: f.adjR2, coef: [], se: [], vif: [] };
  idx.forEach((j, k) => {
    out.coef[j] = f.b[k];
    out.se[j] = Math.sqrt(s2 * f.inv[k][k]);
    out.vif[j] = 1 / (1 - ols(xs.filter((c, m) => m !== k), xs[k]).r2);
  });
  return out;
}

// Draw 100 fresh samples from the same population and refit the same model to each one
function resample() {
  const rho = rhoSlider.value(), use = roomsBox.checked();
  runs = [];
  for (let k = 1; k <= RUNS; k++) runs.push(analyze(makeSample(seed * 1000 + k, rho), use).coef);
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  rhoSlider = createSlider(0, 0.99, 0.97, 0.01);
  rhoSlider.parent(mainElement);
  rhoSlider.position(sliderLeftMargin, drawHeight + 12);
  rhoSlider.size(canvasWidth - sliderLeftMargin - margin);

  roomsBox = createCheckbox(' Rooms in the model', true);
  roomsBox.parent(mainElement);
  roomsBox.position(10, drawHeight + 45);
  roomsBox.style('font-size', '16px');
  sampleButton = createButton('New Sample');
  sampleButton.parent(mainElement);
  sampleButton.position(195, drawHeight + 45);
  sampleButton.mousePressed(() => { seed++; });
  runButton = createButton('Resample ×100');
  runButton.parent(mainElement);
  runButton.position(290, drawHeight + 45);
  runButton.mousePressed(resample);

  describe('A three by three matrix of scatter plots and correlations for square feet, rooms, and age of 60 simulated ' +
    'houses, a bar chart of each feature\'s variance inflation factor with thresholds at 5 and 10, and each regression ' +
    'coefficient with its 95 percent confidence interval. A slider sets how strongly rooms follows square feet, a ' +
    'checkbox removes rooms from the model, and buttons draw a new sample or refit 100 new samples.', LABEL);
}

const fmt = (v, d) => (v < 0 ? '−' : '') + nf(Math.abs(v), 1, d);
const vifClass = v => (v > 10 ? ['crimson', 'severe'] : v > 5 ? ['darkorange', 'high'] : ['seagreen', 'OK']);

function panel(x, y, w, h, title, ts) {
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

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600, w = canvasWidth - 2 * margin, rho = rhoSlider.value();
  const key = seed + '|' + rho + '|' + roomsBox.checked();
  if (key !== lastKey) {                  // new sample, new correlation, or a different model
    data = makeSample(seed, rho);
    an = analyze(data, roomsBox.checked());
    runs = null;
    lastKey = key;
  }
  textWrap(WORD);

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Multicollinearity Detector', canvasWidth / 2, 8);
  textAlign(LEFT, CENTER);
  textSize(narrow ? 14 : defaultTextSize);
  text('Sq ft and rooms correlation: ' + nf(rho, 1, 2), 10, drawHeight + 22);

  if (narrow) {
    drawMatrix(margin, 36, w, 152, 12);
    drawVif(margin, 194, w, 100, 12);
    drawCoefs(margin, 300, w, 140, 12);
    drawVerdict(margin, 446, w, drawHeight - 454, 12);
  } else {
    const lw = 330, rx = margin + lw + 10, rw = w - lw - 10;
    drawMatrix(margin, 44, lw, 336, 14);
    drawFit(margin, 388, lw, drawHeight - 396, 14);
    drawVif(rx, 44, rw, 150, 14);
    drawCoefs(rx, 202, rw, 196, 14);
    drawVerdict(rx, 406, rw, drawHeight - 414, 14);
  }
}

// Upper triangle: correlation r of each pair. Lower triangle: the scatter plot of the same pair.
function drawMatrix(x0, y0, w, h, ts) {
  panel(x0, y0, w, h, 'Feature scatter plots and correlations', ts);
  const top = y0 + ts + 16, cw = (w - 16) / 3, ch = (y0 + h - 8 - top) / 3;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    const cx = x0 + 8 + c * cw, cy = top + r * ch, rr = corr(data.x[r], data.x[c]), strong = Math.abs(rr) > 0.7;
    stroke('silver');
    strokeWeight(1);
    fill(r === c ? 'gainsboro' : r > c ? 'white' : rr < 0 ? color(220, 20, 60, 150 * -rr) : color(65, 105, 225, 150 * rr));
    rect(cx, cy, cw, ch);
    noStroke();
    textAlign(CENTER, CENTER);
    if (r > c) {
      fill(70, 130, 180, 170);
      for (let i = 0; i < N; i++) circle(map(data.x[c][i], RANGE[c][0], RANGE[c][1], cx + 5, cx + cw - 5, true),
        map(data.x[r][i], RANGE[r][0], RANGE[r][1], cy + ch - 5, cy + 5, true), ts > 12 ? 5 : 3.5);
      continue;
    }
    fill('black');
    textStyle(r === c || strong ? BOLD : NORMAL);
    textSize(r === c ? ts + 1 : ts + 5);
    text(r === c ? NAMES[r] : 'r = ' + fmt(rr, 2), cx + cw / 2, cy + ch / 2 - (strong && r !== c ? 7 : 0));
    textStyle(NORMAL);
    textSize(Math.max(11, ts - 2));
    if (r !== c && strong) text('beyond ±0.7', cx + cw / 2, cy + ch / 2 + 12);
  }
}

// The fitted equation in the features' own units, next to the model that generated the data
function drawFit(x0, y0, w, h, ts) {
  panel(x0, y0, w, h, 'This sample: ' + N + ' houses, price in $1000s', ts);
  const term = (b, j) => (b < 0 ? ' − ' : ' + ') + nf(Math.abs(b), 1, j === 0 ? 3 : 2) + ' ' + ['sqft', 'rooms', 'age'][j];
  const fitted = 'Fitted: price = ' + nf(an.b0, 1, 1) + an.idx.map(j => term(an.coef[j] / STEP[j], j)).join('');
  const truth = 'True: price = ' + INTERCEPT + BETA.map(term).join('') + ' + noise';
  fill('black');
  text(fitted + '\n' + truth, x0 + 10, y0 + ts + 18, w - 20, h - ts - 22);
  fill('dimgray');
  textAlign(LEFT, BOTTOM);
  text('The true model is known only because the data are simulated.', x0 + 10, y0 + 30, w - 20, h - 36);
}

function drawVif(x0, y0, w, h, ts) {
  panel(x0, y0, w, h, 'Variance inflation factor: VIF = 1 / (1 − R²ⱼ)', ts);
  const labelW = ts > 12 ? 92 : 78, bx = x0 + labelW + 10, bw = w - labelW - (ts > 12 ? 124 : 110);
  const top = y0 + ts + 16, rowH = (y0 + h - ts - 10 - top) / 3, gx = v => bx + Math.min(v, VIF_MAX) / VIF_MAX * bw;
  for (let v = 0; v <= VIF_MAX; v += 5) {          // axis, with the chapter's thresholds at 5 and 10
    stroke(v === 5 ? 'darkorange' : v === 10 ? 'crimson' : 'gainsboro');
    strokeWeight(v === 5 || v === 10 ? 1.5 : 1);
    line(gx(v), top, gx(v), top + 3 * rowH);
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    textSize(ts - 1);
    text(v === VIF_MAX ? v + '+' : v, gx(v), top + 3 * rowH + 2);
  }
  textSize(ts);
  for (let j = 0; j < 3; j++) {
    const cy = top + (j + 0.5) * rowH, v = an.vif[j];
    noStroke();
    fill('black');
    textAlign(LEFT, CENTER);
    text(NAMES[j], x0 + 10, cy);
    if (v === undefined) {
      fill('gray');
      text('not in model', bx + bw + 10, cy);
      continue;
    }
    const [col, word] = vifClass(v);
    fill(col);
    rect(bx, cy - rowH * 0.32, gx(v) - bx, rowH * 0.64, 0, 3, 3, 0);
    fill('black');
    textStyle(BOLD);
    text(nf(v, 1, 1), bx + bw + 10, cy);
    const tw = textWidth(nf(v, 1, 1));
    textStyle(NORMAL);
    fill(col);
    text(word, bx + bw + 16 + tw, cy);
  }
}

function drawCoefs(x0, y0, w, h, ts) {
  const wide = ts > 12;
  panel(x0, y0, w, h, wide ? 'Price change ($1000s) with 95% confidence interval' : 'Price change ($1000s), 95% interval', ts);
  const labelW = wide ? 92 : 78, bx = x0 + labelW + 10, bw = w - labelW - (wide ? 124 : 110);
  const top = y0 + ts + 16, rowH = (y0 + h - ts - 10 - top) / 3, t = T95[N - an.idx.length - 1];
  const gx = v => bx + constrain((v - COEF_MIN) / (COEF_MAX - COEF_MIN), 0, 1) * bw;
  for (let v = COEF_MIN; v <= COEF_MAX; v += 20) {
    stroke(v === 0 ? 'gray' : 'gainsboro');
    strokeWeight(1);
    line(gx(v), top, gx(v), top + 3 * rowH);
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    textSize(ts - 1);
    if (wide || v % 40 === 0) text(fmt(v, 0), gx(v), top + 3 * rowH + 2);
  }
  textAlign(LEFT, TOP);
  text('▲ true value', x0 + 10, top + 3 * rowH + 2);
  for (let j = 0; j < 3; j++) {
    const cy = top + (j + 0.5) * rowH, c = an.coef[j];
    noStroke();
    fill('black');
    textAlign(LEFT, CENTER);
    textSize(ts);
    text(STEP_LABEL[j], x0 + 10, cy);
    if (c === undefined) {
      fill('gray');
      text('not in model', bx + bw + 10, cy);
      continue;
    }
    if (runs) {                           // one faint tick for each of the 100 refitted samples
      stroke(0, 0, 0, 40);
      strokeWeight(1);
      for (const r of runs) line(gx(r[j]), cy - rowH * 0.36, gx(r[j]), cy + rowH * 0.36);
    }
    const half = t * an.se[j], col = vifClass(an.vif[j])[0];
    stroke(col);
    strokeWeight(4);
    line(gx(c - half), cy, gx(c + half), cy);
    strokeWeight(2);
    fill('white');
    circle(gx(c), cy, 10);
    noStroke();
    fill('black');
    triangle(gx(TRUE[j]), cy + 6, gx(TRUE[j]) - 5, cy + 14, gx(TRUE[j]) + 5, cy + 14);
    text(fmt(c, 1) + ' (SE ' + nf(an.se[j], 1, 1) + ')', bx + bw + 10, cy - (runs ? 8 : 0));
    if (runs) {
      const m = runs.reduce((s, r) => s + r[j], 0) / RUNS;
      const sd = Math.sqrt(runs.reduce((s, r) => s + (r[j] - m) ** 2, 0) / (RUNS - 1));
      fill('dimgray');
      textSize(Math.max(11, ts - 2));
      text('100 fits: SD ' + nf(sd, 1, 1), bx + bw + 10, cy + 8);
    }
  }
}

function drawVerdict(x0, y0, w, h, ts) {
  const worst = Math.max(...an.idx.map(j => an.vif[j])), [col] = vifClass(worst);
  const fitLine = 'R² = ' + nf(an.r2, 1, 3) + ', adjusted R² = ' + nf(an.adjR2, 1, 3) + '.';
  const inflate = 'VIF ' + nf(worst, 1, 1) + ' makes a standard error √VIF = ' + nf(Math.sqrt(worst), 1, 1) +
    ' times what an uncorrelated feature would have. ';
  let head, body;
  if (an.idx.length < 3) {
    head = 'Rooms removed: every VIF is near 1';
    body = 'The intervals are narrow, but the square feet coefficient now absorbs the rooms effect that travels with house ' +
      'size: the higher the correlation, the further it moves from its true value of 40. ' + fitLine;
  } else if (worst > 10) {
    head = 'Severe multicollinearity: VIF above 10';
    body = inflate + 'The square feet and rooms coefficients cannot be trusted one at a time, yet the model still predicts well: ' + fitLine;
  } else if (worst > 5) {
    head = 'High multicollinearity: VIF between 5 and 10';
    body = inflate + 'The intervals for square feet and rooms are clearly wider than the interval for age. ' + fitLine;
  } else {
    head = 'No multicollinearity problem: every VIF below 5';
    body = 'Each feature carries mostly its own information, so each coefficient is estimated about as precisely as this much data allows. ' + fitLine;
  }
  panel(x0, y0, w, h, '', ts);
  fill(an.idx.length < 3 ? 'darkslateblue' : col);
  textStyle(BOLD);
  textSize(ts + 1);
  text(head, x0 + 10, y0 + 7);
  textStyle(NORMAL);
  fill('black');
  textSize(ts);
  text(body, x0 + 10, y0 + ts + 16, w - 20, h - ts - 18);
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  rhoSlider.size(canvasWidth - sliderLeftMargin - margin);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
