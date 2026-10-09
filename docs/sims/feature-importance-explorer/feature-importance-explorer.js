// Feature Importance Explorer
// CANVAS_HEIGHT: 610
// Bloom L4-L5 (Analyze, Evaluate): students compare three ways of ranking the features of one
// linear model, find the features on which the rankings disagree, and judge why by switching the
// correlation between the features off and inspecting one feature at a time.
//
// Data: 300 simulated houses from a seeded generator, 200 for training and 100 for testing.
//   price ($1000s) = 40 + 0.06 square_feet + 8 bedrooms + 14 bathrooms - 1.4 age + 1.0 lot_size + noise
// Model: least squares on the training rows after standardizing each feature with its training
// mean and SD (StandardScaler), solved by Gauss-Jordan elimination on the normal equations.
//   standardized coefficient = raw coefficient * SD of the feature   (the importance of a linear model)
//   permutation importance   = test R^2 - test R^2 with that column shuffled (mean of 30 shuffles)
//   drop-column importance   = test R^2 - test R^2 of a model refitted without that column
// With correlated features these rankings differ, because other columns can stand in for one that
// is removed. VIF = 1 / (1 - R^2 of the feature regressed on the other features).

let containerWidth;
let canvasWidth = 400;
let drawHeight = 530;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const SEED = 20, N_TRAIN = 200, N_TEST = 100, REPEATS = 30, NOISE_SD = 22;
const INTERCEPT = 40, BETA = [0.06, 8, 14, -1.4, 1];
const NAMES = ['square_feet', 'bedrooms', 'bathrooms', 'age', 'lot_size'];
const UNITS = ['sq ft', 'bedroom', 'bathroom', 'year of age', '1000 sq ft of lot'];
const ALL = [0, 1, 2, 3, 4];
const METHODS = [
  { name: 'Standardized coefficient', short: 'Std. coefficient', sub: '|coefficient| × SD, in $1000s', subShort: '$1000s per SD', color: 'steelblue' },
  { name: 'Permutation importance', short: 'Permutation', sub: 'test R² lost when shuffled', subShort: 'R² lost: shuffled', color: 'mediumpurple' },
  { name: 'Drop-column importance', short: 'Drop-column', sub: 'test R² lost when removed', subShort: 'R² lost: removed', color: 'teal' }
];

let rngState = 1, permRun = 1;
let Xtr = [], T = [], yTrain = [], yTest = [];     // feature columns and prices: training rows, test rows
let full = {}, baseR2 = 0;                         // model with all five features and its test R²
let vals = [[], [], []], permSd = [], ranks = [];  // importance by method and feature, SD of the shuffles, ranks
let info = [];                                     // per feature: raw coefficient, one-feature fit, correlation, VIF
let sel = 0, hits = [];
let sortSelect, permButton, corrBox, errBox;

function uniform() {                   // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }
const meanOf = a => a.reduce((s, v) => s + v, 0) / a.length;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// Bedrooms, bathrooms, and lot size follow square feet when correlated is true, and not at all otherwise
function makeData(correlated) {
  rngState = SEED;
  const X = ALL.map(() => []), y = [], k = correlated ? 1 : 0;
  for (let i = 0; i < N_TRAIN + N_TEST; i++) {
    const zs = stdNormal(), mix = c => c * k * zs + Math.sqrt(1 - c * c * k) * stdNormal();
    const row = [Math.max(600, Math.round(1800 + 500 * zs)), clamp(Math.round(3.2 + 0.9 * mix(0.85)), 1, 6),
      clamp(Math.round(2.2 + 0.7 * mix(0.75)), 1, 4), clamp(Math.round(30 + 15 * stdNormal()), 0, 80),
      Math.max(2, Math.round(10 * (8 + 3 * mix(0.4))) / 10)];
    row.forEach((v, j) => X[j].push(v));
    y.push(INTERCEPT + row.reduce((s, v, j) => s + BETA[j] * v, 0) + NOISE_SD * stdNormal());
  }
  Xtr = X.map(c => c.slice(0, N_TRAIN));
  T = X.map(c => c.slice(N_TRAIN));
  yTrain = y.slice(0, N_TRAIN);
  yTest = y.slice(N_TRAIN);
}

// Least squares with an intercept: center the columns, then solve (X'X) b = X'y by Gauss-Jordan elimination
function ols(xs, y) {
  const n = y.length, p = xs.length, mx = xs.map(meanOf), my = meanOf(y);
  const A = xs.map((cj, j) => {
    const row = xs.map((ck, k) => cj.reduce((s, v, i) => s + (v - mx[j]) * (ck[i] - mx[k]), 0));
    row.push(cj.reduce((s, v, i) => s + (v - mx[j]) * (y[i] - my), 0));
    return row;
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
  return { b0, b, r2: 1 - sse / sst };
}

// Fit price on the listed features. Each is standardized first, so b holds standardized coefficients.
function fitModel(idx) {
  const mean = idx.map(j => meanOf(Xtr[j]));
  const sd = idx.map((j, k) => Math.sqrt(meanOf(Xtr[j].map(v => (v - mean[k]) ** 2))));
  const f = ols(idx.map((j, k) => Xtr[j].map(v => (v - mean[k]) / sd[k])), yTrain);
  return { idx, mean, sd, b0: f.b0, b: f.b };
}

// R² of a fitted model on the test rows. cols holds the five test columns, one of which may be shuffled.
function testR2(m, cols) {
  const my = meanOf(yTest);
  let sse = 0, sst = 0;
  yTest.forEach((v, i) => {
    const pred = m.idx.reduce((s, j, k) => s + m.b[k] * (cols[j][i] - m.mean[k]) / m.sd[k], m.b0);
    sse += (v - pred) ** 2;
    sst += (v - my) ** 2;
  });
  return 1 - sse / sst;
}

function corr(a, b) {
  const ma = meanOf(a), mb = meanOf(b);
  let sab = 0, saa = 0, sbb = 0;
  a.forEach((v, i) => { sab += (v - ma) * (b[i] - mb); saa += (v - ma) ** 2; sbb += (b[i] - mb) ** 2; });
  return sab / Math.sqrt(saa * sbb);
}

const rankOf = v => v.map(a => 1 + v.filter(b => b > a).length);

// Shuffle one test column 30 times and record how far the test R² falls each time
function runPermutation() {
  rngState = 1000 + permRun;
  const out = ALL.map(j => {
    const drops = [];
    for (let r = 0; r < REPEATS; r++) {
      const col = T[j].slice();
      for (let i = col.length - 1; i > 0; i--) {
        const k = Math.floor(uniform() * (i + 1));
        [col[i], col[k]] = [col[k], col[i]];
      }
      drops.push(baseR2 - testR2(full, T.map((c, m) => (m === j ? col : c))));
    }
    const m = meanOf(drops);
    return [m, Math.sqrt(meanOf(drops.map(v => (v - m) ** 2)))];
  });
  vals[1] = out.map(o => o[0]);
  permSd = out.map(o => o[1]);
  ranks = vals.map(rankOf);
}

function compute() {
  makeData(corrBox.checked());
  full = fitModel(ALL);
  baseR2 = testR2(full, T);
  vals[0] = full.b.map(Math.abs);
  vals[2] = ALL.map(j => baseR2 - testR2(fitModel(ALL.filter(k => k !== j)), T));
  info = ALL.map(j => {
    const others = ALL.filter(k => k !== j), rs = others.map(k => corr(Xtr[j], Xtr[k]));
    const best = rs.reduce((b, r, i) => (Math.abs(r) > Math.abs(rs[b]) ? i : b), 0);
    return { raw: full.b[j] / full.sd[j], alone: ols([Xtr[j]], yTrain), partner: others[best], r: rs[best],
      vif: 1 / (1 - ols(others.map(k => Xtr[k]), Xtr[j]).r2) };
  });
  permRun = 1;
  runPermutation();
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  sortSelect = createSelect();
  sortSelect.parent(mainElement);
  sortSelect.position(72, drawHeight + 10);
  METHODS.forEach(m => sortSelect.option(m.name));
  permButton = createButton('Run Permutation Test');
  permButton.parent(mainElement);
  permButton.position(265, drawHeight + 10);
  permButton.mousePressed(() => { permRun++; runPermutation(); });
  corrBox = createCheckbox(' Correlated features', true);
  errBox = createCheckbox(' Error bars', true);
  [corrBox, errBox].forEach((c, i) => {
    c.parent(mainElement);
    c.position(10 + 190 * i, drawHeight + 45);
    c.style('font-size', '16px');
  });
  corrBox.changed(compute);
  compute();

  describe('A grid of bar charts that ranks five housing features by three importance measures: standardized ' +
    'coefficient, permutation importance, and drop-column importance. Ranks on which the methods disagree are ' +
    'marked. Clicking a feature shows its scatter plot against price with two fitted lines and its numbers. ' +
    'Controls sort the features, switch the correlation between features on or off, show error bars, and rerun ' +
    'the permutation test.', LABEL);
}

const signed = (v, d) => (v < 0 ? '−' : '+') + nf(Math.abs(v), 1, d);
const num3 = v => (v < -0.0005 ? '−' : '') + nf(Math.abs(v), 1, 3);       // an R² change, to three decimals
const agrees = j => ranks[0][j] === ranks[1][j] && ranks[1][j] === ranks[2][j];

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

  const narrow = canvasWidth < 600, w = canvasWidth - 2 * margin;
  hits = [];
  textWrap(WORD);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Feature Importance Explorer', canvasWidth / 2, 8);
  textAlign(LEFT, CENTER);
  textSize(narrow ? 14 : defaultTextSize);
  text('Sort by:', 10, drawHeight + 22);

  if (narrow) {
    drawGrid(margin, 36, w, 196, true);
    drawScatter(margin, 238, w, 118, 11);
    drawFacts(margin, 362, w, drawHeight - 370, 12);
  } else {
    const sw = 320;
    drawGrid(margin, 44, w, 232, false);
    drawScatter(margin, 284, sw, drawHeight - 292, 13);
    drawFacts(margin + sw + 10, 284, w - sw - 10, drawHeight - 292, 14);
  }
}

// One row per feature, one column per method. Every column has its own scale.
function drawGrid(x0, y0, w, h, narrow) {
  const ts = narrow ? 12 : 14, small = narrow ? 11 : 12;
  panel(x0, y0, w, h, (narrow ? 'Three importance measures (test R² = ' : 'Three importance measures for one model (test R² = ') + nf(baseR2, 1, 3) + ')', ts);
  const nameW = narrow ? 76 : 112, cw = (w - nameW - 10) / 3, headY = y0 + ts + 14, headH = narrow ? 30 : 36;
  const top = headY + headH + 2, rowH = narrow ? 23 : 27, valW = narrow ? 36 : 48, badge = narrow ? 15 : 20;
  const sortBy = METHODS.findIndex(m => m.name === sortSelect.selected());
  const order = ALL.slice().sort((a, b) => vals[sortBy][b] - vals[sortBy][a]);
  const showErr = errBox.checked();

  METHODS.forEach((m, k) => {
    const cx = x0 + nameW + k * cw;
    if (k === sortBy) {
      noStroke();
      fill('lightyellow');
      rect(cx, headY - 2, cw - 4, headH + 5 * rowH + 6, 6);
    }
    noStroke();
    fill(m.color);
    textAlign(LEFT, TOP);
    textStyle(BOLD);
    textSize(ts);
    text(narrow ? m.short : m.name, cx + 4, headY);
    textStyle(NORMAL);
    fill('dimgray');
    textSize(small);
    text(narrow ? m.subShort : m.sub, cx + 4, headY + ts + 3);
  });

  order.forEach((j, r) => {
    const ry = top + r * rowH, cy = ry + rowH / 2;
    if (j === sel) {
      noFill();
      stroke('black');
      strokeWeight(1.5);
      rect(x0 + 5, ry + 1, w - 10, rowH - 2, 5);
    }
    noStroke();
    fill('black');
    textAlign(LEFT, CENTER);
    textStyle(j === sel ? BOLD : NORMAL);
    textSize(narrow ? 11 : ts);
    text(NAMES[j], x0 + 10, cy);
    textStyle(NORMAL);
    METHODS.forEach((m, k) => {
      const cx = x0 + nameW + k * cw + 4, bx = cx + badge + 4, maxW = cw - badge - valW - 16;
      const v = vals[k][j], err = k === 1 && showErr ? permSd[j] : 0;
      const scaleMax = Math.max(...ALL.map(i => vals[k][i] + (k === 1 && showErr ? permSd[i] : 0)));
      const len = maxW * Math.max(0, v) / scaleMax;
      // rank badge: orange when this feature's rank is not the same under all three methods
      noStroke();
      fill(agrees(j) ? 'gainsboro' : 'orange');
      rect(cx, cy - badge / 2, badge, badge, 4);
      fill('black');
      textAlign(CENTER, CENTER);
      textSize(small);
      text(ranks[k][j], cx + badge / 2, cy + 1);
      fill(m.color);
      rect(bx, cy - rowH * 0.3, Math.max(1, len), rowH * 0.6, 0, 3, 3, 0);
      if (err) {
        stroke('black');
        strokeWeight(1.5);
        const a = bx + maxW * Math.max(0, v - err) / scaleMax, b = bx + maxW * (v + err) / scaleMax;
        line(a, cy, b, cy);
        line(a, cy - 4, a, cy + 4);
        line(b, cy - 4, b, cy + 4);
        noStroke();
      }
      fill('black');
      textAlign(LEFT, CENTER);
      text(k === 0 ? signed(full.b[j], 1) : num3(v), bx + maxW * Math.max(0, v + err) / scaleMax + 5, cy + 1);
    });
    hits.push({ j, x: x0 + 5, y: ry, w: w - 10, h: rowH });
  });
  noStroke();
  fill('dimgray');
  textAlign(LEFT, BOTTOM);
  textSize(small);
  const split = ALL.filter(j => !agrees(j)).length + ' of 5';
  text(narrow ? 'Orange rank: the methods disagree (' + split + '). Click a row to inspect it.'
    : 'An orange rank marks a feature that the three methods rank differently (' + split + ' here). Click a row to inspect it.', x0 + 10, y0 + h - 5);
}

// Selected feature against price, with the one-feature line and the multiple-regression line
function drawScatter(x0, y0, w, h, ts) {
  const f = info[sel], xs = Xtr[sel];
  panel(x0, y0, w, h, NAMES[sel] + ' and price, ' + N_TRAIN + ' training houses', ts);
  const px = x0 + 40, py = y0 + ts + 18, pw = w - 52, ph = h - (ts + 18) - (ts > 12 ? 58 : 34);
  const x1 = Math.min(...xs), x2 = Math.max(...xs), y1 = Math.min(...yTrain), y2 = Math.max(...yTrain);
  const gx = v => px + (v - x1) / (x2 - x1) * pw, gy = v => py + ph - (v - y1) / (y2 - y1) * ph;
  stroke('gray');
  strokeWeight(1);
  noFill();
  rect(px, py, pw, ph);
  noStroke();
  fill('dimgray');
  textSize(11);
  textAlign(RIGHT, TOP);
  text('$' + Math.round(y2) + 'k', px - 3, py);
  textAlign(RIGHT, BOTTOM);
  text('$' + Math.round(y1) + 'k', px - 3, py + ph);
  textAlign(LEFT, TOP);
  text(x1, px, py + ph + 2);
  textAlign(RIGHT, TOP);
  text(x2, px + pw, py + ph + 2);
  fill(70, 130, 180, 130);
  xs.forEach((v, i) => circle(gx(v), gy(yTrain[i]), ts > 12 ? 5 : 4));

  drawingContext.save();
  drawingContext.beginPath();
  drawingContext.rect(px, py, pw, ph);
  drawingContext.clip();
  const mx = meanOf(xs), my = meanOf(yTrain);
  stroke('dimgray');                      // price on this feature alone
  strokeWeight(2);
  drawingContext.setLineDash([6, 4]);
  line(gx(x1), gy(f.alone.b0 + f.alone.b[0] * x1), gx(x2), gy(f.alone.b0 + f.alone.b[0] * x2));
  drawingContext.setLineDash([]);
  stroke('crimson');                      // partial dependence: the other four features held constant
  strokeWeight(2.5);
  line(gx(x1), gy(my + f.raw * (x1 - mx)), gx(x2), gy(my + f.raw * (x2 - mx)));
  drawingContext.restore();

  const d = sel === 0 ? 3 : 1, ly = py + ph + 18, wide = ts > 12, half = wide ? 0 : pw / 2 + 20;
  noStroke();
  textSize(ts);
  textAlign(LEFT, TOP);
  fill('dimgray');
  text('- - alone: slope ' + signed(f.alone.b[0], d), x0 + 10, ly);
  fill('crimson');
  text('— others held constant: ' + signed(f.raw, d), x0 + 10 + half, ly + (wide ? ts + 5 : 0));
}

function drawFacts(x0, y0, w, h, ts) {
  const f = info[sel], j = sel, wide = ts > 12;
  panel(x0, y0, w, h, 'Inspecting ' + NAMES[j], ts);
  const lines = [
    'Raw coefficient: ' + signed(f.raw, j === 0 ? 3 : 2) + ' ($1000s) per ' + UNITS[j] + (wide ? '. Units differ, so raw coefficients cannot be compared.' : ''),
    'Standardized: × SD ' + nf(full.sd[j], 1, j === 0 ? 0 : 2) + ' = ' + signed(full.b[j], 1) + ' per SD (rank ' + ranks[0][j] + ')',
    'Permutation importance: ' + num3(vals[1][j]) + ' ± ' + nf(permSd[j], 1, 3) + ' (rank ' + ranks[1][j] + ', run ' + permRun + ')',
    'Drop-column importance: ' + num3(vals[2][j]) + ' (rank ' + ranks[2][j] + ')',
    'Most correlated with ' + NAMES[f.partner] + ', r = ' + nf(f.r, 1, 2).replace('-', '−') + '.  VIF = ' + nf(f.vif, 1, 1)
  ];
  const note = f.vif > 2 && ranks[2][j] > ranks[0][j]
    ? 'Other features carry part of the same information, so a refitted model loses little without this one, even though the current model leans on it. With correlated features no single ranking is reliable.'
    : agrees(j) ? 'All three methods give this feature the same rank.' + (f.vif > 2 ? ' It is correlated with ' + NAMES[f.partner] +
      ', though, so its coefficient and its rank could change with a different sample.' : '')
      : 'Its rank differs between methods because the score of a correlated feature shifted around it, or because two scores are too close to separate (compare the error bars).';
  fill('black');
  text(lines.join('\n'), x0 + 10, y0 + ts + 18, w - 20, h - ts - 20);
  fill('darkslateblue');
  textAlign(LEFT, BOTTOM);
  text(note, x0 + 10, y0 + 30, w - 20, h - 36);
}

function mousePressed() {
  for (const hit of hits) {
    if (mouseX >= hit.x && mouseX <= hit.x + hit.w && mouseY >= hit.y && mouseY <= hit.y + hit.h) sel = hit.j;
  }
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
