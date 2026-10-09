// Feature Engineering Laboratory
// CANVAS_HEIGHT: 620
// Bloom L6 with L3 (Create, Apply): students build new features from five raw housing columns
// (a product, a ratio, a sum, a difference, a square, or a log), preview each one, add it to a
// linear regression model, and see the real change in the cross-validated R^2.
//
// Data: 300 simulated houses from a seeded generator. price (in $1000s) is built from
//   -420 + 0.02 sqft + 0.012 (sqft x school) + 0.25 (sqft / beds) + 55 ln(lot) - 5 age + 0.08 age^2 + noise (sd 25),
// so four engineered features carry information that the raw columns cannot give a linear model.
// Model: ordinary least squares with an intercept on the five raw columns plus the engineered ones.
//   Train R^2: fit on all 300 rows and scored on the same rows.
//   CV R^2: mean R^2 of 5 folds of 60 consecutive rows, each scored by a model fit to the other 240
//           (what cross_val_score(LinearRegression(), X, y, cv=5, scoring='r2').mean() returns).
//   Worth of a feature: CV R^2 of your model minus CV R^2 of the model without that feature.
//   Skew: mean((x - mean)^3) / mean((x - mean)^2)^1.5.   r: Pearson correlation with price.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 540;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const N = 300, K = 5, TOL = 0.005, MAX_NEW = 6, SEED = 2, ROWS = 8;
const VARS = ['sqft', 'beds', 'age', 'lot', 'school'];
const OPS = [
  { label: '× (product)', sym: '×', py: '*', two: true, commutes: true, f: (a, b) => a * b },
  { label: '÷ (ratio)', sym: '÷', py: '/', two: true, f: (a, b) => a / b },
  { label: '+ (sum)', sym: '+', py: '+', two: true, commutes: true, f: (a, b) => a + b },
  { label: '− (difference)', sym: '−', py: '-', two: true, f: (a, b) => a - b },
  { label: '² (square of A)', name: a => a + '²', code: a => 'df[\'' + a + '\'] ** 2', f: a => a * a },
  { label: 'log (of A)', name: a => 'log(' + a + ')', code: a => 'np.log(df[\'' + a + '\'])', f: a => Math.log(a) }
];
const TRUE = [['sqft', 0, 'school'], ['sqft', 1, 'beds'], ['age', 4, 'age'], ['lot', 5, 'lot']];   // the generating formula

let rngState = 1;
let raw = { sqft: [], beds: [], age: [], lot: [], school: [] }, price = [];
let made = [];                          // engineered features in the model: { name, col, r, worth }
let base = null, cur = null, goal = 0;  // { train, cv } of the baseline and current models, CV R^2 of the generating formula
let cand = null;                        // the feature set up in the dropdowns
let hits = [];                          // remove marks drawn in the last frame: { k, x, y, w, h }
let selA, selOp, selB, addButton, resetButton;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }
function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }

function makeData() {
  rngState = SEED;
  for (let i = 0; i < N; i++) {
    const sqft = Math.round(Math.min(4500, Math.max(600, 2100 + 600 * stdNormal())));
    const beds = Math.min(6, Math.max(1, Math.round(sqft / 650 + 0.8 * stdNormal())));
    const age = 1 + Math.floor(60 * uniform());
    const lot = 100 * Math.round(80 * Math.exp(0.8 * stdNormal()));       // right-skewed, median about 8,000 sq ft
    const school = 1 + Math.floor(10 * uniform());
    raw.sqft.push(sqft); raw.beds.push(beds); raw.age.push(age); raw.lot.push(lot); raw.school.push(school);
    price.push(-420 + 0.02 * sqft + 0.012 * sqft * school + 0.25 * sqft / beds + 55 * Math.log(lot) - 5 * age + 0.08 * age * age + 25 * stdNormal());
  }
}

// Least squares on the rows in train, returning a function that predicts any row. Each column is
// centered and scaled to unit length on the training rows, so the normal equations hold correlations.
// Gauss-Jordan elimination pivots on the largest remaining diagonal entry. A column that is an exact
// combination of the others (A + B when A and B are in the model) leaves a zero pivot and gets
// coefficient 0, which predicts exactly what LinearRegression predicts.
function fitModel(cols, train) {
  const p = cols.length, my = mean(train.map(i => price[i]));
  const mu = cols.map(c => mean(train.map(i => c[i])));
  const len = cols.map((c, j) => Math.sqrt(train.reduce((s, i) => s + (c[i] - mu[j]) ** 2, 0)) || 1);
  const A = cols.map(() => new Array(p + 1).fill(0)), b = new Array(p).fill(0);
  for (const i of train) {
    const z = cols.map((c, j) => (c[i] - mu[j]) / len[j]);
    for (let r = 0; r < p; r++) {
      for (let c = 0; c < p; c++) A[r][c] += z[r] * z[c];
      A[r][p] += z[r] * (price[i] - my);
    }
  }
  const left = cols.map((_, j) => j), used = [];
  while (left.length) {
    left.sort((u, v) => A[v][v] - A[u][u]);
    const c = left.shift();
    if (A[c][c] < 1e-9) break;
    used.push(c);
    for (let r = 0; r < p; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      for (let k = 0; k <= p; k++) A[r][k] -= f * A[c][k];
    }
  }
  for (const c of used) b[c] = A[c][p] / A[c][c];
  return i => cols.reduce((s, col, j) => s + b[j] * (col[i] - mu[j]) / len[j], my);
}

function r2(predict, rows) {
  const my = mean(rows.map(i => price[i]));
  let sse = 0, sst = 0;
  for (const i of rows) { sse += (price[i] - predict(i)) ** 2; sst += (price[i] - my) ** 2; }
  return 1 - sse / sst;
}

// Train R^2 and 5-fold CV R^2 of the model with the five raw columns plus the extra columns
function scoreModel(extra) {
  const cols = VARS.map(v => raw[v]).concat(extra), all = price.map((_, i) => i), size = N / K;
  let cv = 0;
  for (let k = 0; k < K; k++) {
    cv += r2(fitModel(cols, all.filter(i => Math.floor(i / size) !== k)), all.filter(i => Math.floor(i / size) === k)) / K;
  }
  return { train: r2(fitModel(cols, all), all), cv };
}

function pearson(a, b) {
  const ma = mean(a), mb = mean(b);
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
  return saa > 0 ? sab / Math.sqrt(saa * sbb) : 0;
}
function skewness(a) {
  const m = mean(a), m2 = mean(a.map(v => (v - m) ** 2)), m3 = mean(a.map(v => (v - m) ** 3));
  return m2 > 0 ? m3 / m2 ** 1.5 : 0;
}

// The feature "a op b". A x A becomes A squared, and A x B is the same feature as B x A.
function build(a, op, b) {
  if (op.sym === '×' && a === b) op = OPS[4];
  if (op.commutes && VARS.indexOf(a) > VARS.indexOf(b)) [a, b] = [b, a];
  return {
    a, b: op.two ? b : null,
    name: op.two ? a + ' ' + op.sym + ' ' + b : op.name(a),
    code: op.two ? 'df[\'' + a + '\'] ' + op.py + ' df[\'' + b + '\']' : op.code(a),
    col: raw[a].map((v, i) => op.f(v, raw[b][i])),
    constant: !!op.two && a === b && !op.commutes          // A / A and A - A are the same for every house
  };
}

function refreshModel() {
  cur = scoreModel(made.map(m => m.col));
  for (const m of made) m.worth = cur.cv - scoreModel(made.filter(o => o !== m).map(o => o.col)).cv;
  refreshCandidate();
}

// Score the model as it would be with the candidate added, and decide whether it can be added
function refreshCandidate() {
  const op = OPS.find(o => o.label === selOp.value());
  cand = build(selA.value(), op, selB.value());
  cand.r = pearson(cand.col, price);
  cand.dup = made.some(m => m.name === cand.name);
  cand.after = scoreModel(made.map(m => m.col).concat([cand.col]));
  cand.gain = cand.after.cv - cur.cv;
  cand.redundant = cand.after.train - cur.train < 1e-9;    // nothing new for a linear model
  if (op.two) selB.removeAttribute('disabled'); else selB.attribute('disabled', '');
  if (cand.dup || cand.constant || made.length >= MAX_NEW) addButton.attribute('disabled', ''); else addButton.removeAttribute('disabled');
}

function addFeature() {
  if (cand.dup || cand.constant || made.length >= MAX_NEW) return;
  made.push({ name: cand.name, col: cand.col, r: cand.r });
  refreshModel();
}

function over(r) { return mouseX >= r.x && mouseX <= r.x + r.w && mouseY >= r.y && mouseY <= r.y + r.h; }
function fmt(v, d = 3) { return (v < 0 ? '−' : '') + Math.abs(v).toFixed(d); }
function signed(v) { return (v < -1e-9 ? '−' : '+') + Math.abs(v).toFixed(3); }
// One number format for a whole column, chosen from its largest value
function colFormat(col) {
  const top = Math.max(...col.map(Math.abs)), whole = top >= 1000 || col.every(Number.isInteger);
  return v => (top >= 1e7 ? (v / 1e6).toFixed(1) + 'M' : whole ? Math.round(v).toLocaleString('en-US')
    : top >= 100 ? v.toFixed(1) : top >= 1 ? v.toFixed(2) : String(Number(v.toPrecision(2)))).replace('-', '−');
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);
  makeData();

  const select = (x, w, options, start) => {
    const s = createSelect();
    s.parent(mainElement);
    s.position(x, drawHeight + 10);
    s.style('width', w + 'px');
    options.forEach(o => s.option(o));
    s.selected(start);
    s.changed(refreshCandidate);
    return s;
  };
  selA = select(34, 76, VARS, 'lot');
  selOp = select(116, 132, OPS.map(o => o.label), OPS[5].label);
  selB = select(278, 76, VARS, 'school');
  addButton = createButton('Add Feature');
  addButton.parent(mainElement);
  addButton.position(10, drawHeight + 45);
  addButton.mousePressed(addFeature);
  resetButton = createButton('Reset');
  resetButton.parent(mainElement);
  resetButton.position(108, drawHeight + 45);
  resetButton.mousePressed(() => { made = []; refreshModel(); });

  base = scoreModel([]);
  goal = scoreModel(TRUE.map(t => build(t[0], OPS[t[1]], t[2]).col)).cv;
  refreshModel();

  describe('A feature engineering laboratory for 300 simulated houses. Dropdowns choose two raw columns and an operation. ' +
    'A preview panel shows the new feature, its histogram, its correlation with price, and the cross-validated R squared ' +
    'the model would have if it were added. A second panel compares the baseline model with the current model, and a ' +
    'third lists the engineered features in the model with what each one is worth.', LABEL);
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
  cursor(hits.some(over) ? HAND : ARROW);
  hits = [];
  textWrap(WORD);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Feature Engineering Laboratory', canvasWidth / 2, 8);

  if (narrow) {
    drawPreview(margin, 34, w, 150, true);
    drawScores(margin, 188, w, 140, true);
    drawList(margin, 332, w, drawHeight - 338, true);
  } else {
    const lw = 310, rx = margin + lw + 10, rw = w - lw - 10;
    drawPreview(margin, 44, lw, drawHeight - 52, false);
    drawScores(rx, 44, rw, 172, false);
    drawList(rx, 224, rw, drawHeight - 232, false);
  }

  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('A:', 10, drawHeight + 21);
  text('B:', 256, drawHeight + 21);
  text('Adds  ' + cand.name, 170, drawHeight + 57);
}

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

function drawHist(col, x, y, w, h, shade, label) {
  const lo = Math.min(...col), hi = Math.max(...col), bins = new Array(12).fill(0), show = colFormat(col);
  for (const v of col) bins[Math.min(11, Math.floor((v - lo) / (hi - lo || 1) * 12))]++;
  const top = Math.max(...bins);
  stroke('white');
  strokeWeight(1);
  fill(shade);
  bins.forEach((c, k) => rect(x + k * w / 12, y + h - c / top * h, w / 12, c / top * h));
  stroke('gray');
  line(x, y + h, x + w, y + h);
  noStroke();
  fill('black');
  textSize(11);
  textAlign(LEFT, TOP);
  text(show(lo), x, y + h + 3);
  textAlign(RIGHT, TOP);
  text(show(hi), x + w, y + h + 3);
  textAlign(CENTER, BOTTOM);
  text(label, x + w / 2, y - 3);
}

// What the Add Feature button would do: the values, their distribution, and the score with the feature added
function candidateVerdict() {
  if (cand.constant) return ['firebrick', 'This is the same number for every house, so it cannot help. Choose two different columns.'];
  if (cand.dup) return ['dimgray', 'This feature is already in your model.'];
  if (cand.redundant) return ['firebrick', 'Adds nothing. A linear model can already weight these columns separately, so R² does not move.'];
  if (made.length >= MAX_NEW) return ['dimgray', 'The lab holds ' + MAX_NEW + ' engineered features. Remove one to add this.'];
  if (cand.gain > TOL) return ['darkgreen', 'Worth adding: CV R² rises by more than ' + TOL + '.'];
  return ['firebrick', cand.gain < 0 ? 'Not worth adding: CV R² goes down, although Train R² goes up.' : 'Not worth adding: CV R² rises by less than ' + TOL + '.'];
}

function drawPreview(x, y, w, h, narrow) {
  const ts = narrow ? 11 : 13, pair = cand.b && cand.b !== cand.a, verdict = candidateVerdict();
  panel(x, y, w, h, 'Preview: ' + cand.name, ts + 1);
  fill('dimgray');
  text('pandas:  ' + cand.code, x + 10, y + (narrow ? 25 : 30));
  const sk = skewness(cand.col), gainLine = 'CV R² ' + fmt(cur.cv) + ' → ' + fmt(cand.after.cv) + '  (' + signed(cand.gain) + ')';
  const rLine = 'r with price:  ' + cand.a + ' ' + fmt(pearson(raw[cand.a], price), 2) +
    (pair ? ',  ' + cand.b + ' ' + fmt(pearson(raw[cand.b], price), 2) : '') + ',  new ' + fmt(cand.r, 2);
  if (narrow) {
    drawHist(cand.col, x + w - 150, y + 58, 140, 62, 'mediumpurple', cand.name + ', skew ' + fmt(sk, 2));
    fill('black');
    textAlign(LEFT, TOP);
    text(rLine, x + 10, y + 46, w - 172, 44);
    textStyle(BOLD);
    text('If added: ' + signed(cand.gain), x + 10, y + 78);
    textStyle(NORMAL);
    fill(verdict[0]);
    text(verdict[1], x + 10, y + 94, w - 172, 56);
    return;
  }
  // first rows of the data: the columns used, the new feature, and the target
  const heads = [cand.a].concat(pair ? [cand.b] : [], [cand.name, 'price']), top = y + 54;
  const edge = pair ? [0.2, 0.4, 0.76, 1] : [0.28, 0.72, 1];       // right edge of each column: the new feature gets the widest
  const cols = [raw[cand.a]].concat(pair ? [raw[cand.b]] : [], [cand.col, price]);
  heads.forEach((hd, c) => {
    const right = x + 6 + edge[c] * (w - 20), show = colFormat(cols[c]);
    textAlign(RIGHT, TOP);
    textStyle(BOLD);
    fill(hd === cand.name ? 'rebeccapurple' : 'black');
    text(hd, right, top);
    textStyle(NORMAL);
    for (let i = 0; i < ROWS; i++) text(show(cols[c][i]), right, top + 19 + i * 17);
  });
  stroke('silver');
  line(x + 10, top + 16, x + w - 10, top + 16);
  noStroke();
  fill('dimgray');
  textAlign(LEFT, TOP);
  text('first ' + ROWS + ' of ' + N + ' houses, price in $1000s', x + 10, top + 21 + ROWS * 17);

  const hw = (w - 44) / 2, hy = top + 62 + ROWS * 17;
  drawHist(raw[cand.a], x + 12, hy, hw, 84, 'silver', cand.a + ', skew ' + fmt(skewness(raw[cand.a]), 2));
  drawHist(cand.col, x + 32 + hw, hy, hw, 84, 'mediumpurple', 'new, skew ' + fmt(sk, 2));
  fill('black');
  textAlign(LEFT, TOP);
  textSize(ts);
  text(rLine, x + 10, hy + 108, w - 20, 36);
  textStyle(BOLD);
  text('If added (purple outline on the bar)', x + 10, hy + 146);
  textStyle(NORMAL);
  text(gainLine, x + 10, hy + 164);
  fill(verdict[0]);
  text(verdict[1], x + 10, hy + 184, w - 20, 54);
}

// Baseline against the current model, with CV R^2 on a zoomed axis
function drawScores(x, y, w, h, narrow) {
  const ts = narrow ? 11 : 13, rh = narrow ? 16 : 20, top = y + (narrow ? 26 : 32), c1 = x + w - (narrow ? 86 : 150), c2 = x + w - 12;
  panel(x, y, w, h, 'Model performance (linear regression)', ts + 1);
  fill('dimgray');
  textAlign(RIGHT, TOP);
  text('Train R²', c1, top);
  text('CV R²', c2, top);
  [['Baseline: 5 raw columns', base], ['Your model: 5 raw + ' + made.length + ' engineered', cur]].forEach(([label, s], k) => {
    const ry = top + (k + 1) * rh;
    fill('black');
    textStyle(k ? BOLD : NORMAL);
    textAlign(LEFT, TOP);
    text(label, x + 10, ry);
    textAlign(RIGHT, TOP);
    text(fmt(s.train), c1, ry);
    text(fmt(s.cv), c2, ry);
  });
  textStyle(NORMAL);
  textAlign(LEFT, TOP);
  fill(cur.cv - base.cv > TOL ? 'darkgreen' : 'black');
  text('Change in CV R² from the baseline: ' + signed(cur.cv - base.cv), x + 10, top + 3 * rh + 2);

  // axis from just below the baseline to 1
  const lo = Math.floor(base.cv * 20 - 1) / 20, bx = x + 14, bw = w - 28, by = y + h - (narrow ? 34 : 44), bh = narrow ? 12 : 16;
  const u = v => bx + constrain((v - lo) / (1 - lo), 0, 1) * bw;
  fill('gainsboro');
  rect(bx, by, bw, bh);
  fill('seagreen');
  rect(bx, by, u(cur.cv) - bx, bh);
  if (!cand.dup && !cand.constant && made.length < MAX_NEW) {            // where the bar would end with the candidate added
    stroke('rebeccapurple');
    strokeWeight(2);
    noFill();
    rect(bx, by - 3, u(cand.after.cv) - bx, bh + 6);
  }
  strokeWeight(2);
  stroke('black');
  line(u(base.cv), by - 5, u(base.cv), by + bh + 5);
  stroke('goldenrod');
  line(u(goal), by - 5, u(goal), by + bh + 5);
  noStroke();
  fill('black');
  textSize(11);
  textAlign(LEFT, TOP);
  text(fmt(lo, 2), bx, by + bh + 7);
  textAlign(CENTER, TOP);
  text('baseline ' + fmt(base.cv), u(base.cv) + 12, by + bh + 7);
  text('goal ' + fmt(goal), u(goal) - 12, by + bh + 7);
  textAlign(RIGHT, TOP);
  text('1.00', bx + bw, by + bh + 7);
}

// The engineered features in the model, what each is worth, and advice on what to do next
function drawList(x, y, w, h, narrow) {
  const ts = narrow ? 11 : 13, rh = narrow ? 15 : 20, top = y + (narrow ? 26 : 32), c1 = x + w * 0.4, c2 = x + w - (narrow ? 34 : 46);
  panel(x, y, w, h, 'Your engineered features (' + made.length + ' of ' + MAX_NEW + ')', ts + 1);
  fill('dimgray');
  text('Feature', x + 10, top);
  textAlign(RIGHT, TOP);
  text('r with price', c1, top);
  text('Worth: CV R² lost if removed', c2, top);
  made.forEach((m, k) => {
    const ry = top + (k + 1) * rh, weak = m.worth < TOL;
    fill(weak ? 'firebrick' : 'black');
    textAlign(LEFT, TOP);
    text(m.name, x + 10, ry);
    textAlign(RIGHT, TOP);
    text(fmt(m.r, 2), c1, ry);
    text(signed(m.worth) + (weak ? '  low' : ''), c2, ry);
    fill('firebrick');
    textStyle(BOLD);
    text('✕', x + w - 12, ry);
    textStyle(NORMAL);
    hits.push({ k, x: x + w - 30, y: ry - 2, w: 26, h: rh });
  });
  const weak = made.filter(m => m.worth < TOL).map(m => m.name);
  let s = 'Goal: lift CV R² from ' + fmt(base.cv) + ' to the gold mark at ' + fmt(goal) + ', the score of the formula that generated these prices. ' +
    'Set up a feature below, read its preview, and add the ones you expect to help.';
  if (weak.length === 1) s = weak[0] + ' is worth less than ' + TOL + ' of CV R² in this model. Click its red ✕: a simpler model with the same score is the better model.';
  else if (weak.length) s = weak.join(', ') + ' are each worth less than ' + TOL + ' while the others are in the model. Remove one with its red ✕ and look again: ' +
    'the worth of the others can change when one leaves.';
  else if (made.length && cur.cv >= goal - TOL) s = 'Your model now scores as well as the formula that generated the prices, with ' + made.length +
    ' engineered feature' + (made.length > 1 ? 's' : '') + '. The noise in the prices is what keeps R² below 1.';
  else if (made.length) s = 'CV R² is up ' + signed(cur.cv - base.cv) + ' from the baseline and ' + fmt(goal - cur.cv) + ' below the gold mark. ' +
    'Think about which columns work together, which are skewed, and which have a curved effect.';
  fill('black');
  textAlign(LEFT, TOP);
  const my = top + (MAX_NEW + 1) * rh + 6;
  text(s, x + 10, my, w - 20, y + h - my - 4);
}

// Clicking a red X removes that feature from the model
function mousePressed() {
  const hit = hits.find(over);
  if (hit) { made.splice(hit.k, 1); refreshModel(); }
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
