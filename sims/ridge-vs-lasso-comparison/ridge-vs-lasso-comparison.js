// Ridge vs Lasso Comparison
// CANVAS_HEIGHT: 600
// Bloom L4-L5 (Analyze, Evaluate): students move one λ for both methods, compare how the two
// coefficient paths behave, and judge which penalty suits each data set.
//
// Model: 50 rows and 8 features from a seeded generator. Features and target are standardized
// (mean 0, SD 1, ddof = 0), so there is no intercept to fit. Both methods use the chapter's form
//   Ridge: minimize RSS + λ Σ β_j²      solved exactly from (XᵀX + λI) β = Xᵀy
//   Lasso: minimize RSS + λ Σ |β_j|     solved by coordinate descent with soft-thresholding,
//                                        run until no coefficient changes by more than 1e-12
// scikit-learn equivalents on the same standardized data: Ridge(alpha=λ) and Lasso(alpha=λ/(2n)),
// because scikit-learn's Lasso minimizes RSS/(2n) + alpha Σ|β_j|. Here n = 50, so alpha = λ/100.
// Paths: both models are fit at 101 values of λ from 0.1 to 10,000 (every 0.05 in log10 λ).
// Cross-validation: 5 folds of 10 consecutive rows. Each fold's model is fit on the other 40 rows,
// centered with their own means, and scored by squared error on the held-out rows. As in
// scikit-learn, Ridge keeps λ and Lasso keeps alpha = λ/(2n) fixed across folds.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 520;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let sliderLeftMargin = 110;
let defaultTextSize = 16;

const N = 50, P = 8, FOLDS = 5, GRID_N = 101;
const lambdaAt = g => Math.pow(10, -1 + 0.05 * g);
const FEATURE_COLORS = ['crimson', 'forestgreen', 'mediumpurple', 'saddlebrown', 'deeppink', 'teal', 'goldenrod', 'slategray'];
const RIDGE = 'steelblue', LASSO = 'darkorange';
// row(z): one row of raw features and the target, built from independent standard normals z()
const DATASETS = [
  { name: 'Housing', seed: 90, names: ['sq ft', 'beds', 'baths', 'age', 'lot', 'miles', 'door #', 'day'],
    about: 'House price from 8 features. Sq ft, beds, and baths rise and fall together. Beds, door #, and listing day have no effect of their own.',
    row: z => {
      const size = z(), x = [size + 0.35 * z(), 0.8 * size + 0.6 * z(), 0.8 * size + 0.6 * z(), z(), 0.3 * size + z(), z(), z(), z()];
      return [x, 0.55 * x[0] + 0.15 * x[2] - 0.25 * x[3] + 0.15 * x[4] - 0.3 * x[5] + 0.8 * z()];
    } },
  { name: 'Synthetic (sparse)', seed: 32, names: ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'],
    about: 'Eight unrelated features. Only x1, x2, and x3 affect y. The other five are pure noise.',
    row: z => {
      const x = Array.from({ length: P }, () => z());
      return [x, x[0] - 0.7 * x[1] + 0.5 * x[2] + 0.8 * z()];
    } },
  { name: 'Medical', seed: 5, names: ['weight', 'BMI', 'waist', 'age', 'salt', 'active', 'sleep', 'coffee'],
    about: 'Blood pressure from 8 features. Weight, BMI, and waist measure nearly the same thing, and every feature has some real effect.',
    row: z => {
      const body = z(), x = [body + 0.35 * z(), body + 0.35 * z(), body + 0.35 * z(), z(), z(), -0.3 * body + z(), z(), z()];
      return [x, 0.2 * (x[0] + x[1] + x[2]) + 0.4 * x[3] + 0.2 * x[4] - 0.2 * x[5] - 0.1 * x[6] + 0.05 * x[7] + 0.8 * z()];
    } }
];

let rngState = 1;
let cache = [];                    // fitted paths for each data set, built on first use
let lambdaSlider, dataSelect, cvCheckbox;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }

// Solve A x = b by Gaussian elimination with partial pivoting
function solve(A, b) {
  const n = b.length, M = A.map((row, i) => row.concat(b[i]));
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

// Lasso by coordinate descent on G = XᵀX and c = Xᵀy. One pass updates every coefficient:
//   β_j = S(c_j − Σ_(k≠j) G_jk β_k, λ/2) / G_jj,   S(a, t) = sign(a) max(|a| − t, 0)
// S returns exactly 0 whenever |a| ≤ t, which is how Lasso removes a feature.
function lassoFit(G, c, lam, start) {
  const b = start.slice();
  for (let pass = 0; pass < 50000; pass++) {
    let change = 0;
    for (let j = 0; j < b.length; j++) {
      let a = c[j];
      for (let k = 0; k < b.length; k++) if (k !== j) a -= G[j][k] * b[k];
      const next = Math.abs(a) <= lam / 2 ? 0 : (a - Math.sign(a) * lam / 2) / G[j][j];
      change = Math.max(change, Math.abs(next - b[j]));
      b[j] = next;
    }
    if (change < 1e-12) break;
  }
  return b;
}

// Ridge and Lasso fits at every λ of the grid, using only the given rows (centered by their own means)
function fitPaths(X, y, rows) {
  const xm = X[0].map((_, j) => rows.reduce((s, i) => s + X[i][j], 0) / rows.length);
  const ym = rows.reduce((s, i) => s + y[i], 0) / rows.length;
  const G = xm.map((_, j) => xm.map((_, k) => rows.reduce((s, i) => s + (X[i][j] - xm[j]) * (X[i][k] - xm[k]), 0)));
  const c = xm.map((_, j) => rows.reduce((s, i) => s + (X[i][j] - xm[j]) * (y[i] - ym), 0));
  const ridge = [], lasso = [];
  let b = new Array(P).fill(0);
  for (let g = GRID_N - 1; g >= 0; g--) {      // largest λ first: each Lasso fit starts from the previous one
    const lam = lambdaAt(g);
    ridge[g] = solve(G.map((row, j) => row.map((v, k) => v + (j === k ? lam : 0))), c);
    lasso[g] = b = lassoFit(G, c, lam * rows.length / N, b);
  }
  return { ridge, lasso, xm, ym, ols: solve(G, c) };
}

function buildDataset(index) {
  rngState = DATASETS[index].seed;
  const raw = Array.from({ length: N }, () => DATASETS[index].row(stdNormal));
  const standardize = col => {
    const m = col.reduce((s, v) => s + v, 0) / N, sd = Math.sqrt(col.reduce((s, v) => s + (v - m) ** 2, 0) / N);
    return col.map(v => (v - m) / sd);
  };
  const cols = Array.from({ length: P }, (_, j) => standardize(raw.map(r => r[0][j])));
  const X = raw.map((_, i) => cols.map(col => col[i])), y = standardize(raw.map(r => r[1]));
  const all = X.map((_, i) => i), d = fitPaths(X, y, all);
  // cross-validated mean squared error at every λ
  d.cvRidge = new Array(GRID_N).fill(0);
  d.cvLasso = new Array(GRID_N).fill(0);
  for (let f = 0; f < FOLDS; f++) {
    const held = all.filter(i => Math.floor(i * FOLDS / N) === f), fold = fitPaths(X, y, all.filter(i => !held.includes(i)));
    for (let g = 0; g < GRID_N; g++) {
      for (const [path, out] of [[fold.ridge, d.cvRidge], [fold.lasso, d.cvLasso]]) {
        for (const i of held) out[g] += (y[i] - fold.ym - X[i].reduce((s, v, j) => s + (v - fold.xm[j]) * path[g][j], 0)) ** 2 / N;
      }
    }
  }
  const argmin = a => a.indexOf(Math.min(...a));
  d.bestRidge = argmin(d.cvRidge);
  d.bestLasso = argmin(d.cvLasso);
  const every = d.ridge.concat(d.lasso).flat().concat(d.ols, 0), span = Math.max(...every) - Math.min(...every);
  d.lo = Math.min(...every) - 0.08 * span;
  d.hi = Math.max(...every) + 0.08 * span;
  d.X = X; d.y = y;
  return d;
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  lambdaSlider = createSlider(0, GRID_N - 1, 40, 1);
  lambdaSlider.parent(mainElement);
  lambdaSlider.position(sliderLeftMargin, drawHeight + 8);
  lambdaSlider.size(canvasWidth - sliderLeftMargin - margin);

  dataSelect = createSelect();
  dataSelect.parent(mainElement);
  dataSelect.position(10, drawHeight + 45);
  DATASETS.forEach(d => dataSelect.option(d.name));
  dataSelect.style('font-size', '15px');

  cvCheckbox = createCheckbox(' Show cross-validation best λ', false);
  cvCheckbox.parent(mainElement);
  cvCheckbox.position(170, drawHeight + 46);
  cvCheckbox.style('font-size', '16px');

  describe('Two coefficient path plots, ridge on the left and lasso on the right, show eight coefficients against ' +
    'lambda on a log scale. A slider sets lambda for both. A bar chart compares the ridge and lasso coefficients ' +
    'at that lambda, and a panel reports how many coefficients lasso has set to zero and how far ridge has shrunk ' +
    'the largest one. A menu chooses one of three data sets and a checkbox marks the lambda chosen by cross-validation.', LABEL);
}

const signed = (v, digits) => (Math.abs(v) < 0.5 * Math.pow(10, -digits) ? 0 : v).toFixed(digits).replace('-', '−');
const showLambda = lam => lam >= 100 ? Math.round(lam).toLocaleString('en-US') : lam >= 10 ? lam.toFixed(1) : lam.toFixed(2);

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600;
  const index = DATASETS.findIndex(s => s.name === dataSelect.value());
  if (!cache[index]) cache[index] = buildDataset(index);
  const d = cache[index], sel = lambdaSlider.value(), w = canvasWidth - 2 * margin;

  noStroke();
  fill('black');
  textStyle(NORMAL);
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Ridge vs Lasso Comparison', canvasWidth / 2, 8);
  fill('dimgray');
  textSize(narrow ? 11 : 13);
  textWrap(WORD);
  text('Both minimize RSS + λ × penalty on standardized data. In scikit-learn: Ridge(alpha=λ), Lasso(alpha=λ/(2n)), n = 50.',
    margin, narrow ? 31 : 38, w, 30);

  const top = narrow ? 62 : 60, pathH = narrow ? 150 : 190, pathW = (w - 8) / 2;
  drawPath(margin, top, pathW, pathH, 'Ridge (L2 penalty)', RIDGE, d.ridge, sel, d.bestRidge, d, narrow);
  drawPath(margin + pathW + 8, top, pathW, pathH, 'Lasso (L1 penalty)', LASSO, d.lasso, sel, d.bestLasso, d, narrow);
  const y2 = top + pathH + 8, h2 = drawHeight - 8 - y2;
  if (narrow) {
    drawBars(margin, y2, w, 138, d, index, sel, narrow);
    drawInsights(margin, y2 + 144, w, h2 - 144, d, index, sel, narrow);
  } else {
    drawBars(margin, y2, w * 0.56, h2, d, index, sel, narrow);
    drawInsights(margin + w * 0.56 + 8, y2, w * 0.44 - 8, h2, d, index, sel, narrow);
  }

  // control label
  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('λ = ' + showLambda(lambdaAt(sel)), 10, drawHeight + 18);
}

function drawPanelBox(x, y, w, h, title, ts, col) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  noStroke();
  fill(col || 'black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text(title, x + 10, y + 7);
  textStyle(NORMAL);
  textSize(ts);
}

// Round tick values (1, 2, or 5 times a power of ten) between lo and hi
function ticks(lo, hi, n) {
  const raw = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map(m => m * p).find(s => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

// Coefficient values and gridlines shared by the three charts
function drawValueAxis(px, py, pw, ph, d) {
  const gy = v => py + ph - (v - d.lo) / (d.hi - d.lo) * ph;
  for (const v of ticks(d.lo, d.hi, 5)) {
    stroke(v === 0 ? 'gray' : 'gainsboro');
    strokeWeight(1);
    line(px, gy(v), px + pw, gy(v));
    noStroke();
    fill('dimgray');
    textAlign(RIGHT, CENTER);
    text(signed(v, 1), px - 4, gy(v));
  }
  return gy;
}

// One coefficient path: every coefficient against λ, with markers at the chosen λ
function drawPath(x0, y0, w, h, title, col, path, sel, cvBest, d, narrow) {
  const ts = narrow ? 11 : 12;
  drawPanelBox(x0, y0, w, h, title, ts + 1, col);
  const px = x0 + (narrow ? 32 : 38), py = y0 + 28, pw = x0 + w - 12 - px, ph = y0 + h - (narrow ? 30 : 34) - py;
  const gx = g => px + g / (GRID_N - 1) * pw;
  textSize(ts);
  const gy = drawValueAxis(px, py, pw, ph, d);
  textAlign(CENTER, TOP);
  for (let e = -1; e <= 4; e++) {
    if (narrow && e % 2 === 0) continue;
    text(['0.1', '1', '10', '100', '1000', '10k'][e + 1], gx((e + 1) * 20), py + ph + 3);
  }
  fill('black');
  text('λ (log scale)', px + pw / 2, py + ph + (narrow ? 15 : 17));

  noFill();
  strokeWeight(2);
  for (let j = 0; j < P; j++) {
    stroke(FEATURE_COLORS[j]);
    beginShape();
    for (let g = 0; g < GRID_N; g++) vertex(gx(g), gy(path[g][j]));
    endShape();
  }
  if (cvCheckbox.checked()) {
    stroke('green');
    strokeWeight(2);
    drawingContext.setLineDash([3, 3]);
    line(gx(cvBest), py, gx(cvBest), py + ph);
    drawingContext.setLineDash([]);
    noStroke();
    fill('green');
    textAlign(gx(cvBest) > px + pw / 2 ? RIGHT : LEFT, TOP);
    text('CV best', gx(cvBest) + (gx(cvBest) > px + pw / 2 ? -4 : 4), py + 1);
  }
  stroke('black');
  strokeWeight(1.5);
  drawingContext.setLineDash([5, 4]);
  line(gx(sel), py, gx(sel), py + ph);
  drawingContext.setLineDash([]);
  for (let j = 0; j < P; j++) {       // a coefficient that is exactly 0 gets a hollow marker
    const zero = path[sel][j] === 0;
    stroke(zero ? 'gray' : 'white');
    strokeWeight(1);
    fill(zero ? 'white' : FEATURE_COLORS[j]);
    circle(gx(sel), gy(path[sel][j]), narrow ? 7 : 9);
  }
}

// Ridge and Lasso coefficients side by side at the chosen λ. The black line is the unpenalized value.
function drawBars(x0, y0, w, h, d, index, sel, narrow) {
  const ts = narrow ? 11 : 12, names = DATASETS[index].names;
  drawPanelBox(x0, y0, w, h, 'Coefficients at λ = ' + showLambda(lambdaAt(sel)), ts + 1);
  // legend
  let lx = x0 + w - 10;
  textAlign(RIGHT, CENTER);
  for (const [label, col] of [['no penalty', 'black'], ['Lasso', LASSO], ['Ridge', RIDGE]]) {
    noStroke();
    fill('black');
    text(label, lx, y0 + 15);
    lx -= textWidth(label) + 16;
    fill(col);
    if (col === 'black') rect(lx, y0 + 14, 12, 2); else rect(lx, y0 + 9, 12, 12);
    lx -= narrow ? 6 : 10;
  }
  const px = x0 + (narrow ? 32 : 38), py = y0 + 30, pw = x0 + w - 12 - px, ph = y0 + h - 24 - py;
  const gy = drawValueAxis(px, py, pw, ph, d), group = pw / P, bar = Math.min(group * 0.34, 20);
  for (let j = 0; j < P; j++) {
    const cx = px + (j + 0.5) * group, r = d.ridge[sel][j], l = d.lasso[sel][j];
    noStroke();
    fill(RIDGE);
    rect(cx - bar, Math.min(gy(0), gy(r)), bar, Math.abs(gy(r) - gy(0)));
    fill(LASSO);
    rect(cx, Math.min(gy(0), gy(l)), bar, Math.abs(gy(l) - gy(0)));
    stroke('black');
    strokeWeight(2);
    line(cx - bar - 2, gy(d.ols[j]), cx + bar + 2, gy(d.ols[j]));
    if (l === 0) {                      // removed by Lasso
      stroke('gray');
      strokeWeight(1);
      fill('white');
      circle(cx + bar / 2, gy(0), narrow ? 7 : 9);
    }
    noStroke();
    fill(FEATURE_COLORS[j]);
    circle(cx - textWidth(names[j]) / 2 - 3, py + ph + 12, 7);
    fill(l === 0 ? 'gray' : 'black');
    textAlign(CENTER, CENTER);
    text(names[j], cx + 4, py + ph + 12);
  }
}

// What the two methods have done at the chosen λ, computed from the fits
function drawInsights(x0, y0, w, h, d, index, sel, narrow) {
  const ts = narrow ? 11 : 13, lh = ts + 4, ix = x0 + 10, iw = w - 20, names = DATASETS[index].names;
  drawPanelBox(x0, y0, w, h, DATASETS[index].name + ' data', ts);
  const big = d.ols.reduce((best, v, j) => Math.abs(v) > Math.abs(d.ols[best]) ? j : best, 0);
  const shrunk = b => Math.round(100 * (1 - b[big] / d.ols[big]));
  const gone = names.filter((_, j) => d.lasso[sel][j] === 0), ridgeZeros = d.ridge[sel].filter(v => v === 0).length;
  const paragraphs = [[DATASETS[index].about, 'dimgray'],
    ['Ridge: ' + (ridgeZeros ? ridgeZeros : 'none') + ' of the 8 coefficients is exactly 0. The largest one (' + names[big] + ') has shrunk by ' +
      shrunk(d.ridge[sel]) + '%, from ' + signed(d.ols[big], 2) + ' to ' + signed(d.ridge[sel][big], 2) + '.', 'midnightblue'],
    ['Lasso: ' + (gone.length === 0 ? 'no coefficient is 0 yet.' : gone.length === P ? 'all 8 coefficients are exactly 0, so the model predicts the mean.'
      : gone.length + ' of 8 coefficients ' + (gone.length === 1 ? 'is' : 'are') + ' exactly 0 (' + gone.join(', ') + ').') +
      (gone.length < P ? ' The largest one (' + names[big] + ') has shrunk by ' + shrunk(d.lasso[sel]) + '%.' : ''), 'chocolate']];
  if (cvCheckbox.checked()) {
    const r = d.cvRidge[d.bestRidge], l = d.cvLasso[d.bestLasso];
    paragraphs.push(['5-fold CV: Ridge is best at λ = ' + showLambda(lambdaAt(d.bestRidge)) + ' (error ' + r.toFixed(3) + '), Lasso at λ = ' +
      showLambda(lambdaAt(d.bestLasso)) + ' (error ' + l.toFixed(3) + '). ' +
      (Math.abs(r - l) < 0.02 * Math.min(r, l) ? 'They predict about equally well here.' : (r < l ? 'Ridge' : 'Lasso') + ' predicts better on this data.'), 'darkgreen']);
  }
  let ty = y0 + 9 + lh;
  textWrap(WORD);
  textLeading(lh);
  for (const [str, col] of paragraphs) {
    const lines = Math.ceil(textWidth(str) / (iw * 0.9));
    fill(col);
    text(str, ix, ty, iw, (lines + 1) * lh);
    ty += lines * lh + (narrow ? 2 : 6);
  }
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  lambdaSlider.size(canvasWidth - sliderLeftMargin - margin);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
