// Multiple Regression Pipeline
// CANVAS_HEIGHT: 625
// Bloom L3-L4 (Apply, Analyze): students step through the ten stages of a multiple regression
// workflow, follow the shape of the data from stage to stage, read the real result of each stage,
// and switch on the mistake that most often goes wrong there.
//
// Every result is computed here from 500 simulated houses (seeded generator), in pipeline order:
//   engineering   log_lot = ln(lot_size), is_new = 1 if age < 5
//   split         a seeded shuffle, then 400 training rows and 100 test rows
//   preprocessing median imputation, z = (x - mean) / std and one-hot columns (first level dropped),
//                 all learned from the training rows only
//   VIF           1 / (1 - R^2) from regressing each numeric column on the other numeric columns;
//                 the column with the highest VIF is dropped while that VIF is above 10
//   selection     forward selection on mean 5-fold CV R^2 of the training rows, gain of 0.005 required
//   model         ordinary least squares with an intercept; R^2 = 1 - SSE / SST; RMSE = sqrt(mean error^2)
// price ($1000s) = 60 + 0.07 square_feet + 14 bathrooms - 0.9 age + 24 ln(lot_size / 8000) + 30 is_new
//                  + 35 if Suburb or 70 if Urban + noise (sd 22). bedrooms, rooms and style have no effect.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 580;
let controlHeight = 45;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const N = 500, N_TRAIN = 400, K = 5, TOL = 0.005, SEED = 11;
const NUM = ['square_feet', 'bedrooms', 'bathrooms', 'rooms', 'age', 'lot_size', 'log_lot', 'is_new'];
const CATS = { neighborhood: ['Rural', 'Suburb', 'Urban'], style: ['Colonial', 'Modern', 'Ranch'] };
const STAGES = [
  { name: 'Raw Data', short: 'Raw data', color: 'dimgray', code: 'df = pd.read_csv(\'housing.csv\')\ndf.info()',
    explain: 'Load the data and look before you change anything: how many rows, which columns are numbers, which are categories, and where values are missing.',
    risk: 'Skipping this look. A category stored as text or a column with gaps surfaces later as an error or, worse, as a model that is quietly wrong.' },
  { name: 'Feature Engineering', mid: 'Engineering', short: 'Engineer', color: 'royalblue', code: 'df[\'log_lot\'] = np.log(df[\'lot_size\'])\ndf[\'is_new\'] = (df[\'age\'] < 5).astype(int)',
    explain: 'Create features from what you know about houses: a log for a skewed column, a flag for a threshold. These are row-by-row formulas, so they are safe before the split.',
    risk: 'Building a feature from the target, such as price per square foot. The model is then handed the answer, and the feature cannot be computed for a house that has no price yet.' },
  { name: 'Train/Test Split', short: 'Split', color: 'purple', code: 'X_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42)',
    explain: 'Set 20% of the rows aside before anything learns from the data. The test rows are not used again until stage 9.',
    risk: 'Splitting after preprocessing. Means, medians and encoders would be computed with the test rows included, so the test score would no longer be honest.' },
  { name: 'Preprocessing', short: 'Preprocess', color: 'chocolate', code: '# num: median SimpleImputer, then StandardScaler\n# cat: OneHotEncoder(drop=\'first\')\nX_train_p = prep.fit_transform(X_train)\nX_test_p = prep.transform(X_test)',
    explain: 'Fill the gaps, put the numeric columns on one scale, and turn categories into 0/1 columns. Every statistic is learned from the training rows and then applied to both sets.',
    risk: 'Calling fit_transform on the test set, or fitting the scaler on all 500 rows. That is data leakage. Leaving categories as text makes scikit-learn raise an error.' },
  { name: 'Check Multicollinearity', mid: 'Multicollinearity', short: 'VIF check', color: 'darkgoldenrod', code: 'vif = [variance_inflation_factor(X_num.values, i)\n       for i in range(X_num.shape[1])]',
    explain: 'Compute the variance inflation factor of each numeric feature: VIF = 1 / (1 − R²), where R² comes from predicting that feature from the other features. Above 10 is severe.',
    risk: 'Ignoring a high VIF. The model may still predict well, but the coefficients of the tangled features swing with small changes in the data and cannot be interpreted.' },
  { name: 'Feature Selection', short: 'Select', color: 'teal', code: 'selected, scores = forward_selection(X_train_p, y_train)',
    explain: 'Forward selection: start with no features and keep adding the one that raises the 5-fold cross-validated R² of the training rows the most, until no addition gains 0.005.',
    risk: 'Choosing features by their test score, or keeping every column. A feature picked by peeking at the test rows makes the final score too optimistic.' },
  { name: 'Model Training', short: 'Train', color: 'seagreen', code: 'model = LinearRegression()\nmodel.fit(X_train_p[selected], y_train)',
    explain: 'Fit ordinary least squares to the selected columns of the training rows. The model is now one intercept and one coefficient per feature.',
    risk: 'Judging the model by its training R². The coefficients were fit to exactly these rows, so this is the most optimistic number you will see.' },
  { name: 'Cross-Validation', short: 'Cross-val', color: 'steelblue', code: 'cv = cross_val_score(model, X_train_p[selected],\n                     y_train, cv=5, scoring=\'r2\')',
    explain: 'Estimate performance on unseen data without touching the test set: five models, each trained on four folds of the training rows and scored on the fifth.',
    risk: 'Reading the mean without the spread. Also, the features were chosen with these same rows, so this estimate leans a little optimistic. Stage 9 is the honest check.' },
  { name: 'Final Evaluation', short: 'Evaluate', color: 'crimson', code: 'r2 = model.score(X_test_p[selected], y_test)\ny_pred = model.predict(X_test_p[selected])',
    explain: 'Score the model once on the 100 test rows it has never seen. Compare the result with the cross-validation estimate and check the predictions against the actual prices.',
    risk: 'Going back to tune the model after seeing this score. The test set then becomes part of training, and nothing is left to measure real performance.' },
  { name: 'Feature Importance', mid: 'Importance', short: 'Importance', color: 'sienna', code: 'pd.Series(model.coef_, index=selected).abs().sort_values()',
    explain: 'The numeric features were standardized, so each coefficient is the change in price for a change of one standard deviation. That makes their sizes comparable.',
    risk: 'Reading importance as cause. A large coefficient says the feature helps to predict price in this data, not that changing it would change the price.' }
];

let rngState = 1;
let D = {}, Z = {}, R = {};            // raw columns, preprocessed columns, numbers computed by the pipeline
let results = [];                      // what each stage shows
let train = [], test = [];             // row numbers of the two sets
let cur = 0, hits = [];                // selected stage, and the flow boxes drawn in the last frame
let prevButton, nextButton, riskBox;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }
function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }
function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
// Least squares of y on the listed columns (plus an intercept) over the listed rows: normal
// equations solved by Gauss-Jordan elimination with partial pivoting. b[0] is the intercept.
function fit(cols, y, rows) {
  const m = cols.length + 1, A = [];
  for (let a = 0; a < m; a++) A.push(new Array(m + 1).fill(0));
  for (const i of rows) {
    const v = [1, ...cols.map(c => c[i])];
    for (let a = 0; a < m; a++) {
      for (let b = 0; b < m; b++) A[a][b] += v[a] * v[b];
      A[a][m] += v[a] * y[i];
    }
  }
  for (let c = 0; c < m; c++) {
    let p = c;
    for (let r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < m; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      for (let k = c; k <= m; k++) A[r][k] -= f * A[c][k];
    }
  }
  return A.map((row, a) => row[m] / row[a]);
}
function predict(b, cols, i) { return cols.reduce((s, c, j) => s + b[j + 1] * c[i], b[0]); }
function rSquared(b, cols, y, rows) {
  const my = mean(rows.map(i => y[i]));
  let sse = 0, sst = 0;
  for (const i of rows) { sse += (y[i] - predict(b, cols, i)) ** 2; sst += (y[i] - my) ** 2; }
  return 1 - sse / sst;
}
// R^2 of each of the 5 folds (blocks of 80 consecutive training rows) for a model on the named columns
function cvFolds(names) {
  const cols = names.map(n => Z[n]), size = N_TRAIN / K, out = [];
  for (let k = 0; k < K; k++) {
    const held = train.slice(k * size, (k + 1) * size), rest = train.filter((_, j) => j < k * size || j >= (k + 1) * size);
    out.push(rSquared(fit(cols, D.price, rest), cols, D.price, held));
  }
  return out;
}
function vifs(names) {
  return names.map(n => {
    const others = names.filter(o => o !== n).map(o => Z[o]);
    return { label: n, v: 1 / (1 - rSquared(fit(others, Z[n], train), others, Z[n], train)) };
  });
}
function skew(a) { const m = mean(a), d = a.map(v => v - m); return mean(d.map(v => v ** 3)) / mean(d.map(v => v * v)) ** 1.5; }

function runPipeline() {
  rngState = SEED;
  for (const n of ['square_feet', 'bedrooms', 'bathrooms', 'rooms', 'age', 'lot_size', 'neighborhood', 'style', 'price']) D[n] = [];
  for (let i = 0; i < N; i++) {                                                   // 1. raw data
    const sqft = Math.round(clamp(2000 + 500 * stdNormal(), 700, 4000)), beds = clamp(Math.round(sqft / 650 + 0.7 * stdNormal()), 1, 6);
    const baths = clamp(Math.round(2 * (0.5 + sqft / 1100 + 0.5 * stdNormal())) / 2, 1, 4.5), age = Math.floor(60 * uniform());
    const lot = 100 * Math.round(80 * Math.exp(0.9 * stdNormal())), u = uniform(), hood = u < 0.25 ? 0 : u < 0.75 ? 1 : 2;
    D.square_feet.push(sqft); D.bedrooms.push(beds); D.bathrooms.push(baths); D.age.push(age);
    D.rooms.push(beds + Math.ceil(baths) + 2 + (uniform() < 0.12 ? 1 : 0));
    D.neighborhood.push(CATS.neighborhood[hood]);
    D.style.push(CATS.style[Math.floor(3 * uniform())]);
    D.price.push(60 + 0.07 * sqft + 14 * baths - 0.9 * age + 24 * Math.log(lot / 8000) + (age < 5 ? 30 : 0) + 35 * hood + 22 * stdNormal());
    D.lot_size.push(uniform() < 0.04 ? null : lot);                               // about 4% of lot sizes are missing
  }
  D.log_lot = D.lot_size.map(v => v === null ? null : Math.log(v));               // 2. engineering, row by row
  D.is_new = D.age.map(a => a < 5 ? 1 : 0);
  const order = D.price.map((_, i) => i);                                         // 3. shuffle, then split 80/20
  for (let i = N - 1; i > 0; i--) { const j = Math.floor((i + 1) * uniform()); [order[i], order[j]] = [order[j], order[i]]; }
  train = order.slice(0, N_TRAIN); test = order.slice(N_TRAIN);
  R.prep = {};
  for (const n of NUM) {                                                          // 4. impute and scale with training statistics
    const seen = train.map(i => D[n][i]).filter(v => v !== null).sort((a, b) => a - b), h = (seen.length - 1) / 2;
    const med = (seen[Math.floor(h)] + seen[Math.ceil(h)]) / 2, full = D[n].map(v => v === null ? med : v);
    const m = mean(train.map(i => full[i])), sd = Math.sqrt(mean(train.map(i => (full[i] - m) ** 2)));
    Z[n] = full.map(v => (v - m) / sd);
    R.prep[n] = { med, m, sd };
  }
  for (const c in CATS) CATS[c].slice(1).forEach(level => { Z[c + '_' + level] = D[c].map(v => v === level ? 1 : 0); });
  let numeric = NUM.slice();                                                      // 5. drop the worst column while its VIF is above 10
  R.vif = vifs(numeric); R.dropped = [];
  for (let v = R.vif; ; v = vifs(numeric)) {
    const worst = v.reduce((a, b) => b.v > a.v ? b : a);
    R.vifAfter = worst;
    if (worst.v <= 10) break;
    R.dropped.push(worst.label);
    numeric = numeric.filter(n => n !== worst.label);
  }
  R.candidates = numeric.concat(Object.keys(Z).filter(n => !NUM.includes(n)));
  R.selected = []; R.path = [];                                                   // 6. forward selection on CV R^2
  for (let score = -Infinity; ;) {
    let best = null, bestScore = score + TOL;
    for (const n of R.candidates) {
      if (R.selected.includes(n)) continue;
      const s = mean(cvFolds(R.selected.concat([n])));
      if (s > bestScore) { best = n; bestScore = s; }
    }
    if (!best) break;
    R.selected.push(best); score = bestScore;
    R.path.push({ label: best, v: score });
  }
  const cols = R.selected.map(n => Z[n]);                                         // 7 to 10. fit, cross-validate, test
  R.b = fit(cols, D.price, train);
  R.trainR2 = rSquared(R.b, cols, D.price, train);
  R.folds = cvFolds(R.selected);
  R.cvMean = mean(R.folds); R.cvStd = Math.sqrt(mean(R.folds.map(v => (v - R.cvMean) ** 2)));
  R.testR2 = rSquared(R.b, cols, D.price, test);
  R.pred = test.map(i => predict(R.b, cols, i));
  R.resid = test.map((i, k) => D.price[i] - R.pred[k]);
  R.rmse = Math.sqrt(mean(R.resid.map(e => e * e)));
}

function num(v, d) { return (v < 0 ? '−' : '') + Math.abs(v).toFixed(d); }

// For each stage: the shape of the data leaving it, and its real result as lines of text plus bars
function buildResults() {
  const p = R.prep, k = R.selected.length, nTest = N - N_TRAIN, nCols = Object.keys(Z).length, nLeft = R.candidates.length;
  const gaps = rows => rows.filter(r => D.lot_size[r] === null).length, whole = v => Math.round(v).toLocaleString('en-US');
  const known = col => col.filter(v => v !== null), priceOf = rows => num(mean(rows.map(r => D.price[r])), 1);
  const coefs = R.selected.map((n, j) => ({ label: n, v: Math.abs(R.b[j + 1]), text: num(R.b[j + 1], 1), color: R.b[j + 1] < 0 ? 'indianred' : 'steelblue' }));
  const top = Math.max(...coefs.map(o => o.v)), bias = mean(R.resid);
  return [
    { shape: N + ' × 8', lines: [
      'df.shape → (' + N + ', 9): 8 feature columns and the target, price',
      '6 numeric: square_feet, bedrooms, bathrooms, rooms, age, lot_size', '2 categorical: neighborhood and style, 3 levels each',
      'Missing: lot_size has ' + gaps(train.concat(test)) + ' empty cells, the other columns none',
      'price: mean $' + num(mean(D.price), 1) + 'k, from $' + num(Math.min(...D.price), 1) + 'k to $' + num(Math.max(...D.price), 1) + 'k'] },
    { shape: N + ' × 10', lines: [
      'Two new columns: 8 features become 10', 'is_new = 1 for ' + D.is_new.filter(v => v).length + ' of ' + N + ' houses',
      'lot_size is skewed (skew ' + num(skew(known(D.lot_size)), 2) + '). log_lot is not (skew ' + num(skew(known(D.log_lot)), 2) + ')',
      'log_lot is empty wherever lot_size is empty: fixed in stage 4'] },
    { shape: N_TRAIN + ' | ' + nTest, lines: [
      'X_train (' + N_TRAIN + ', 10)    y_train (' + N_TRAIN + ',)', 'X_test (' + nTest + ', 10)    y_test (' + nTest + ',)',
      'Mean price: $' + priceOf(train) + 'k in train, $' + priceOf(test) + 'k in test. A shuffled split gives two similar samples.'],
      bars: [{ label: 'train', v: N_TRAIN, text: N_TRAIN + ' (80%)' }, { label: 'test', v: nTest, text: nTest + ' (20%)' }], max: N_TRAIN },
    { shape: N_TRAIN + ' × ' + nCols, lines: [
      'Imputer: the training median of lot_size, ' + whole(p.lot_size.med) + ', fills ' + gaps(train) + ' train and ' + gaps(test) + ' test cells',
      'Scaler: square_feet has training mean ' + whole(p.square_feet.m) + ' and std ' + whole(p.square_feet.sd) +
        ', so 2,500 sq ft becomes ' + num((2500 - p.square_feet.m) / p.square_feet.sd, 2),
      'Encoder: neighborhood → neighborhood_Suburb, neighborhood_Urban (Rural is the baseline); style → style_Modern, style_Ranch',
      NUM.length + ' numeric + ' + (nCols - NUM.length) + ' one-hot columns: X_train_p (' + N_TRAIN + ', ' + nCols + '), X_test_p (' + nTest + ', ' + nCols + ')'] },
    { shape: N_TRAIN + ' × ' + nLeft, lines: [
      'Bars: VIF of the ' + NUM.length + ' numeric columns. Red is above 10, orange is above 5.',
      R.dropped.length ? 'Dropped ' + R.dropped.join(', ') + ': the other columns predict it almost exactly.' : 'No VIF is above 10, so nothing is dropped.',
      'Highest VIF after that: ' + R.vifAfter.label + ' ' + num(R.vifAfter.v, 1) + '.  X_train_p (' + N_TRAIN + ', ' + nLeft + ')'],
      bars: R.vif.map(o => ({ label: o.label, v: o.v, text: num(o.v, 1), color: o.v > 10 ? 'crimson' : o.v > 5 ? 'darkorange' : 'seagreen' })),
      max: Math.max(12, ...R.vif.map(o => o.v)), mark: 10, markLabel: 'limit 10' },
    { shape: N_TRAIN + ' × ' + k, bars: R.path.map(o => ({ label: o.label, v: o.v, text: num(o.v, 3) })), max: 1, lines: [
      'Kept ' + k + ' of ' + nLeft + ' columns. Bars: CV R² after each addition, in the order added.',
      'Not selected: ' + R.candidates.filter(n => !R.selected.includes(n)).join(', ')] },
    { shape: k + ' coefs', bars: coefs, max: top, lines: [
      'intercept_ = ' + num(R.b[0], 1) + '  (price is in $1000s)', 'Train R² = ' + num(R.trainR2, 3) + '.  Bars: coef_, blue raises the price, red lowers it.'] },
    { shape: 'CV ' + num(R.cvMean, 2), lines: [
      'Mean CV R² = ' + num(R.cvMean, 3) + ' (+/- ' + num(2 * R.cvStd, 3) + ', two standard deviations)',
      'Train R² was ' + num(R.trainR2, 3) + (R.trainR2 - R.cvMean < 0.03 ? ': a small gap, so little overfitting.' : ': a gap this wide is a sign of overfitting.')],
      bars: R.folds.map((v, j) => ({ label: 'fold ' + (j + 1), v, text: num(v, 3) })), max: 1, mark: R.cvMean, markLabel: 'mean' },
    { shape: 'test ' + num(R.testR2, 2), scatter: true, lines: [
      'Test R² = ' + num(R.testR2, 3) + '   (CV estimate: ' + num(R.cvMean, 3) + ')',
      'RMSE = $' + num(R.rmse, 1) + 'k, the typical size of a miss',
      'Mean residual = ' + (bias < 0 ? '−$' : '$') + Math.abs(bias).toFixed(1) + 'k' +
        (Math.abs(bias) < 2 * R.rmse / Math.sqrt(nTest) ? ': no steady over- or under-pricing' : ': the model is off in one direction on these rows')] },
    { shape: 'ranking', bars: coefs.slice().sort((a, b) => b.v - a.v), max: top, lines: [
      'Bars: size of each coefficient in $1000s, largest first. Red is negative.',
      'Numeric columns: per standard deviation. One-hot columns: the gap to the baseline level, Rural.'] }
  ];
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);
  runPipeline();
  results = buildResults();

  prevButton = createButton('Previous');
  prevButton.parent(mainElement);
  prevButton.position(10, drawHeight + 10);
  prevButton.mousePressed(() => { cur = max(0, cur - 1); });
  nextButton = createButton('Next');
  nextButton.parent(mainElement);
  nextButton.position(88, drawHeight + 10);
  nextButton.mousePressed(() => { cur = min(STAGES.length - 1, cur + 1); });
  riskBox = createCheckbox(' Show what can go wrong', true);
  riskBox.parent(mainElement);
  riskBox.position(148, drawHeight + 10);
  riskBox.style('font-size', '16px');

  describe('A ten-stage multiple regression pipeline drawn as a flowchart: raw data, feature engineering, train/test split, ' +
    'preprocessing, multicollinearity check, feature selection, model training, cross-validation, final evaluation, and ' +
    'feature importance. Each box shows the shape of the data leaving the stage. A panel explains the selected stage with ' +
    'its code and its most common mistake, and a second panel shows the real result computed from 500 simulated houses.', LABEL);
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600, w = canvasWidth - 2 * margin, res = results[cur];
  if (cur === 0) prevButton.attribute('disabled', ''); else prevButton.removeAttribute('disabled');
  if (cur === STAGES.length - 1) nextButton.attribute('disabled', ''); else nextButton.removeAttribute('disabled');
  cursor(hits.some(r => mouseX >= r.x && mouseX <= r.x + r.w && mouseY >= r.y && mouseY <= r.y + r.h) ? HAND : ARROW);
  hits = [];
  textWrap(WORD);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Multiple Regression Pipeline', canvasWidth / 2, 8);

  if (narrow) {
    drawFlow(8, 36, canvasWidth - 16, 38, true);
    drawStage(margin, 124, w, 232, true);
    drawResult(margin, 362, w, drawHeight - 370, res, true);
  } else {
    const lw = Math.round(w * 0.48);
    drawFlow(margin, 44, w, 46, false);
    drawStage(margin, 156, lw, drawHeight - 164, false);
    drawResult(margin + lw + 10, 156, w - lw - 10, drawHeight - 164, res, false);
  }
}

// Two rows of five stage boxes. Each shows the shape of the data that leaves it once it has been reached.
function drawFlow(x0, y0, w, h, narrow) {
  const gap = narrow ? 4 : 16, bw = (w - 4 * gap) / 5;
  STAGES.forEach((s, i) => {
    const x = x0 + (i % 5) * (bw + gap), y = y0 + Math.floor(i / 5) * (h + (narrow ? 6 : 10)), c = color(s.color);
    stroke(c);
    strokeWeight(i === cur ? 3 : 1.5);
    if (i === cur) fill(c); else if (i < cur) { c.setAlpha(45); fill(c); } else fill('white');
    rect(x, y, bw, h, 8);
    noStroke();
    fill('dimgray');
    if (i % 5 < 4 && !narrow) triangle(x + bw + gap * 0.2, y + h / 2 - 5, x + bw + gap * 0.2, y + h / 2 + 5, x + bw + gap * 0.85, y + h / 2);
    fill(i === cur ? 'white' : 'black');
    textAlign(CENTER, CENTER);
    textStyle(BOLD);
    textSize(narrow ? 11 : 13);
    text((i + 1) + (narrow ? ' ' + s.short : '  ' + (s.mid || s.name)), x + bw / 2, y + h * 0.3);
    textStyle(NORMAL);
    textSize(narrow ? 11 : 12);
    text(i <= cur || i < 6 ? results[i].shape : '· · ·', x + bw / 2, y + h * 0.73);
    hits.push({ i, x, y, w: bw, h });
  });
}

function panel(x, y, w, h, title, titleColor, ts) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  noStroke();
  fill(titleColor);
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text(title, x + 10, y + 7);
  textStyle(NORMAL);
  textSize(ts);
}

// What the stage does, its code, and (when the checkbox is on) its most common mistake
function drawStage(x, y, w, h, narrow) {
  const s = STAGES[cur], ts = narrow ? 11 : 16, lead = narrow ? 14 : 20, cs = narrow ? 11 : 13;
  panel(x, y, w, h, 'Stage ' + (cur + 1) + ' of 10: ' + s.name, s.color, ts);
  fill('black');
  text(s.explain, x + 10, y + ts + 16, w - 20, lead * (narrow ? 3 : 5) + 4);
  const cy = y + ts + 22 + lead * (narrow ? 3 : 5), chh = s.code.split('\n').length * (cs + 3) + 12;
  fill('whitesmoke');
  rect(x + 8, cy, w - 16, chh, 6);
  fill('black');
  textSize(cs);
  text(s.code, x + 14, cy + 7);
  if (!riskBox.checked()) return;
  const ry = cy + chh + 8, rh = y + h - ry - 8;
  fill('mistyrose');
  rect(x + 8, ry, w - 16, rh, 6);
  fill('firebrick');
  textStyle(BOLD);
  textSize(ts);
  text('What can go wrong', x + 14, ry + 6);
  textStyle(NORMAL);
  fill('black');
  text(s.risk, x + 14, ry + ts + 12, w - 28, rh - ts - 14);
}

// The real result of the stage: lines of text, then bars or the predicted-against-actual plot
function drawResult(x, y, w, h, res, narrow) {
  const ts = narrow ? 11 : 16, lead = narrow ? 14 : 20;
  panel(x, y, w, h, 'Result on this data', 'black', ts);
  let ty = y + (narrow ? 26 : 36);
  for (const line of res.lines) {                        // a line wider than the panel gets a box with room for its wrapped rows
    const rows = textWidth(line) <= w - 20 ? 1 : Math.ceil(textWidth(line) / (w - 20) + 0.25);
    if (rows === 1) text(line, x + 10, ty); else text(line, x + 10, ty, w - 20, rows * lead + 4);
    ty += rows * lead + (narrow ? 2 : 5);
  }
  const top = ty + (res.mark ? 22 : 8), ch = y + h - top - 10;
  textSize(narrow ? 11 : 14);
  if (res.bars) {
    const lw = narrow ? 118 : 146, bx = x + 10 + lw, bw = w - 20 - lw - (narrow ? 58 : 62), rh = Math.min(narrow ? 15 : 24, ch / res.bars.length);
    if (res.mark) {                                      // the VIF limit, or the mean of the fold scores
      const mx = bx + res.mark / res.max * bw;
      stroke('gray');
      strokeWeight(1);
      drawingContext.setLineDash([4, 3]);
      line(mx, top - 3, mx, top + res.bars.length * rh + 2);
      drawingContext.setLineDash([]);
      noStroke();
      fill('black');
      textAlign(CENTER, BOTTOM);
      text(res.markLabel, mx, top - 4);
    }
    res.bars.forEach((b, j) => {
      const my = top + j * rh + rh / 2, len = Math.max(1, b.v / res.max * bw);
      noStroke();
      fill('black');
      textAlign(RIGHT, CENTER);
      text(b.label, bx - 6, my);
      fill(b.color || STAGES[cur].color);
      rect(bx, my - rh * 0.36, len, rh * 0.72);
      fill('black');
      textAlign(LEFT, CENTER);
      text(b.text, bx + len + 5, my);
    });
  }
  if (res.scatter) {
    const all = test.map(i => D.price[i]).concat(R.pred), lo = Math.floor(Math.min(...all) / 50) * 50, hi = Math.ceil(Math.max(...all) / 50) * 50;
    const px = x + 50, pw = w - 66, ph = ch - 30, gx = v => px + (v - lo) / (hi - lo) * pw, gy = v => top + ph - (v - lo) / (hi - lo) * ph;
    stroke('gray');
    strokeWeight(1);
    line(px, top, px, top + ph);
    line(px, top + ph, px + pw, top + ph);
    stroke('crimson');
    drawingContext.setLineDash([5, 4]);
    line(gx(lo), gy(lo), gx(hi), gy(hi));
    drawingContext.setLineDash([]);
    stroke('white');
    fill(70, 130, 180, 200);
    test.forEach((i, k) => circle(gx(D.price[i]), gy(R.pred[k]), narrow ? 5 : 7));
    noStroke();
    fill('black');
    textSize(narrow ? 11 : 12);
    textAlign(CENTER, TOP);
    text(lo, gx(lo), top + ph + 3);
    text(hi, gx(hi), top + ph + 3);
    text('actual price ($1000s). Dashed line: prediction = actual', px + pw / 2, top + ph + 16);
    textAlign(RIGHT, CENTER);
    text(lo, px - 4, gy(lo));
    text(hi, px - 4, gy(hi));
    text('pred.', px - 4, top + ph / 2);
  }
}

function mousePressed() {
  const hit = hits.find(r => mouseX >= r.x && mouseX <= r.x + r.w && mouseY >= r.y && mouseY <= r.y + r.h);
  if (hit) cur = hit.i;
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
