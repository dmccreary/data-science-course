// Feature Selection Race
// CANVAS_HEIGHT: 710
// Bloom L3-L4 (Apply, Analyze): students step three feature selection methods through the same
// data, predict each next move from the scores shown on the feature chips, and compare the
// features kept, the final score, and the work each method needed.
//
// Model: 200 simulated rows with 8 candidate features, from a seeded generator. Every score is the
// mean R^2 of 5-fold cross-validation (5 blocks of 40 consecutive rows, each scored by an ordinary
// least-squares model with an intercept fit to the other 160), which is what scikit-learn's
// cross_val_score(LinearRegression(), X[features], y, cv=5, scoring='r2').mean() returns.
// Every feature has to earn TOL = 0.005 of CV R^2:
//   Forward   starts empty; adds the feature with the largest gain; stops when no gain exceeds TOL.
//   Backward  starts full; removes the feature whose removal costs least; stops when every removal
//             would cost TOL or more.
//   Stepwise  starts empty; makes the single addition or removal that is best after charging TOL
//             per feature (largest of gain - TOL and TOL - cost); stops when none is positive.
// Nothing is scripted: the paths and the winner are computed from the data.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 630;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const N = 200, K = 5, P = 8, TOL = 0.005, STEP_MS = 1400;
const HOUSE = ['sqft', 'beds', 'baths', 'age', 'lot', 'garage', 'school', 'dist'];
const HOUSE_MIX = z => [z[0], 0.7 * z[0] + 0.7 * z[1], 0.6 * z[0] + 0.8 * z[2], z[3], z[4], z[5], z[6], z[7]];
// mix turns independent standard normal draws z into the 8 features. target = sum of beta * feature + sd * noise.
const SCENARIOS = [
  { name: 'Few useful features', rows: 'houses', target: 'price', names: HOUSE, mix: HOUSE_MIX,
    beta: [3, 0, 0, -1.5, 0, 0, 1.2, 0], sd: 2 },
  { name: 'Most features useful', rows: 'houses', target: 'price', names: HOUSE, mix: HOUSE_MIX,
    beta: [2, 0, 1, -1.2, 1, 0.9, 1.1, -1], sd: 1.5 },
  // rooms is close to beds + baths, so it stands in for both until they are in the model themselves
  { name: 'A redundant stand-in', rows: 'houses', target: 'price', names: ['rooms', 'beds', 'baths', 'sqft', 'age', 'lot', 'school', 'dist'],
    mix: z => [(z[1] + z[2] + 0.5 * z[0]) / 1.5, z[1], z[2], z[3], z[4], z[5], z[6], z[7]],
    beta: [0, 1.5, 1.2, 1.2, -0.8, 0, 0, 0], sd: 1.3 },
  // profit depends on sales minus costs, and the two are almost perfectly correlated (r = 0.985)
  { name: 'A pair that works together', rows: 'stores', target: 'profit', names: ['sales', 'costs', 'staff', 'area', 'age', 'ads', 'rating', 'hours'],
    mix: z => [z[0], 0.985 * z[0] + 0.1726 * z[1], z[2], z[3], z[4], z[5], z[6], z[7]],
    beta: [8, -8, 0, 0, 0, 1.2, 1, 0], sd: 1 }
];
const METHODS = [
  { name: 'Forward selection', short: 'Forward', color: 'royalblue', add: true, drop: false, rule: 'starts empty, adds the feature that gains the most', stop: 'no addition gains more than ' },
  { name: 'Backward elimination', short: 'Backward', color: 'sienna', add: false, drop: true, rule: 'starts with all 8, removes the feature that costs the least', stop: 'every removal costs more than ' },
  { name: 'Stepwise selection', short: 'Stepwise', color: 'seagreen', add: true, drop: true, rule: 'starts empty, adds or removes, whichever is better', stop: 'no addition or removal is worth ' }
];

let rngState = 1, seed = 1, scen = 0;
let X = [], Y = [];
let race = [];              // one state per method: { set, score, start, next, best, done, work, path }
let win = -1;               // index of the winning method once all three have stopped
let running = false, lastStep = 0;
let nextButton, runButton, resetButton, dataButton, dataSelect;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }

function makeData() {
  const sc = SCENARIOS[scen];
  rngState = 1000 * seed + scen;
  X = []; Y = [];
  for (let i = 0; i < N; i++) {
    const z = [];
    for (let j = 0; j < 9; j++) z.push(stdNormal());
    X.push(sc.mix(z));
    Y.push(X[i].reduce((s, v, j) => s + sc.beta[j] * v, 0) + sc.sd * z[8]);
  }
  resetRace();
}

// Ordinary least squares with an intercept on the listed rows and columns. Solves the normal
// equations (A'A) b = A'y by Gauss-Jordan elimination with partial pivoting. b[0] is the intercept.
function fitOLS(rows, cols) {
  const m = cols.length + 1, A = [];
  for (let a = 0; a < m; a++) A.push(new Array(m + 1).fill(0));
  for (const i of rows) {
    const v = [1, ...cols.map(c => X[i][c])];
    for (let a = 0; a < m; a++) {
      for (let b = 0; b < m; b++) A[a][b] += v[a] * v[b];
      A[a][m] += v[a] * Y[i];
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

// Mean R^2 over K folds of consecutive rows. R^2 of a fold = 1 - SSE / SST on the held-out rows.
function cvScore(set) {
  const cols = [], size = N / K;
  set.forEach((on, j) => { if (on) cols.push(j); });
  let total = 0;
  for (let k = 0; k < K; k++) {
    const train = [], held = [];
    for (let i = 0; i < N; i++) (i >= k * size && i < (k + 1) * size ? held : train).push(i);
    const b = fitOLS(train, cols), my = held.reduce((s, i) => s + Y[i], 0) / held.length;
    let sse = 0, sst = 0;
    for (const i of held) {
      const pred = cols.reduce((s, c, a) => s + b[a + 1] * X[i][c], b[0]);
      sse += (Y[i] - pred) ** 2;
      sst += (Y[i] - my) ** 2;
    }
    total += 1 - sse / sst;
  }
  return total / K;
}

// Score every move the method is allowed to make from its current set and pick the best one.
// next[f] is the CV R^2 after flipping feature f. A method with no worthwhile move is done.
function look(m, s) {
  s.next = new Array(P).fill(null);
  s.best = -1;
  let bestGain = 0;
  for (let f = 0; f < P; f++) {
    if (s.set[f] ? !m.drop : !m.add) continue;
    const trial = s.set.slice();
    trial[f] = !trial[f];
    s.next[f] = cvScore(trial);
    s.work++;
    const gain = s.next[f] - s.score + (s.set[f] ? TOL : -TOL);
    if (gain > bestGain) { bestGain = gain; s.best = f; }
  }
  s.done = s.best < 0;
}

function resetRace() {
  running = false;
  race = METHODS.map(m => {
    const s = { set: new Array(P).fill(!m.add), work: 1, path: [] };
    s.score = s.start = cvScore(s.set);
    look(m, s);
    return s;
  });
  finishStep();
}

function stepRace() {
  race.forEach((s, k) => {
    if (s.done) return;
    const f = s.best;
    s.set[f] = !s.set[f];
    s.score = s.next[f];
    s.path.push({ f, add: s.set[f], score: s.score });
    look(METHODS[k], s);
  });
  finishStep();
}

// Winner: the highest CV R^2, where scores within TOL count as a tie that goes to the method with
// fewer features and then to the one that scored fewer models.
function finishStep() {
  const over = race.every(s => s.done), best = Math.max(...race.map(s => s.score));
  win = -1;
  if (over) {
    running = false;
    race.forEach((s, k) => {
      if (s.score < best - TOL) return;
      if (win < 0 || kept(k).length < kept(win).length || (kept(k).length === kept(win).length && s.work < race[win].work)) win = k;
    });
  }
  if (!nextButton) return;
  for (const b of [nextButton, runButton]) { if (over) b.attribute('disabled', ''); else b.removeAttribute('disabled'); }
  runButton.html(running ? 'Pause' : 'Start Race');
}

function kept(k) { return race[k].set.map((on, j) => on ? j : -1).filter(j => j >= 0); }
function fmt(v) { return (v < 0 ? '−' : '') + Math.abs(v).toFixed(3); }
function signed(v) { return (v < -1e-9 ? '−' : '+') + Math.abs(v).toFixed(3); }

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  const button = (label, x, action) => {
    const b = createButton(label);
    b.parent(mainElement);
    b.position(x, drawHeight + 8);
    b.mousePressed(action);
    return b;
  };
  nextButton = button('Next Step', 10, () => { running = false; stepRace(); });
  runButton = button('Start Race', 95, () => { running = !running; lastStep = millis() - STEP_MS; finishStep(); });
  resetButton = button('Reset', 183, resetRace);
  dataButton = button('New Data', 243, () => { seed++; makeData(); });

  dataSelect = createSelect();
  dataSelect.parent(mainElement);
  dataSelect.position(55, drawHeight + 45);
  SCENARIOS.forEach(s => dataSelect.option(s.name));
  dataSelect.changed(() => { scen = SCENARIOS.findIndex(s => s.name === dataSelect.value()); makeData(); });

  makeData();

  describe('Three feature selection methods race on the same simulated data: forward selection, backward elimination, ' +
    'and stepwise selection. Each track has eight feature chips that light up when the feature is in the model and show ' +
    'how the cross-validated R squared would change if the chip were flipped. A table compares the features kept, the ' +
    'final score, and the number of models scored, and a panel names the winner and explains why.', LABEL);
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  if (running && millis() - lastStep >= STEP_MS) { lastStep = millis(); stepRace(); }

  const narrow = canvasWidth < 600, w = canvasWidth - 2 * margin, sc = SCENARIOS[scen];
  textWrap(WORD);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Feature Selection Race', canvasWidth / 2, 8);
  textSize(narrow ? 11 : 14);
  fill('dimgray');
  text(N + ' simulated ' + sc.rows + ', target: ' + sc.target + (narrow ? '. Scores: 5-fold CV R².' : '. Every score is a 5-fold cross-validated R² (CV R²).'),
    canvasWidth / 2, narrow ? 33 : 38);

  const top = narrow ? 50 : 60, th = 110, tableH = narrow ? 76 : 90;
  race.forEach((s, k) => drawTrack(k, margin, top + k * (th + 6), w, th, narrow));
  const ty = top + 3 * (th + 6);
  drawTable(margin, ty, w, tableH, narrow);
  drawVerdict(margin, ty + tableH + 6, w, drawHeight - ty - tableH - 14, narrow);

  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Data:', 10, drawHeight + 56);
}

// One method: its eight chips, the change each legal move would make, and the path so far
function drawTrack(k, x, y, w, h, narrow) {
  const m = METHODS[k], s = race[k], sc = SCENARIOS[scen];
  fill(k === win ? 'lemonchiffon' : 'white');
  stroke(k === win ? 'goldenrod' : 'silver');
  strokeWeight(k === win ? 2 : 1);
  rect(x, y, w, h, 10);
  noStroke();
  fill(m.color);
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(narrow ? 13 : 15);
  text(m.name + (k === win ? ' ★' : ''), x + 10, y + 6);
  const nameW = textWidth(m.name + (k === win ? ' ★' : ''));
  fill('black');
  textAlign(RIGHT, TOP);
  text((s.done ? 'stopped,  ' : '') + 'CV R² ' + fmt(s.score), x + w - 10, y + 6);
  textStyle(NORMAL);
  if (!narrow) {
    fill('dimgray');
    textAlign(LEFT, TOP);
    textSize(13);
    text(m.rule, x + 22 + nameW, y + 8);
  }

  const gap = narrow ? 3 : 6, cw = (w - 20 - 7 * gap) / P, ch = narrow ? 32 : 38, cy = y + (narrow ? 24 : 28);
  for (let f = 0; f < P; f++) {
    const cx = x + 10 + f * (cw + gap), on = s.set[f];
    fill(on ? m.color : 'gainsboro');
    stroke(f === s.best ? 'black' : 'silver');
    strokeWeight(f === s.best ? 3 : 1);
    rect(cx, cy, cw, ch, 6);
    noStroke();
    fill(on ? 'white' : 'black');
    textAlign(CENTER, CENTER);
    textStyle(BOLD);
    textSize(narrow ? 11 : 14);
    text(sc.names[f], cx + cw / 2, cy + ch * 0.3);
    textStyle(NORMAL);
    textSize(narrow ? 11 : 12);
    if (s.next[f] !== null) text(signed(s.next[f] - s.score), cx + cw / 2, cy + ch * 0.75);
    if (win >= 0 && sc.beta[f] !== 0) {               // revealed at the finish: a feature the target was built from
      fill('goldenrod');
      rect(cx + 3, cy + ch + 1, cw - 6, 4, 2);
    }
  }
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textSize(narrow ? 11 : 13);
  text(pathText(k), x + 10, cy + ch + 8, w - 20, y + h - cy - ch - 9);
}

// The moves made so far with the score after each, then the next move or the reason for stopping
function pathText(k) {
  const s = race[k], names = SCENARIOS[scen].names;
  let t = 'Start ' + fmt(s.start);
  for (const p of s.path) t += ' → ' + (p.add ? '+' : '−') + names[p.f] + ' ' + fmt(p.score);
  if (s.done) return t + '.  Stopped: ' + METHODS[k].stop + TOL + '.';
  return t + '.  Next: ' + (s.set[s.best] ? 'remove ' : 'add ') + names[s.best] + '.';
}

// How a method's final feature set compares with the features the target was really built from
function versus(k, narrow) {
  const sc = SCENARIOS[scen], set = race[k].set, extra = [], missed = [];
  sc.beta.forEach((b, j) => { if (b === 0 && set[j]) extra.push(sc.names[j]); if (b !== 0 && !set[j]) missed.push(sc.names[j]); });
  if (!extra.length && !missed.length) return 'exact match';
  const part = (a, word) => a.length ? [narrow ? a.length + ' ' + word : word + ': ' + a.join(', ')] : [];
  return [...part(extra, 'extra'), ...part(missed, 'missed')].join('; ');
}

function drawTable(x, y, w, h, narrow) {
  const sc = SCENARIOS[scen], rh = narrow ? 17 : 21, top = y + (narrow ? 20 : 24);
  const cFeat = x + (narrow ? 82 : 100), cScore = x + (narrow ? 244 : 468), cWork = x + (narrow ? 288 : 570), cTruth = x + (narrow ? 298 : 590);
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  line(x + 8, top, x + w - 8, top);
  noStroke();
  fill('dimgray');
  textSize(narrow ? 11 : 13);
  textAlign(LEFT, BOTTOM);
  text('Method', x + 10, top - 3);
  text('Features in the model', cFeat, top - 3);
  text(narrow ? 'Truth check' : 'Compared with the truth', cTruth, top - 3);
  textAlign(RIGHT, BOTTOM);
  text('CV R²', cScore, top - 3);
  text(narrow ? 'Scored' : 'Models scored', cWork, top - 3);

  race.forEach((s, k) => {
    const mid = top + 3 + k * rh + rh / 2, list = kept(k);
    if (k === win) { fill('lemonchiffon'); rect(x + 4, mid - rh / 2, w - 8, rh, 4); }
    fill(METHODS[k].color);
    textAlign(LEFT, CENTER);
    textStyle(BOLD);
    text(METHODS[k].short + (k === win ? ' ★' : ''), x + 10, mid);
    textStyle(NORMAL);
    if (narrow) {                                      // eight squares in chip order, then the count
      for (let f = 0; f < P; f++) {
        fill(s.set[f] ? METHODS[k].color : 'gainsboro');
        rect(cFeat + f * 11, mid - 5, 9, 10, 2);
      }
      fill('black');
      text(list.length + ' of ' + P, cFeat + 92, mid);
    } else {
      fill('black');
      text(list.length ? list.map(j => sc.names[j]).join(', ') : '(none yet)', cFeat, mid);
    }
    fill(win >= 0 ? 'black' : 'gray');
    text(win >= 0 ? versus(k, narrow) : narrow ? 'at the finish' : 'revealed at the finish', cTruth, mid);
    fill('black');
    textAlign(RIGHT, CENTER);
    text(fmt(s.score), cScore, mid);
    text(s.work, cWork, mid);
  });
}

// Instructions during the race, then the winner and what the three paths show
function drawVerdict(x, y, w, h, narrow) {
  const sc = SCENARIOS[scen], ts = narrow ? 11 : 14, names = a => a.map(j => sc.names[j]).join(', ');
  let title = 'How to read the race';
  let t = 'The number on a chip is the change in CV R² if that method flipped the chip: adding a gray feature or removing a ' +
    'colored one. A feature has to earn ' + TOL + ': an addition must gain more than that, a removal must cost less. The black ' +
    'outline marks each method\'s next move. Predict all three, then press Next Step. Winner: the highest CV R². Scores within ' +
    TOL + ' tie, and a tie goes to fewer features, then to fewer models scored.';
  if (win >= 0) {
    const W = METHODS[win].short, size = kept(win).length, who = a => a.map(k => METHODS[k].short).join(' and ');
    const tied = race.map((s, k) => k).filter(k => k !== win && race[k].score >= Math.max(...race.map(s => s.score)) - TOL);
    const larger = tied.filter(k => kept(k).length > size), level = tied.filter(k => kept(k).length === size);
    const same = race.every((s, k) => kept(k).join() === kept(0).join());
    title = 'Result: ' + W + ' wins';
    t = tied.length ? '' : W + ' has the highest CV R² by more than ' + TOL + '. ';
    if (larger.length) t += 'Scores within ' + TOL + ' are a tie, so fewer features wins: ' + W + ' needs ' + size + ', ' +
      larger.map(k => METHODS[k].short + ' ' + kept(k).length).join(', ') + '. ';
    if (level.length) t += (same ? 'All three methods end with the same ' + size + ' features' : W + ' and ' + who(level) + ' tie on score and size') +
      ', so work decides: ' + W + ' scored ' + race[win].work + ' models against ' + level.map(k => race[k].work).join(' and ') + '. ';
    if (same) t += win === 1 ? 'Most features are kept, so pruning down takes fewer steps than building up. '
      : 'Few features are kept, so building up takes fewer steps than pruning down. ';
    const stuck = kept(0).filter(j => !race[win].set[j]), undone = race[2].path.filter(p => !p.add).map(p => p.f);
    const unseen = kept(1).filter(j => !race[0].set[j]);
    if (stuck.length && kept(win).every(j => race[0].set[j])) t += 'Forward kept ' + names(stuck) +
      ': it can never remove a feature, even one that stops helping once others are in. ';
    if (undone.length) t += 'Stepwise added ' + names(undone) + ' and later removed it. ';
    if (unseen.length > 1 && race[1].score > race[0].score + TOL) t += 'Forward never added ' + names(unseen) +
      ': each one alone gains too little, and they only help together. A method that starts with every feature can see that. ';
    t += 'Truth (the data is simulated): ' + sc.target + ' was built from ' + names(sc.beta.map((b, j) => b !== 0 ? j : -1).filter(j => j >= 0)) +
      ' plus noise (gold bars).';
  }
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
  text(t, x + 10, y + ts + 14, w - 20, h - ts - 14);
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
