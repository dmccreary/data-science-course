// Residual Pattern Detective
// CANVAS_HEIGHT: 635
// Bloom L4-L5 (Analyze, Evaluate): students study four unlabeled residual plots, judge what each one
// says about the model, and get feedback. A plot receives its label only after a correct diagnosis.
//
// Model: each case is 60 points with x uniform on 0 to 10, drawn from a seeded generator.
//   healthy:  y = 5 + 3x + N(0, 2)                      straight line, constant noise
//   curved:   y = 5 + 3x +/- 0.4 (x - 5)^2 + N(0, 1.2)  the true relationship bends
//   funnel:   y = (5 + 3x) exp(0.16 z)                  noise grows with the size of y
//   clusters: y = 5 + 3x + 6 g + N(0, 0.8)              g = 1 for about 40% of rows (a hidden group)
// A straight line is fit to every case by least squares (slope = Sxy / Sxx). The plots show
// residual = actual - predicted against the predicted value. The evidence line reports the mean and
// sample SD of the residuals in the left, middle, and right thirds of each plot.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 520;
let controlHeight = 115;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const N = 60;
const PATTERNS = [
  { name: 'Healthy residuals', diagnosis: 'Model is working well', color: 'forestgreen',
    caption: 'Random pattern = model is working well.',
    means: 'The points scatter evenly above and below 0 at every predicted value. There is no systematic bias, so a straight line suits these data.',
    fix: 'Nothing to fix. Keep the model and report its metrics.',
    notThis: 'A model that is working well leaves no pattern: at every predicted value about half of the residuals are above 0, and the spread stays the same from left to right. This plot has a pattern.' },
  { name: 'Curved pattern', diagnosis: 'Relationship is non-linear', color: 'gold',
    caption: 'Curved pattern = try polynomial features.',
    means: 'The residuals are on one side of 0 at both ends and on the other side in the middle. The straight line is missing a non-linear relationship.',
    fix: 'Add a polynomial feature such as x squared, then check the residual plot again.',
    notThis: 'A non-linear relationship bends the residuals: they sit on one side of 0 at both ends and on the other side in the middle. This plot does not bend like that.' },
  { name: 'Funnel shape', diagnosis: 'Variance is not constant', color: 'darkorange',
    caption: 'Funnel shape = variance problems.',
    means: 'The residuals spread out as the predictions increase. The error variance is not constant (heteroscedasticity), so large predictions are much less certain than small ones.',
    fix: 'Consider a log transformation of the target, then check the residual plot again.',
    notThis: 'Non-constant variance makes the vertical spread grow steadily from one side of the plot to the other, like a funnel. That is not the main pattern here.' },
  { name: 'Clustered groups', diagnosis: 'A group variable is missing', color: 'crimson',
    caption: 'Clusters = missing categorical variable.',
    means: 'The residuals form separate bands at different levels. Each band is a group of rows that really differs from the others by a constant amount, and the model does not know about the groups.',
    fix: 'Include the grouping variable as a feature.',
    notThis: 'A missing group variable splits the residuals into separate bands at different heights. These points do not form bands.' }
];

let rngState = 1, seed = 11;
let cases = [];               // four cases in shuffled order
let selected = 0, wrongGuesses = 0;
let rects = [], notes = {};   // pixel rectangles, set by layout()
let diagnosisButtons = [], newButton;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }
function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }
function sampleSD(a) { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1)); }
function signed(v) { return (v < -0.05 ? '−' : v > 0.05 ? '+' : '') + nf(Math.abs(v), 1, 1); }

// Generate one data set, fit a straight line by least squares, and keep the residuals
function makeCase(kind) {
  const x = [], y = [], group = [], bend = uniform() < 0.5 ? 1 : -1;
  for (let i = 0; i < N; i++) {
    const xi = 10 * uniform(), z = stdNormal(), g = uniform() < 0.4 ? 1 : 0, line = 5 + 3 * xi;
    x.push(xi);
    group.push(g);
    y.push(kind === 0 ? line + 2 * z : kind === 1 ? line + bend * 0.4 * (xi - 5) ** 2 + 1.2 * z :
      kind === 2 ? line * Math.exp(0.16 * z) : line + 6 * g + 0.8 * z);
  }
  const mx = mean(x), my = mean(y);
  let sxx = 0, sxy = 0;
  for (let i = 0; i < N; i++) {
    sxx += (x[i] - mx) ** 2;
    sxy += (x[i] - mx) * (y[i] - my);
  }
  const slope = sxy / sxx, intercept = my - slope * mx;
  const pred = x.map(v => intercept + slope * v), res = y.map((v, i) => v - pred[i]);

  // evidence: residuals in thirds of the predicted values, or by hidden group for the cluster case
  const order = pred.map((_, i) => i).sort((a, b) => pred[a] - pred[b]);
  const thirds = [0, 1, 2].map(k => order.slice(k * N / 3, (k + 1) * N / 3).map(i => res[i]));
  let evidence = 'Left, middle, and right thirds of this plot: mean residual ' + thirds.map(t => signed(mean(t))).join(', ') +
    '; SD ' + thirds.map(t => nf(sampleSD(t), 1, 1)).join(', ') + '.';
  if (kind === 3) {
    const a = res.filter((_, i) => group[i] === 0), b = res.filter((_, i) => group[i] === 1);
    evidence = 'Mean residual by hidden group: ' + signed(mean(a)) + ' for ' + a.length + ' rows and ' + signed(mean(b)) +
      ' for ' + b.length + ' rows. ' + evidence.replace('Left', 'The left').replace(': mean', ' cannot show this: mean');
  }
  return { kind, x, y, group, pred, res, evidence, solved: false, lastWrong: -1,
    lim: 1.12 * Math.max(...res.map(Math.abs)), lo: Math.min(...pred), hi: Math.max(...pred) };
}

function newCases() {
  rngState = seed;
  const kinds = [0, 1, 2, 3];
  for (let i = 3; i > 0; i--) {             // Fisher-Yates shuffle of the panel order
    const j = Math.floor(uniform() * (i + 1));
    [kinds[i], kinds[j]] = [kinds[j], kinds[i]];
  }
  cases = kinds.map(makeCase);
  selected = 0;
  wrongGuesses = 0;
  updateButtons();
}

function diagnose(d) {
  const c = cases[selected];
  if (c.solved) return;
  if (d === c.kind) c.solved = true;
  else { c.lastWrong = d; wrongGuesses++; }
  updateButtons();
}

// The diagnosis buttons apply only while the selected case is unsolved
function updateButtons() {
  for (const b of diagnosisButtons) {
    if (cases[selected].solved) b.attribute('disabled', '');
    else b.removeAttribute('disabled');
  }
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  PATTERNS.forEach((p, d) => {
    const b = createButton(p.diagnosis);
    b.parent(mainElement);
    b.mousePressed(() => diagnose(d));
    diagnosisButtons.push(b);
  });
  newButton = createButton('New Cases');
  newButton.parent(mainElement);
  newButton.position(10, drawHeight + 80);
  newButton.mousePressed(() => { seed++; newCases(); });
  placeButtons();
  newCases();

  describe('Four unlabeled residual plots in a two by two grid. The student selects a plot and chooses one of four ' +
    'diagnoses: the model is working well, the relationship is non-linear, the variance is not constant, or a group ' +
    'variable is missing. A notes panel gives feedback, and a correct diagnosis labels the plot and explains the fix.', LABEL);
}

function placeButtons() {
  const bw = Math.min(250, (canvasWidth - 30) / 2);
  diagnosisButtons.forEach((b, i) => {
    b.position(10 + (i % 2) * (bw + 10), drawHeight + 8 + Math.floor(i / 2) * 35);
    b.size(bw, 28);
  });
}

function layout(narrow) {
  const w = canvasWidth - 2 * margin, top = narrow ? 36 : 42, gap = 6;
  const gridW = narrow ? w : Math.round(w * 0.58), gridH = narrow ? 262 : drawHeight - top - 10;
  const cw = (gridW - gap) / 2, ch = (gridH - gap) / 2;
  rects = [0, 1, 2, 3].map(i => ({ x: margin + (i % 2) * (cw + gap), y: top + Math.floor(i / 2) * (ch + gap), w: cw, h: ch }));
  notes = narrow ? { x: margin, y: top + gridH + gap, w, h: drawHeight - top - gridH - gap - 8 } :
    { x: margin + gridW + 10, y: top, w: w - gridW - 10, h: gridH };
}
function caseAt(mx, my) { return rects.findIndex(r => mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h); }

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600;
  layout(narrow);
  const hover = caseAt(mouseX, mouseY);
  cursor(hover >= 0 ? HAND : ARROW);

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Residual Pattern Detective', canvasWidth / 2, 8);

  for (let i = 0; i < 4; i++) drawCase(i, rects[i], i === hover, narrow);
  drawNotes(narrow);

  const solved = cases.filter(c => c.solved).length;
  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(narrow ? 14 : defaultTextSize);
  text('Solved ' + solved + ' of 4.   Wrong guesses: ' + wrongGuesses, 110, drawHeight + 92);
}

// One residual plot. Its name and icon appear only after a correct diagnosis.
function drawCase(i, r, hover, narrow) {
  const c = cases[i], pat = PATTERNS[c.kind], sel = i === selected, ts = narrow ? 11 : 12;
  fill('whitesmoke');
  stroke(sel ? 'royalblue' : hover ? 'gray' : 'silver');
  strokeWeight(sel ? 3 : 1);
  rect(r.x, r.y, r.w, r.h, 8);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(narrow ? 12 : 14);
  text('Case ' + 'ABCD'[i] + (c.solved ? ': ' + pat.name : ''), r.x + 8, r.y + 7);
  textStyle(NORMAL);
  if (c.solved) drawIcon(c.kind, r.x + r.w - 17, r.y + 15, 9);

  const px = r.x + 30, py = r.y + 28, pw = r.w - 40, ph = r.h - 28 - 22;
  const gx = v => px + 6 + (v - c.lo) / (c.hi - c.lo) * (pw - 12), gy = v => py + ph / 2 - v / c.lim * ph / 2;
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(px, py, pw, ph);
  const tick = [20, 10, 5, 2, 1].find(t => t <= c.lim) || 0.5;
  noStroke();
  fill('dimgray');
  textSize(ts);
  textAlign(RIGHT, CENTER);
  for (const v of [-tick, 0, tick]) text(v === 0 ? '0' : signed(v).replace('.0', ''), px - 3, gy(v));
  textAlign(CENTER, BOTTOM);
  text('predicted value', px + pw / 2, r.y + r.h - 4);
  push();
  translate(r.x + 9, py + ph / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, CENTER);
  text('residual', 0, 0);
  pop();

  stroke('red');
  strokeWeight(1.5);
  drawingContext.setLineDash([6, 4]);
  line(px, gy(0), px + pw, gy(0));
  drawingContext.setLineDash([]);
  stroke('white');
  strokeWeight(0.5);
  fill(30, 80, 220, 190);
  for (let k = 0; k < N; k++) circle(gx(c.pred[k]), gy(c.res[k]), narrow ? 6 : 7);
}

// Green check for a healthy plot, warning triangle for the three problem patterns
function drawIcon(kind, x, y, s) {
  fill(PATTERNS[kind].color);
  stroke('black');
  strokeWeight(1);
  if (kind === 0) {
    circle(x, y, 2 * s);
    stroke('white');
    strokeWeight(2.5);
    noFill();
    line(x - s * 0.5, y, x - s * 0.1, y + s * 0.4);
    line(x - s * 0.1, y + s * 0.4, x + s * 0.5, y - s * 0.4);
  } else {
    triangle(x - s, y + s * 0.8, x + s, y + s * 0.8, x, y - s);
    noStroke();
    fill('black');
    textAlign(CENTER, CENTER);
    textStyle(BOLD);
    textSize(s * 1.3);
    text('!', x, y + s * 0.15);
    textStyle(NORMAL);
  }
}

// Prompt, feedback on a wrong guess, or the explanation of a solved case
function drawNotes(narrow) {
  const c = cases[selected], pat = PATTERNS[c.kind], ts = narrow ? 12 : 15, lh = ts + 6, x = notes.x + 10;
  const allSolved = cases.every(k => k.solved), gapLine = narrow ? '\n' : '\n\n';
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(notes.x, notes.y, notes.w, notes.h, 10);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text('Case ' + 'ABCD'[selected] + (c.solved ? ': ' + pat.name : ' notes'), x, notes.y + 7);

  let head, body;
  if (c.solved) {
    fill('darkgreen');
    head = 'Correct: ' + pat.diagnosis.toLowerCase();
    body = pat.caption + ' ' + pat.means + gapLine + 'What to do: ' + pat.fix + gapLine + c.evidence + gapLine +
      (allSolved ? 'All four cases are solved, with ' + wrongGuesses + (wrongGuesses === 1 ? ' wrong guess.' : ' wrong guesses.') +
        ' Click any plot to review it, or press New Cases.' : 'Click another plot to continue.');
  } else if (c.lastWrong >= 0) {
    fill('firebrick');
    head = 'Not "' + PATTERNS[c.lastWrong].diagnosis.toLowerCase() + '"';
    body = PATTERNS[c.lastWrong].notThis + gapLine + 'Look at the plot again and choose another diagnosis.';
  } else {
    fill('mediumblue');
    head = 'What is your diagnosis?';
    body = 'A straight line was fit to ' + N + ' data points. This plot shows what the line missed: residual = actual − predicted, ' +
      'plotted against the predicted value.' + gapLine + 'What does the pattern say about the model? Choose one of the four ' +
      'diagnoses below. Click another plot to work on a different case.';
  }
  textSize(ts);
  text(head, x, notes.y + 9 + lh);
  textStyle(NORMAL);
  fill('black');
  textWrap(WORD);
  const y = notes.y + 11 + 2 * lh;
  text(body, x, y, notes.w - 20, notes.y + notes.h - y - 2);
}

function mousePressed() {
  const i = caseAt(mouseX, mouseY);
  if (i < 0) return;
  selected = i;
  updateButtons();
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  placeButtons();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
