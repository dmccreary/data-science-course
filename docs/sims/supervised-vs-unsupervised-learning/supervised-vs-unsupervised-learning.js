// Supervised vs Unsupervised Learning
// CANVAS_HEIGHT: 620
// Bloom L2 (Understand): students step through three stages (training data, training, new data)
// and compare side by side what a supervised and an unsupervised learner is given, what it
// learns, and what it outputs. Pointing at a use case explains it.
//
// Both models are really fit to the seeded data on screen:
//   Supervised: logistic regression P(spam) = 1 / (1 + e^-(b + w1 z1 + w2 z2)) on standardized
//   features z, trained by gradient descent on the mean log loss. The boundary is P = 0.5.
//   Unsupervised: K-Means with k = 3 on standardized features (Lloyd's algorithm started from
//   farthest-first centers). Groups are lettered A, B, C in order of spending.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 575;
let controlHeight = 45;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const N_PER_CLASS = 20, N_PER_BLOB = 14, K = 3, DATA_SEED = 12;
const LABEL_NAMES = ['not spam', 'spam'], LABEL_COLORS = ['teal', 'orangered'];
const GROUP_COLORS = ['royalblue', 'darkorchid', 'goldenrod'];
const STEPS = ['Training data', 'Train the model', 'New data'];
const STEP_TEXT = [
  'Both data sets have two features per row. Only the emails come with the answer attached: each one is labeled spam or not spam. ' +
  'Point at a use case above to read about it.',
  'With labels, the model learns a boundary that separates the two answers. Without labels, K-Means can only group the customers that sit close together.',
  'A new email gets a predicted label. A new customer is only placed in the nearest group, and a person still has to decide what each group means. ' +
  'Click inside either plot to move the new point.'
];
const ROWS = [
  { label: 'Training data', short: 'Data' }, { label: 'What it learns', short: 'Learns' }, { label: 'Output', short: 'Output' }
];
const SIDES = [
  { key: 'sup', name: 'Supervised Learning', insight: 'Learning with a teacher', color: 'seagreen',
    xLabel: 'Words in ALL CAPS (%)', yLabel: 'Exclamation marks / 100 words', yShort: '! per 100 words',
    xMax: 30, yMax: 10, xTicks: [0, 10, 20, 30], yTicks: [0, 5, 10],
    plain: ['Features X and labels y', 'A rule that maps X to y', 'A predicted label for new data'],
    math: ['{(x₁, y₁), (x₂, y₂), …}', 'a function f with f(x) ≈ y', 'ŷ = f(new x)'],
    uses: [['Spam detection', 'Spam detection: emails already marked spam or not spam train a classifier that labels new mail.'],
      ['Price prediction', 'Price prediction: past house sales, with features and the sale price, train a regression model that predicts a price.'],
      ['Medical diagnosis', 'Medical diagnosis: patient records with a confirmed diagnosis train a model that suggests the likely condition for a new patient.']] },
  { key: 'uns', name: 'Unsupervised Learning', insight: 'Learning to find structure', color: 'royalblue',
    xLabel: 'Annual spending ($1000s)', yLabel: 'Visits per month', yShort: 'Visits per month',
    xMax: 10, yMax: 20, xTicks: [0, 5, 10], yTicks: [0, 10, 20],
    plain: ['Features X only, no labels', 'Groups of similar rows', 'A group number, not a label'],
    math: ['{x₁, x₂, …}   (no y)', 'k centers μ that minimize Σ ‖x − nearest μ‖²', 'c(x) = index of the nearest μ'],
    uses: [['Customer segments', 'Customer segments: clustering groups customers with similar habits. Nobody says in advance what the groups are.'],
      ['Anomaly detection', 'Anomaly detection: rows that sit far from every group are flagged as unusual, such as a suspicious purchase.'],
      ['Topic modeling', 'Topic modeling: documents that use similar words are grouped into topics that no one labeled beforehand.']] }
];

let rngState = 1;
let emails = [];        // { x, y, label }   label 1 = spam, 0 = not spam
let customers = [];     // { x, y, group }
let logit = null;       // fitted logistic regression: { mx, my, sx, sy, b, w1, w2, correct }
let kmeans = null;      // fitted K-Means: { mx, my, sx, sy, centers: [{ x, y, n }] } in data units
let newEmail = { x: 14, y: 4.5 }, newCustomer = { x: 6.4, y: 10.5 };
let step = 1;
let prevButton, nextButton, mathCheckbox;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }
function clampTo(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

function makeData() {
  rngState = DATA_SEED;
  emails = [];
  for (let i = 0; i < 2 * N_PER_CLASS; i++) {
    const spam = i < N_PER_CLASS ? 0 : 1;
    emails.push({ x: clampTo((spam ? 17 : 7) + 4.5 * stdNormal(), 0.5, 29.5),
      y: clampTo((spam ? 5.6 : 2.6) + 1.5 * stdNormal(), 0.2, 9.8), label: spam });
  }
  customers = [];
  for (const [bx, by] of [[2.2, 5], [5.5, 14.5], [8, 8]]) {
    for (let i = 0; i < N_PER_BLOB; i++) {
      customers.push({ x: clampTo(bx + 0.75 * stdNormal(), 0.3, 9.7), y: clampTo(by + 1.7 * stdNormal(), 0.5, 19.5), group: 0 });
    }
  }
}

// Mean and standard deviation (ddof = 0) of both coordinates, used to standardize the features
function scaler(rows) {
  const n = rows.length, mean = k => rows.reduce((s, r) => s + r[k], 0) / n;
  const mx = mean('x'), my = mean('y');
  const sd = (k, m) => Math.sqrt(rows.reduce((s, r) => s + (r[k] - m) ** 2, 0) / n);
  return { mx, my, sx: sd('x', mx), sy: sd('y', my) };
}

// Logistic regression by gradient descent on the mean log loss
function fitLogistic() {
  const sc = scaler(emails), n = emails.length;
  let b = 0, w1 = 0, w2 = 0;
  for (let it = 0; it < 5000; it++) {
    let gb = 0, g1 = 0, g2 = 0;
    for (const e of emails) {
      const z1 = (e.x - sc.mx) / sc.sx, z2 = (e.y - sc.my) / sc.sy;
      const err = 1 / (1 + Math.exp(-(b + w1 * z1 + w2 * z2))) - e.label;
      gb += err; g1 += err * z1; g2 += err * z2;
    }
    b -= 0.5 * gb / n; w1 -= 0.5 * g1 / n; w2 -= 0.5 * g2 / n;
  }
  logit = { ...sc, b, w1, w2 };
  logit.correct = emails.filter(e => (pSpam(e.x, e.y) > 0.5 ? 1 : 0) === e.label).length;
}

function pSpam(x, y) {
  const m = logit;
  return 1 / (1 + Math.exp(-(m.b + m.w1 * (x - m.mx) / m.sx + m.w2 * (y - m.my) / m.sy)));
}

// K-Means on standardized features. Returns centers in data units, sorted by spending.
function fitKMeans() {
  const sc = scaler(customers);
  const pts = customers.map(c => [(c.x - sc.mx) / sc.sx, (c.y - sc.my) / sc.sy]);
  const d2 = (p, q) => (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2;
  const nearest = (p, cs) => cs.reduce((best, c, i) => d2(p, c) < d2(p, cs[best]) ? i : best, 0);
  let cs = [pts[0]];                                  // farthest-first starting centers
  while (cs.length < K) cs.push(pts.reduce((far, p) => d2(p, cs[nearest(p, cs)]) > d2(far, cs[nearest(far, cs)]) ? p : far));
  let assign = [];
  for (let it = 0; it < 100; it++) {
    const next = pts.map(p => nearest(p, cs));
    if (next.every((a, i) => a === assign[i])) break;
    assign = next;
    cs = cs.map((c, j) => {
      const mine = pts.filter((_, i) => assign[i] === j);
      return mine.length ? [mine.reduce((s, p) => s + p[0], 0) / mine.length, mine.reduce((s, p) => s + p[1], 0) / mine.length] : c;
    });
  }
  const order = cs.map((_, j) => j).sort((a, b) => cs[a][0] - cs[b][0]);
  customers.forEach((c, i) => { c.group = order.indexOf(assign[i]); });
  kmeans = { ...sc, centers: order.map(j => ({ x: sc.mx + sc.sx * cs[j][0], y: sc.my + sc.sy * cs[j][1], n: assign.filter(a => a === j).length })) };
}

// Index of the K-Means center nearest to a point, measured on the standardized features
function nearestGroup(x, y) {
  const m = kmeans, d = c => ((x - c.x) / m.sx) ** 2 + ((y - c.y) / m.sy) ** 2;
  return m.centers.reduce((best, c, i) => d(c) < d(m.centers[best]) ? i : best, 0);
}

function goTo(n) {
  step = n;
  if (step <= 1) prevButton.attribute('disabled', ''); else prevButton.removeAttribute('disabled');
  if (step >= STEPS.length) nextButton.attribute('disabled', ''); else nextButton.removeAttribute('disabled');
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  prevButton = createButton('Previous');
  prevButton.parent(mainElement);
  prevButton.position(10, drawHeight + 10);
  prevButton.mousePressed(() => { if (step > 1) goTo(step - 1); });
  nextButton = createButton('Next');
  nextButton.parent(mainElement);
  nextButton.position(88, drawHeight + 10);
  nextButton.mousePressed(() => { if (step < STEPS.length) goTo(step + 1); });
  mathCheckbox = createCheckbox(' Show math notation', false);
  mathCheckbox.parent(mainElement);
  mathCheckbox.position(145, drawHeight + 11);
  mathCheckbox.style('font-size', '16px');

  makeData();
  fitLogistic();
  fitKMeans();
  goTo(1);

  describe('Two panels side by side. The supervised panel plots emails colored by their spam or not spam label, then the boundary ' +
    'a classifier learns, then a prediction for a new email. The unsupervised panel plots unlabeled customers, then the three groups ' +
    'K-Means finds, then the group nearest a new customer. A table compares the training data, what is learned, and the output.', LABEL);
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600, ts = narrow ? 11 : 14;
  const w = canvasWidth - 2 * margin, gap = 10, half = (w - gap) / 2;
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textStyle(NORMAL);
  textSize(narrow ? 17 : 24);
  text('Supervised vs Unsupervised Learning', canvasWidth / 2, 8);

  // the three stages, with the current one filled in
  const pillW = (w - 2 * gap) / 3;
  textSize(narrow ? 11 : 14);
  for (let i = 0; i < STEPS.length; i++) {
    const on = i === step - 1;
    fill(on ? 'gold' : 'white');
    stroke(on ? 'darkgoldenrod' : 'silver');
    strokeWeight(1);
    rect(margin + i * (pillW + gap), 38, pillW, 22, 11);
    noStroke();
    fill(on ? 'black' : 'gray');
    textStyle(on ? BOLD : NORMAL);
    textAlign(CENTER, CENTER);
    text((i + 1) + '. ' + STEPS[i], margin + i * (pillW + gap) + pillW / 2, 49);
  }
  textStyle(NORMAL);

  const panelY = 68, plotH = narrow ? 120 : 166, panelH = plotH + (narrow ? 154 : 170);
  SIDES.forEach((s, i) => drawSide(s, margin + i * (half + gap), panelY, half, panelH, plotH, narrow, ts));

  // comparison table: the row for the current stage is highlighted
  const tableY = panelY + panelH + 8, rowH = narrow ? 30 : 28, labelW = narrow ? 50 : 120;
  const cellW = (w - labelW) / 2, showMath = mathCheckbox.checked();
  for (let r = 0; r < ROWS.length; r++) {
    const y = tableY + r * rowH, on = r === step - 1;
    fill(on ? 'lightyellow' : 'white');
    stroke('silver');
    strokeWeight(1);
    rect(margin, y, w, rowH);
    line(margin + labelW, y, margin + labelW, y + rowH);
    line(margin + labelW + cellW, y, margin + labelW + cellW, y + rowH);
    noStroke();
    textAlign(LEFT, TOP);
    textSize(ts);
    textStyle(BOLD);
    fill('black');
    text(narrow ? ROWS[r].short : ROWS[r].label, margin + 6, y + (rowH - ts) / 2);
    textStyle(NORMAL);
    textWrap(WORD);
    SIDES.forEach((s, i) => {
      const str = (showMath ? s.math : s.plain)[r];
      const oneLine = textWidth(str) <= cellW - 12;
      fill(i === 0 ? 'darkgreen' : 'navy');
      text(str, margin + labelW + i * cellW + 6, y + (oneLine ? (rowH - ts) / 2 : 2), cellW - 12, rowH - 2);
    });
  }

  // use cases: one row of chips per side (a column when narrow); pointing at one explains it
  const chipsY = tableY + ROWS.length * rowH + 8, chipH = narrow ? 19 : 24;
  let hovered = null;
  SIDES.forEach((s, i) => {
    let cx = margin + i * (half + gap), cy = chipsY;
    textSize(narrow ? 11 : 13);
    for (const [name, explanation] of s.uses) {
      const cw = narrow ? half : (half - 12) / 3;
      const over = mouseX >= cx && mouseX <= cx + cw && mouseY >= cy && mouseY <= cy + chipH;
      if (over) hovered = { text: explanation, color: s.color };
      fill(over ? s.color : 'white');
      stroke(s.color);
      strokeWeight(1.5);
      rect(cx, cy, cw, chipH, chipH / 2);
      noStroke();
      fill(over ? 'white' : 'black');
      textAlign(CENTER, CENTER);
      text(name, cx + cw / 2, cy + chipH / 2 + 1);
      if (narrow) cy += chipH + 2; else cx += cw + 6;
    }
  });

  // explanation of the current stage, or of the use case under the mouse
  const textY = chipsY + (narrow ? 3 * (chipH + 2) : chipH) + 7;
  noStroke();
  fill(hovered ? hovered.color : 'black');
  if (hovered && hovered.color === 'seagreen') fill('darkgreen');
  textAlign(LEFT, TOP);
  textSize(narrow ? 11 : 14);
  textWrap(WORD);
  text(hovered ? hovered.text : STEP_TEXT[step - 1], margin, textY, w, drawHeight - textY - 2);
}

// One learning paradigm: header, legend, scatter plot for the current stage, and a status line
function drawSide(s, x, y, w, h, plotH, narrow, ts) {
  const sup = s.key === 'sup', headH = narrow ? 38 : 44;
  fill('white');
  stroke(s.color);
  strokeWeight(1.5);
  rect(x, y, w, h, 8);
  fill(s.color);
  rect(x, y, w, headH, 8, 8, 0, 0);
  noStroke();
  fill('white');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(narrow ? 14 : 18);
  text(s.name, x + 10, y + 5);
  textStyle(ITALIC);
  textSize(narrow ? 11 : 13);
  text(s.insight, x + 10, y + (narrow ? 22 : 26));
  textStyle(NORMAL);

  // legend row
  const items = sup ? LABEL_NAMES.map((n, i) => [LABEL_COLORS[i], n]).reverse()
    : step === 1 ? [['gray', narrow ? 'customer (no label)' : 'one customer (no label)']]
      : GROUP_COLORS.map((c, i) => [c, (narrow ? '' : 'Group ') + 'ABC'[i]]).concat([[null, '✕ center']]);
  let lx = x + 10;
  textSize(narrow ? 11 : 13);
  textAlign(LEFT, CENTER);
  for (const [c, name] of items) {
    if (c) { fill(c); circle(lx + 5, y + headH + 12, 9); lx += 13; }
    fill('black');
    text(name, lx, y + headH + 12);
    lx += textWidth(name) + (narrow ? 7 : 12);
  }

  // plot frame, ticks, and axis titles
  const px = x + (narrow ? 32 : 48), py = y + headH + 24, pw = w - (narrow ? 42 : 62), ph = plotH;
  s.plot = { x: px, y: py, w: pw, h: ph };
  const gx = v => px + v / s.xMax * pw, gy = v => py + ph - v / s.yMax * ph;
  fill('white');
  stroke('gray');
  strokeWeight(1);
  rect(px, py, pw, ph);
  noStroke();
  fill('dimgray');
  textSize(narrow ? 11 : 12);
  textAlign(CENTER, TOP);
  for (const v of s.xTicks) text(v, gx(v), py + ph + 3);
  textAlign(RIGHT, CENTER);
  for (const v of s.yTicks) text(v, px - 4, gy(v));
  fill('black');
  textAlign(CENTER, TOP);
  text(s.xLabel, px + pw / 2, py + ph + 17);
  push();
  translate(x + (narrow ? 8 : 12), py + ph / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, CENTER);
  text(narrow ? s.yShort : s.yLabel, 0, 0);
  pop();

  const dot = narrow ? 7 : 9;
  let status;
  push();                                       // clip the drawing to the plot frame
  drawingContext.beginPath();
  drawingContext.rect(px, py, pw, ph);
  drawingContext.clip();
  if (sup) {
    const n = emails.length, nSpam = emails.filter(e => e.label === 1).length;
    status = n + ' emails. Every one has a label: ' + nSpam + ' spam and ' + (n - nSpam) + ' not spam.';
    if (step >= 2) {
      // boundary P(spam) = 0.5 is the line b + w1 z1 + w2 z2 = 0; shade the side each label is predicted on
      const m = logit, yAt = xv => m.my - m.sy * (m.b + m.w1 * (xv - m.mx) / m.sx) / m.w2;
      const y0 = yAt(0), y1 = yAt(s.xMax), far = 20 * s.yMax, spamAbove = pSpam(0, y0 + 1) > 0.5;
      noStroke();
      fill(spamAbove ? color(255, 69, 0, 38) : color(0, 128, 128, 38));
      quad(gx(0), gy(y0), gx(s.xMax), gy(y1), gx(s.xMax), gy(far), gx(0), gy(far));
      fill(spamAbove ? color(0, 128, 128, 38) : color(255, 69, 0, 38));
      quad(gx(0), gy(y0), gx(s.xMax), gy(y1), gx(s.xMax), gy(-far), gx(0), gy(-far));
      stroke('black');
      strokeWeight(2);
      line(gx(0), gy(y0), gx(s.xMax), gy(y1));
      status = 'Logistic regression learned the boundary line from the labels. ' + m.correct + ' of ' + n +
        ' training emails are on the correct side (' + nf(100 * m.correct / n, 1, 1) + '%).';
    }
    stroke('white');
    strokeWeight(1);
    for (const e of emails) { fill(LABEL_COLORS[e.label]); circle(gx(e.x), gy(e.y), dot); }
    if (step === 3) {
      const p = pSpam(newEmail.x, newEmail.y), label = p > 0.5 ? 1 : 0;
      drawNewPoint(gx(newEmail.x), gy(newEmail.y), LABEL_COLORS[label], dot);
      status = 'New email: ' + nf(newEmail.x, 1, 1) + '% caps, ' + nf(newEmail.y, 1, 1) + ' exclamation marks. Predicted label: ' +
        LABEL_NAMES[label] + ', with P(spam) = ' + nf(p, 1, 2) + '.';
    }
  } else {
    status = customers.length + ' customers and no labels: nothing says which customers belong together.';
    stroke('white');
    strokeWeight(1);
    for (const c of customers) { fill(step >= 2 ? GROUP_COLORS[c.group] : 'gray'); circle(gx(c.x), gy(c.y), dot); }
    if (step >= 2) {
      const cs = kmeans.centers;
      status = 'K-Means (k = 3) found groups of ' + cs[0].n + ', ' + cs[1].n + ', and ' + cs[2].n +
        ' customers. It cannot say what the groups mean.';
      if (step === 3) {
        const g = nearestGroup(newCustomer.x, newCustomer.y);
        stroke('dimgray');
        strokeWeight(1);
        drawingContext.setLineDash([4, 3]);
        line(gx(newCustomer.x), gy(newCustomer.y), gx(cs[g].x), gy(cs[g].y));
        drawingContext.setLineDash([]);
        drawNewPoint(gx(newCustomer.x), gy(newCustomer.y), GROUP_COLORS[g], dot);
        status = 'New customer: $' + nf(newCustomer.x, 1, 1) + 'k, ' + nf(newCustomer.y, 1, 1) + ' visits. Nearest center: Group ' +
          'ABC'[g] + '. That is a group number, not a label.';
      }
      cs.forEach((c, i) => {                    // group centers
        for (const [col, wt] of [['white', 6], [GROUP_COLORS[i], 3]]) {
          stroke(col);
          strokeWeight(wt);
          line(gx(c.x) - 6, gy(c.y) - 6, gx(c.x) + 6, gy(c.y) + 6);
          line(gx(c.x) - 6, gy(c.y) + 6, gx(c.x) + 6, gy(c.y) - 6);
        }
      });
    }
  }
  pop();

  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textSize(ts);
  textWrap(WORD);
  const sy = py + ph + (narrow ? 34 : 36);
  text(status, x + 8, sy, w - 16, y + h - sy - 2);
}

// The new, unlabeled row: a diamond filled with the color of the model's answer
function drawNewPoint(cx, cy, col, dot) {
  const r = dot * 0.95;
  fill(col);
  stroke('black');
  strokeWeight(2);
  quad(cx, cy - r, cx + r, cy, cx, cy + r, cx - r, cy);
}

// At the last stage a click inside a plot moves that plot's new point
function mousePressed() {
  if (step !== 3) return;
  for (const s of SIDES) {
    const p = s.plot;
    if (!p || mouseX < p.x || mouseX > p.x + p.w || mouseY < p.y || mouseY > p.y + p.h) continue;
    const pt = { x: (mouseX - p.x) / p.w * s.xMax, y: (p.y + p.h - mouseY) / p.h * s.yMax };
    if (s.key === 'sup') newEmail = pt; else newCustomer = pt;
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
