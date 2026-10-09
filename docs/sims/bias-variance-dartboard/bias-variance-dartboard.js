// Bias-Variance Dartboard
// CANVAS_HEIGHT: 580
// Bloom L2-L3 (Understand, Apply): students read four reference dartboards, then set a model
// complexity, throw darts, and use the measured bias and variance to say which kind of error
// dominates. Nothing moves until the slider or a button is used.
//
// Model: the bullseye is the true value and each dart is one prediction. The board radius is 10
// units. A dart lands at an aim point plus normal noise with the same SD in x and in y.
//   bias     = distance from the bullseye to the average landing point (the X marker)
//   variance = average squared distance of the darts from their own average point (divide by n)
//   total    = average squared distance of the darts from the bullseye = bias^2 + variance (exact)
// The slider is a designed analogy, not a fitted model: complexity c puts the aim point
// 0.6 (10 - c) units from the bullseye and gives the darts an SD of 0.6 + 0.3 (c - 1) per axis,
// so the true bias falls from 5.4 to 0 while the true variance 2 SD^2 rises from 0.72 to 21.78.
// Every bias, variance, and total on screen is computed from the darts that are drawn.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 500;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let sliderLeftMargin = 185;
let defaultTextSize = 16;

const REFS = [
  { title: 'Low bias, low variance', note: 'The goal: accurate and consistent', offset: 0, sd: 0.8, seed: 11 },
  { title: 'Low bias, high variance', note: 'Accurate on average, but inconsistent', offset: 0, sd: 2.8, seed: 17 },
  { title: 'High bias, low variance', note: 'Consistent, but systematically wrong', offset: 5, sd: 0.8, seed: 21 },
  { title: 'High bias, high variance', note: 'The worst: wrong and inconsistent', offset: 4.5, sd: 2.6, seed: 27 }
];
const AIM_X = -Math.cos(Math.PI / 6), AIM_Y = Math.sin(Math.PI / 6);   // aim points lie up and to the left
const MAX_DARTS = 200, BAR_MAX = 40;
const aimOffset = c => 0.6 * (10 - c);            // true bias at complexity c
const spread = c => 0.6 + 0.3 * (c - 1);          // SD of the darts in each direction at complexity c

let rngState = 1, mainState = 1;
let darts = [];                                   // [x, y] of each dart on the main board, in board units
let complexitySlider, throwButton, clearButton;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }

function throwDarts(n, offset, sd) {
  const out = [];
  for (let i = 0; i < n; i++) out.push([offset * AIM_X + sd * stdNormal(), offset * AIM_Y + sd * stdNormal()]);
  return out;
}

// Bias, variance, and total error of a set of darts, measured from the darts themselves
function dartStats(d) {
  const n = d.length;
  const mx = d.reduce((s, p) => s + p[0], 0) / n, my = d.reduce((s, p) => s + p[1], 0) / n;
  const variance = d.reduce((s, p) => s + (p[0] - mx) ** 2 + (p[1] - my) ** 2, 0) / n;
  const total = d.reduce((s, p) => s + p[0] ** 2 + p[1] ** 2, 0) / n;
  return { mx, my, bias: Math.hypot(mx, my), variance, total };
}

// Add n darts at the current complexity, continuing the seeded stream for this setting
function addDarts(n) {
  const c = complexitySlider.value();
  rngState = mainState;
  darts = darts.concat(throwDarts(Math.min(n, MAX_DARTS - darts.length), aimOffset(c), spread(c)));
  mainState = rngState;
  if (darts.length >= MAX_DARTS) throwButton.attribute('disabled', ''); else throwButton.removeAttribute('disabled');
}

function resetBoard(n) {
  mainState = 10780 + complexitySlider.value();
  darts = [];
  addDarts(n);
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  complexitySlider = createSlider(1, 10, 3, 1);
  complexitySlider.parent(mainElement);
  complexitySlider.position(sliderLeftMargin, drawHeight + 8);
  complexitySlider.size(canvasWidth - sliderLeftMargin - margin);
  complexitySlider.input(() => resetBoard(10));

  throwButton = createButton('Throw 10 Darts');
  throwButton.parent(mainElement);
  throwButton.position(10, drawHeight + 45);
  throwButton.mousePressed(() => addDarts(10));

  clearButton = createButton('Clear Board');
  clearButton.parent(mainElement);
  clearButton.position(130, drawHeight + 45);
  clearButton.mousePressed(() => resetBoard(0));

  for (const ref of REFS) {
    rngState = ref.seed;
    ref.darts = throwDarts(12, ref.offset, ref.sd);
  }
  resetBoard(10);

  describe('Four reference dartboards show the combinations of low and high bias with low and high variance. A fifth board ' +
    'shows darts thrown by a model whose complexity is set with a slider. Bars report the squared bias, the variance, ' +
    'and their sum, all measured from the darts on the board.', LABEL);
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600;
  const w = canvasWidth - 2 * margin;
  const c = complexitySlider.value();
  const s = darts.length ? dartStats(darts) : null;
  // which reference board do the darts resemble? One error must be at least twice the other.
  let match = -1;
  if (s && s.bias ** 2 > 2 * s.variance) match = 2;
  if (s && s.variance > 2 * s.bias ** 2) match = 1;

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Bias-Variance Dartboard', canvasWidth / 2, 8);

  const top = narrow ? 36 : 44;
  const refW = narrow ? w : w * 0.47, cellW = refW / 2, cellH = narrow ? 152 : (drawHeight - top - 6) / 2;
  for (let i = 0; i < 4; i++) {
    drawRef(margin + (i % 2) * cellW, top + floor(i / 2) * cellH, cellW, cellH, REFS[i], i === match, narrow);
  }
  if (narrow) drawMain(margin + 3, top + 2 * cellH + 2, w - 6, drawHeight - top - 2 * cellH - 8, c, s, match, narrow);
  else drawMain(margin + refW + 10, top + 3, w - refW - 10, drawHeight - top - 12, c, s, match, narrow);

  // control label
  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Model complexity: ' + c, 10, drawHeight + 18);
  fill('dimgray');
  textSize(narrow ? 12 : 14);
  text('1 = simplest, 10 = most complex', 235, drawHeight + 57);
}

// A dartboard with its darts, the average landing point (X), and the bias segment from the bullseye
function drawBoard(cx, cy, R, d, dot) {
  const u = R / 10;
  stroke('gray');
  strokeWeight(1);
  for (let ring = 5; ring >= 1; ring--) {
    fill(ring % 2 ? 'white' : 'lightsteelblue');
    circle(cx, cy, ring * 4 * u);
  }
  noStroke();
  fill('crimson');
  circle(cx, cy, 2 * u);
  stroke('white');
  strokeWeight(1);
  fill('black');
  for (const p of d) circle(cx + p[0] * u, cy - p[1] * u, dot);
  if (!d.length) return;
  const st = dartStats(d), ax = cx + st.mx * u, ay = cy - st.my * u, k = dot * 0.7;
  stroke('darkorange');
  strokeWeight(dot * 0.45);
  line(cx, cy, ax, ay);
  for (const [col, wt] of [['black', dot * 0.55], ['gold', dot * 0.25]]) {
    stroke(col);
    strokeWeight(wt);
    line(ax - k, ay - k, ax + k, ay + k);
    line(ax - k, ay + k, ax + k, ay - k);
  }
}

// One of the four reference boards, outlined when the main board matches it
function drawRef(x, y, w, h, ref, lit, narrow) {
  const ts = narrow ? 11 : 13, R = narrow ? 34 : 66, st = dartStats(ref.darts);
  fill(lit ? 'lemonchiffon' : 'white');
  stroke(lit ? 'goldenrod' : 'silver');
  strokeWeight(lit ? 3 : 1);
  rect(x + 3, y + 3, w - 6, h - 6, 10);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text(ref.title, x + w / 2, y + 9);
  textStyle(NORMAL);
  const cy = y + (narrow ? 27 : 32) + R;
  drawBoard(x + w / 2, cy, R, ref.darts, narrow ? 4 : 6);
  noStroke();
  fill('black');
  textSize(ts);
  textAlign(CENTER, TOP);
  textWrap(WORD);
  text(ref.note, x + 8, cy + R + 5, w - 16, 2 * ts + 8);
  fill('dimgray');
  textAlign(CENTER, BOTTOM);
  text('bias ' + nf(st.bias, 1, 2) + ', variance ' + nf(st.variance, 1, 2), x + w / 2, y + h - 8);
}

// The interactive board and the errors measured from its darts
function drawMain(x, y, w, h, c, s, match, narrow) {
  const ts = narrow ? 11 : 14, lh = ts + (narrow ? 3 : 6);
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);

  // wide: board on top, text below. narrow: board on the left, text on the right.
  const R = narrow ? (h - 16) / 2 : 92;
  const cx = narrow ? x + 10 + R : x + w / 2, cy = narrow ? y + h / 2 : y + 56 + R;
  const tx = narrow ? x + 2 * R + 22 : x + 14, tw = x + w - tx - 10;
  let ty = y + (narrow ? 6 : 8);
  drawBoard(cx, cy, R, darts, narrow ? 5 : 7);

  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 2);
  text('Your model: complexity ' + c, tx, ty);
  textStyle(NORMAL);
  textSize(ts);
  ty += lh + 4;
  fill('dimgray');
  text('Setting: true bias ' + nf(aimOffset(c), 1, 1) + ', true variance ' + nf(2 * spread(c) ** 2, 1, 1), tx, ty);
  ty = narrow ? ty + lh : cy + R + 14;
  fill('black');
  if (!s) {
    textWrap(WORD);
    text('The board is empty. Press Throw 10 Darts to see where this model lands.', tx, ty, tw, 60);
    return;
  }
  text('Measured from ' + darts.length + ' darts: bias = ' + nf(s.bias, 1, 2) + (narrow ? '' : '  (bullseye to ✕)'), tx, ty);
  ty += lh + 2;

  // bars on a common scale: bias squared, variance, and the two stacked
  const rowH = narrow ? 15 : 22, lw = ts * 5.7, vw = ts * 3.3, bw = tw - lw - vw, sc = bw / BAR_MAX;
  const b2 = s.bias ** 2;
  const rows = [['Bias²', b2, [[b2, 'darkorange']]], ['Variance', s.variance, [[s.variance, 'mediumpurple']]],
    ['Total error', s.total, [[b2, 'darkorange'], [s.variance, 'mediumpurple']]]];
  for (let i = 0; i < 3; i++) {
    const by = ty + i * rowH;
    noStroke();
    fill('black');
    textAlign(LEFT, CENTER);
    text(rows[i][0], tx, by + rowH / 2);
    text(nf(rows[i][1], 1, 2), tx + lw + bw + 6, by + rowH / 2);
    fill('whitesmoke');
    stroke('silver');
    rect(tx + lw, by + 2, bw, rowH - 4);
    noStroke();
    let used = 0;
    for (const [v, col] of rows[i][2]) {
      const len = Math.min(v, BAR_MAX - used);
      fill(col);
      rect(tx + lw + used * sc, by + 2, len * sc, rowH - 4);
      used += len;
    }
  }
  ty += 3 * rowH + (narrow ? 3 : 8);

  textAlign(LEFT, TOP);
  textWrap(WORD);
  if (!narrow) {
    fill('dimgray');
    textSize(13);
    text('Total error = bias² + variance. It is also the average squared distance of the darts from the bullseye.', tx, ty, tw, 36);
    ty += 42;
  }
  const verdicts = { '-1': ['Balanced: neither error dominates. Compare the total error with the settings on either side.', 'darkgreen'],
    1: ['Mostly variance: centered on average but scattered, like an overfit model.', 'rebeccapurple'],
    2: ['Mostly bias: a tight group that misses the same way, like an underfit model.', 'chocolate'] };
  fill(verdicts[match][1]);
  textStyle(BOLD);
  textSize(narrow ? 11 : 14);
  text(verdicts[match][0], tx, ty, tw, y + h - ty - 2);
  textStyle(NORMAL);
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  complexitySlider.size(canvasWidth - sliderLeftMargin - margin);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
