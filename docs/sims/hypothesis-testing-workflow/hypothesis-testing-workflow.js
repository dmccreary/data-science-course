// Hypothesis Testing Workflow
// CANVAS_HEIGHT: 580
// Bloom L3 (Apply): students carry out the steps of a hypothesis test on a coin-flip example.
// They step through the flowchart, set the number of heads and the significance level, and
// see which branch of the decision the data lead to.
//
// Model: the null hypothesis H0 says the coin is fair, so the number of heads X in 100 flips is
// Binomial(100, 0.5). The two-sided exact p-value (as scipy.stats.binomtest computes it) is the
// total probability, under H0, of every count that is no more likely than the observed count.
// Reject H0 when p < alpha. The confidence interval for the true proportion is the exact
// (Clopper-Pearson) interval at level 1 - alpha, found by bisection on the binomial tails.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 500;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let sliderLeftMargin = 205;
let defaultTextSize = 16;

const N_FLIPS = 100, P0 = 0.5;
const NODES = [
  { label: 'Research Question', color: 'lightskyblue' },
  { label: 'State Hypotheses H₀ and H₁', color: 'lightgreen' },
  { label: 'Choose Significance Level α', color: 'lightgreen' },
  { label: 'Collect Data, Compute Test Statistic', color: 'lightgreen' },
  { label: 'Calculate P-value', color: 'orange' },
  { label: 'p-value < α ?', color: 'gold' },
  { label: 'Reject H₀', color: 'lightcoral' },
  { label: 'Fail to Reject H₀', color: 'silver' },
  { label: 'Report Results', color: 'lightskyblue' }
];
const LOG_FACT = [0];
for (let i = 1; i <= N_FLIPS; i++) LOG_FACT.push(LOG_FACT[i - 1] + Math.log(i));

let prevButton, nextButton, alphaSelect, headsSlider;
let cur = 0;                // index of the selected node
let nodeBoxes = [];         // clickable rectangles of the last frame

// Binomial probability of k heads in N_FLIPS flips when P(heads) = p
function pmf(k, p) {
  return Math.exp(LOG_FACT[N_FLIPS] - LOG_FACT[k] - LOG_FACT[N_FLIPS - k] + k * Math.log(p) + (N_FLIPS - k) * Math.log(1 - p));
}
// Is a count of i at least as extreme as the observed count k? (no more likely under H0)
function asExtreme(i, k) { return pmf(i, P0) <= pmf(k, P0) * (1 + 1e-7); }
function pValue(k) {
  let sum = 0;
  for (let i = 0; i <= N_FLIPS; i++) if (asExtreme(i, k)) sum += pmf(i, P0);
  return Math.min(1, sum);
}
// P(X >= k) when upper is true, otherwise P(X <= k), for P(heads) = p
function tail(k, p, upper) {
  let sum = 0;
  for (let i = upper ? k : 0; i <= (upper ? N_FLIPS : k); i++) sum += pmf(i, p);
  return sum;
}
// Exact (Clopper-Pearson) interval: the lower limit makes P(X >= k) = alpha / 2, the upper makes P(X <= k) = alpha / 2
function confidenceInterval(k, alpha) {
  const solve = upper => {
    let lo = 0, hi = 1;
    for (let i = 0; i < 40; i++) {
      const mid = (lo + hi) / 2;
      if ((tail(k, mid, upper) > alpha / 2) === upper) hi = mid; else lo = mid;
    }
    return (lo + hi) / 2;
  };
  return [solve(true), solve(false)];
}

// Position of a node along the path: the two outcomes share position 6
function position(i) { return i <= 5 ? i : i === 8 ? 7 : 6; }

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  prevButton = createButton('Previous');
  prevButton.parent(mainElement);
  prevButton.position(10, drawHeight + 8);
  prevButton.mousePressed(() => step(-1));
  nextButton = createButton('Next');
  nextButton.parent(mainElement);
  nextButton.position(88, drawHeight + 8);
  nextButton.mousePressed(() => step(1));

  const row = createDiv();
  row.parent(mainElement);
  row.position(150, drawHeight + 8);
  row.style('font-size', '16px');
  createSpan('Significance level α: ').parent(row);
  alphaSelect = createSelect();
  alphaSelect.parent(row);
  ['0.10', '0.05', '0.01'].forEach(a => alphaSelect.option(a));
  alphaSelect.selected('0.05');
  alphaSelect.style('font-size', '15px');

  headsSlider = createSlider(30, 70, 60, 1);
  headsSlider.parent(mainElement);
  headsSlider.position(sliderLeftMargin, drawHeight + 45);
  headsSlider.size(canvasWidth - sliderLeftMargin - margin);

  describe('A flowchart of a hypothesis test: research question, hypotheses, significance level, data and test ' +
    'statistic, p-value, a decision diamond, the two outcomes, and reporting. A coin-flip example runs through it. ' +
    'A slider sets the number of heads in 100 flips and a menu sets alpha. A panel explains the selected step with ' +
    'the numbers of the example, Python code, and a chart of the binomial distribution with the p-value shaded.', LABEL);
}

// Move along the path. From the decision, Next goes to the outcome that the data lead to.
function step(dir) {
  const st = testState();
  const path = [0, 1, 2, 3, 4, 5, st.reject ? 6 : 7, 8];
  cur = path[constrain(position(cur) + dir, 0, 7)];
}

function testState() {
  const k = headsSlider.value(), alpha = parseFloat(alphaSelect.value());
  const p = pValue(k);
  return { k, alpha, p, reject: p < alpha, pText: nf(p, 1, p < 0.001 ? 5 : 4) };
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const st = testState();
  const narrow = canvasWidth < 600;
  if (position(cur) === 0) prevButton.attribute('disabled', ''); else prevButton.removeAttribute('disabled');
  if (position(cur) === 7) nextButton.attribute('disabled', ''); else nextButton.removeAttribute('disabled');
  textWrap(WORD);

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Hypothesis Testing Workflow', canvasWidth / 2, 8);

  const flowW = narrow ? 162 : 290;
  drawFlow(margin, flowW, st, narrow);
  drawPanel(margin + flowW + 8, 44, canvasWidth - 2 * margin - flowW - 8, drawHeight - 52, st, narrow);

  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Heads in ' + N_FLIPS + ' flips: ' + st.k, 10, drawHeight + 56);
}

function arrow(x1, y1, x2, y2, col, weight) {
  stroke(col);
  strokeWeight(weight);
  line(x1, y1, x2, y2);
  const a = atan2(y2 - y1, x2 - x1);
  noStroke();
  fill(col);
  triangle(x2, y2, x2 - 8 * cos(a - 0.45), y2 - 8 * sin(a - 0.45), x2 - 8 * cos(a + 0.45), y2 - 8 * sin(a + 0.45));
}

// The flowchart. Steps already reached are in full color, later steps are pale, and the
// outcome that the data lead to is only shown once the student has reached the decision.
function drawFlow(x0, w, st, narrow) {
  const cx = x0 + w / 2, nw = min(w - 8, 250), nh = 36, top = 48, pitch = (drawHeight - 10 - nh - top) / 7;
  const ts = narrow ? 11 : 14, here = position(cur), revealed = here >= 6, taken = st.reject ? 6 : 7;
  const rowMid = i => top + i * pitch + nh / 2;
  // [center x, center y, width, height] of each node
  const geo = NODES.map((n, i) => i <= 5 ? [cx, rowMid(i), nw, nh] : i === 8 ? [cx, rowMid(7), nw, nh]
    : [cx + (i === 6 ? -1 : 1) * (nw / 4 + 3), rowMid(6), nw / 2 - 6, nh]);
  geo[5][2] = nw * 0.84;
  geo[5][3] = nh + 12;

  // arrows
  for (let i = 0; i < 5; i++) arrow(cx, geo[i][1] + geo[i][3] / 2, cx, geo[i + 1][1] - geo[i + 1][3] / 2 - 1, 'dimgray', 1.5);
  for (const b of [6, 7]) {
    const side = b === 6 ? -1 : 1, on = revealed && b === taken;
    const col = on ? 'black' : revealed ? 'silver' : 'dimgray', wt = on ? 2.5 : 1.5;
    const sx = cx + side * geo[5][2] / 4, sy = geo[5][1] + geo[5][3] / 4;
    arrow(sx, sy, geo[b][0], geo[b][1] - nh / 2 - 1, col, wt);
    arrow(geo[b][0], geo[b][1] + nh / 2, geo[b][0], geo[8][1] - nh / 2 - 1, col, wt);
    noStroke();
    fill(col);
    textSize(ts);
    textStyle(on ? BOLD : NORMAL);
    textAlign(side < 0 ? RIGHT : LEFT, CENTER);
    text(b === 6 ? 'Yes' : 'No', geo[b][0] + side * (narrow ? 14 : 22), sy + 8);
    textStyle(NORMAL);
  }

  // nodes
  nodeBoxes = [];
  NODES.forEach((n, i) => {
    const [nx, ny, bw, bh] = geo[i];
    const reached = position(i) <= here && (i < 6 || i > 7 || i === taken);
    const c = color(n.color);
    c.setAlpha(reached ? 255 : i === 6 || i === 7 ? (revealed ? 60 : 120) : 110);
    fill('white');
    noStroke();
    if (i !== 5) rect(nx - bw / 2, ny - bh / 2, bw, bh, i === 0 || i === 8 ? 18 : 6);
    fill(c);
    stroke(i === cur ? 'black' : 'gray');
    strokeWeight(i === cur ? 3 : 1);
    if (i === 5) quad(nx - bw / 2, ny, nx, ny - bh / 2, nx + bw / 2, ny, nx, ny + bh / 2);
    else rect(nx - bw / 2, ny - bh / 2, bw, bh, i === 0 || i === 8 ? 18 : 6);
    noStroke();
    fill(reached || i === cur ? 'black' : 'dimgray');
    textSize(ts);
    textStyle(i === cur ? BOLD : NORMAL);
    textAlign(CENTER, CENTER);
    text(n.label, nx - bw / 2 + 4, ny - bh / 2, bw - 8, bh);
    textStyle(NORMAL);
    nodeBoxes.push([nx - bw / 2, ny - bh / 2, nx + bw / 2, ny + bh / 2]);
  });
  const over = nodeAt(mouseX, mouseY);
  cursor(over >= 0 ? HAND : ARROW);
}

// Explanation, coin example, and Python code for the selected node
function stepContent(i, st) {
  const k = st.k, a = alphaSelect.value(), prop = nf(k / N_FLIPS, 1, 2), cmp = st.reject ? 'below' : 'not below';
  const far = Math.abs(k - 50), extreme = far === 0 ? 'any count' : (50 - far) + ' or fewer, or ' + (50 + far) + ' or more';
  const outcome = 'p = ' + st.pText + ' is ' + cmp + ' α = ' + a;
  switch (i) {
    case 0: return ['Start with a question that data can answer. What are you trying to determine?',
      'Is this coin fair? You will flip it ' + N_FLIPS + ' times and count the heads.', 'from scipy import stats'];
    case 1: return ['The null hypothesis H₀ is the default claim of no effect. The alternative H₁ is what you conclude ' +
      'if the data make H₀ hard to believe. H₀ is the claim you try to disprove.',
      'H₀: the coin is fair, P(heads) = 0.5. H₁: the coin is not fair, P(heads) ≠ 0.5.', '# H0: p = 0.5    H1: p != 0.5'];
    case 2: return ['α is your threshold for "unlikely". Choose it before you see the data. It is the chance of a ' +
      'Type I error: rejecting H₀ when H₀ is actually true.', 'α = ' + a + ' (change it with the menu below).', 'alpha = ' + a];
    case 3: return ['Collect the data and reduce them to one number, the test statistic. Which statistic you use ' +
      '(a count, t, chi-square) depends on the data.',
      k + ' heads in ' + N_FLIPS + ' flips. The test statistic is the number of heads, ' + k + '. A fair coin gives about 50.',
      'n_flips, n_heads = ' + N_FLIPS + ', ' + k];
    case 4: return ['The p-value is the probability, calculated assuming H₀ is true, of a result at least as extreme ' +
      'as the one observed. It is not the probability that H₀ is true.',
      'For a fair coin, the chance of a count as extreme as ' + k + ' (' + extreme + ') is p = ' + st.pText + '.',
      'p = stats.binomtest(' + k + ', ' + N_FLIPS + ', p=0.5).pvalue'];
    case 5: return ['Compare the p-value with α. A p-value below α means the data would be unusual if H₀ were true.',
      'Is ' + st.pText + ' < ' + a + '? ' + (st.reject ? 'Yes, so reject H₀.' : 'No, so fail to reject H₀.'),
      'if p < alpha:    # ' + (st.reject ? 'True' : 'False')];
    case 6: return ['Statistically significant: the evidence supports H₁. Check the effect size too, since a real ' +
      'effect can be too small to matter. If H₀ is actually true, this is a Type I error.',
      st.reject ? outcome + ', so reject H₀: the coin appears to be unfair.' : 'Not the outcome for these data: ' + outcome + '.',
      'print("Reject H0")'];
    case 7: return ['Not statistically significant: not enough evidence for H₁. This does not prove H₀ is true. ' +
      'If H₀ is actually false, this is a Type II error (probability β).',
      st.reject ? 'Not the outcome for these data: ' + outcome + '.'
        : outcome + ', so fail to reject H₀: no convincing evidence that the coin is unfair.',
      'print("Fail to reject H0")'];
    default: {
      const ci = confidenceInterval(k, st.alpha), level = round(100 * (1 - st.alpha));
      return ['Report the test statistic, the p-value, the effect size, and a confidence interval, not only ' +
        '"significant" or "not significant".',
        k + ' heads in ' + N_FLIPS + ' flips, exact binomial test, p = ' + st.pText + ', α = ' + a + '. Proportion of heads ' +
        prop + ' (' + nf(Math.abs(k / N_FLIPS - P0), 1, 2) + ' from 0.5). ' + level + '% CI: ' + nf(ci[0], 1, 3) + ' to ' + nf(ci[1], 1, 3) + '.',
        'stats.binomtest(' + k + ', ' + N_FLIPS + ').proportion_ci(' + nf(1 - st.alpha, 1, 2) + ')'];
    }
  }
}

function drawPanel(x, y, w, h, st, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = narrow ? 12 : 15, lh = ts + 4, tx = x + 10, tw = w - 20;
  const [explain, example, code] = stepContent(cur, st);
  const rows = narrow ? [2, 6, 6, 2] : [1, 4, 3, 1];       // text rows kept for title, explanation, example, code
  let cy = y + 9;
  noStroke();
  textAlign(LEFT, TOP);
  fill('midnightblue');
  textStyle(BOLD);
  textSize(ts + 1);
  text('Step ' + (position(cur) + 1) + ' of 8: ' + NODES[cur].label, tx, cy, tw, rows[0] * (lh + 1) + 4);
  cy += rows[0] * (lh + 1) + 6;
  textStyle(NORMAL);
  textSize(ts);
  fill('black');
  text(explain, tx, cy, tw, rows[1] * lh + 4);
  cy += rows[1] * lh + 6;
  fill('saddlebrown');
  text('Coin example: ' + example, tx, cy, tw, rows[2] * lh + 4);
  cy += rows[2] * lh + 8;
  fill('whitesmoke');
  stroke('gainsboro');
  rect(tx - 4, cy - 4, tw + 8, rows[3] * lh + 8, 6);
  noStroke();
  fill('darkslategray');
  text(code, tx, cy, tw, rows[3] * lh + 4);
  cy += rows[3] * lh + 12;
  drawChart(tx, cy, tw, y + h - cy - 6, st, narrow);
}

// Binomial distribution of the number of heads under H0, with the counts that make up the p-value in red
function drawChart(x, y, w, h, st, narrow) {
  const lo = 25, hi = 75, base = y + h - 16, topY = y + 34, barW = w / (hi - lo + 1);
  const peak = pmf(50, P0);
  noStroke();
  textSize(narrow ? 11 : 13);
  textAlign(LEFT, TOP);
  fill('black');
  text('Heads in ' + N_FLIPS + ' flips if H₀ is true', x, y);
  fill('crimson');
  text((narrow ? 'red area: p = ' : 'red bars: at least as extreme as ' + st.k + ', total p = ') + st.pText, x, y + 16);
  for (let i = lo; i <= hi; i++) {
    const bh = pmf(i, P0) / peak * (base - topY);
    fill(asExtreme(i, st.k) ? 'crimson' : 'lightsteelblue');
    rect(x + (i - lo) * barW, base - bh, max(1, barW - 1), bh);
  }
  stroke('gray');
  strokeWeight(1);
  line(x, base, x + w, base);
  // marker at the observed count
  const ox = x + (st.k - lo + 0.5) * barW;
  noStroke();
  fill('black');
  triangle(ox, base - 2, ox - 5, base - 12, ox + 5, base - 12);
  textSize(11);
  textAlign(CENTER, TOP);
  fill('dimgray');
  for (let v = 30; v <= 70; v += 10) text(v, x + (v - lo + 0.5) * barW, base + 3);
}

function nodeAt(x, y) {
  return nodeBoxes.findIndex(b => x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]);
}

function mousePressed() {
  const hit = nodeAt(mouseX, mouseY);
  if (hit >= 0) cur = hit;
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  headsSlider.size(canvasWidth - sliderLeftMargin - margin);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
