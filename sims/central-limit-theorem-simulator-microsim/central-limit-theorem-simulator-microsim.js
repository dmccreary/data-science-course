// Central Limit Theorem Simulator
// CANVAS_HEIGHT: 585
// Bloom L4 (Analyze): students compare the shape of a population with the shape of the distribution
// of sample means, and examine how the sample size n changes that distribution.
//
// Model: each sample is n independent values from the chosen population. Its mean is one block in
// the lower histogram. The central limit theorem says the sample means are approximately normal with
//   mean = mu   and   standard deviation (standard error) = sigma / sqrt(n),
// where mu and sigma are the population mean and standard deviation. The red curve is that normal
// curve scaled to the number of samples. The observed SD of the sample means uses n - 1 (ddof = 1).
// Sampling uses a seeded generator (mulberry32), so every visit shows the same samples.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 470;
let controlHeight = 115;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;
let labelWidth = 130;       // width of the label span in each control row
let valueWidth = 40;        // width of the value span in the slider row

const SEED = 20240, PRELOAD = 1000, MAX_SAMPLES = 5000, PER_FRAME = 5;
const SIZES = [2, 5, 10, 30, 100];
const npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
// Each population has its exact mean and standard deviation, its density on 0..100, and a sampler
const POPS = [
  { name: 'Normal', short: 'normal', mu: 50, sigma: 15, pdf: x => npdf(x, 50, 15), draw: () => 50 + 15 * stdNormal() },
  { name: 'Uniform', short: 'uniform', mu: 50, sigma: 100 / Math.sqrt(12), pdf: () => 0.01, draw: () => 100 * uniform() },
  { name: 'Exponential (right-skewed)', short: 'right-skewed', mu: 20, sigma: 20, pdf: x => Math.exp(-x / 20) / 20,
    draw: () => -20 * Math.log(uniform()) },
  // half N(25, 8) and half N(75, 8): variance = 8^2 + 25^2
  { name: 'Bimodal', short: 'bimodal', mu: 50, sigma: Math.sqrt(8 * 8 + 25 * 25),
    pdf: x => 0.5 * (npdf(x, 25, 8) + npdf(x, 75, 8)), draw: () => (uniform() < 0.5 ? 25 : 75) + 8 * stdNormal() }
];

let popSelect, sizeRow, sampleButton, startButton, clearButton;
let rngState = SEED;
let running = false;
let count = 0, sum = 0, sumSq = 0;      // number of sample means, their sum, and their sum of squares
let bins = [], binW = 1;                // histogram of the sample means
let lastSample = [], lastMean = 0;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }

function currentPop() { return POPS.find(p => p.name === popSelect.value()); }
function currentN() { return SIZES[sizeRow.slider.value()]; }

// Draw one sample of n values and record its mean
function takeSample() {
  const pop = currentPop(), n = currentN();
  lastSample = Array.from({ length: n }, pop.draw);
  lastMean = lastSample.reduce((a, x) => a + x, 0) / n;
  count++;
  sum += lastMean;
  sumSq += lastMean * lastMean;
  const b = Math.floor(lastMean / binW);
  if (b >= 0 && b < bins.length) bins[b]++;
}

// Start again from the seed with `howMany` samples already taken
function restart(howMany) {
  rngState = SEED;
  running = false;
  startButton.html('Start');
  const se = currentPop().sigma / Math.sqrt(currentN());
  binW = [5, 2.5, 2, 1, 0.5, 0.25].find(wd => wd <= se / 3) || 0.25;     // about 18 bars across 6 standard errors
  bins = new Array(Math.round(100 / binW)).fill(0);
  count = 0; sum = 0; sumSq = 0; lastSample = [];
  for (let i = 0; i < howMany; i++) takeSample();
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  const row = createDiv();
  row.parent(mainElement);
  row.position(10, drawHeight + 8);
  row.style('font-size', '16px');
  const labelSpan = createSpan('Population:');
  labelSpan.parent(row);
  labelSpan.style('display', 'inline-block');
  labelSpan.style('width', labelWidth + 'px');
  popSelect = createSelect();
  popSelect.parent(row);
  POPS.forEach(p => popSelect.option(p.name));
  popSelect.selected('Uniform');
  popSelect.style('font-size', '15px');
  popSelect.changed(() => restart(PRELOAD));

  sizeRow = makeSliderRow('Sample size (n)', 0, SIZES.length - 1, 3, 1, 1);
  sizeRow.slider.input(() => restart(PRELOAD));
  sizeRow.slider.size(max(60, canvasWidth - labelWidth - valueWidth - 40));

  sampleButton = makeButton('Take 1 Sample', 10, () => { if (count < MAX_SAMPLES) takeSample(); });
  startButton = makeButton('Start', 128, () => {
    running = !running && count < MAX_SAMPLES;
    startButton.html(running ? 'Pause' : 'Start');
  });
  clearButton = makeButton('Clear', 190, () => restart(0));
  restart(PRELOAD);

  describe('Two stacked charts on the same axis from 0 to 100. The top chart shows the population distribution and ' +
    'the latest sample. The bottom chart is a histogram of sample means with the normal curve predicted by the ' +
    'central limit theorem. A table compares the mean and standard deviation of the sample means with the ' +
    'population mean and the standard error, sigma divided by the square root of n.', LABEL);
}

function makeButton(label, x, action) {
  const b = createButton(label);
  b.parent(document.querySelector('main'));
  b.position(x, drawHeight + 80);
  b.mousePressed(action);
  return b;
}

// A control row built from a div: the label and value sit in fixed-width spans
function makeSliderRow(label, minValue, maxValue, startValue, step, rowIndex) {
  const row = createDiv();
  row.parent(document.querySelector('main'));
  row.position(10, drawHeight + 8 + rowIndex * 35);
  row.style('font-size', '16px');
  const labelSpan = createSpan(label + ':');
  labelSpan.parent(row);
  labelSpan.style('display', 'inline-block');
  labelSpan.style('width', labelWidth + 'px');
  const valueSpan = createSpan('');
  valueSpan.parent(row);
  valueSpan.style('display', 'inline-block');
  valueSpan.style('width', valueWidth + 'px');
  valueSpan.style('font-weight', 'bold');
  const slider = createSlider(minValue, maxValue, startValue, step);
  slider.parent(row);
  slider.style('vertical-align', 'middle');
  return { slider, valueSpan };
}

function draw() {
  updateCanvasSize();

  // the animation only advances while running
  if (running) {
    for (let i = 0; i < PER_FRAME && count < MAX_SAMPLES; i++) takeSample();
    if (count >= MAX_SAMPLES) { running = false; startButton.html('Start'); }
  }

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const pop = currentPop(), n = currentN(), se = pop.sigma / Math.sqrt(n);
  sizeRow.valueSpan.html(n);
  const narrow = canvasWidth < 600, ts = narrow ? 12 : 15;
  const w = canvasWidth - 2 * margin, px = margin + 44, pw = w - 44 - 16;
  const gx = v => px + constrain(v, 0, 100) / 100 * pw;

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Central Limit Theorem Simulator', canvasWidth / 2, 8);

  // ---- top chart: the population and the latest sample ----
  const aTop = 40, aAxis = 150, bTop = 176, bAxis = 340;
  chartFrame(aTop, aAxis, px, pw, gx, ts);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textSize(ts);
  textStyle(BOLD);
  text('Population (' + pop.short + '): μ = ' + nf(pop.mu, 1, 2) + ', σ = ' + nf(pop.sigma, 1, 2), margin + 10, aTop + 7);
  textStyle(NORMAL);
  let peak = 0;
  for (let x = 0; x <= 100; x++) peak = max(peak, pop.pdf(x));
  const curveH = aAxis - aTop - 34;
  fill(100, 149, 237, 110);
  stroke('royalblue');
  strokeWeight(2);
  beginShape();
  vertex(gx(0), aAxis);
  for (let x = 0; x <= 100; x += 0.5) vertex(gx(x), aAxis - pop.pdf(x) / peak * curveH);
  vertex(gx(100), aAxis);
  endShape(CLOSE);

  // ---- bottom chart: histogram of the sample means with the predicted normal curve ----
  chartFrame(bTop, bAxis, px, pw, gx, ts);
  const expectedPeak = count * binW * npdf(pop.mu, pop.mu, se);
  const yMax = max(5, 1.2 * expectedPeak, 1.1 * max(bins));
  const gy = c => bAxis - c / yMax * (bAxis - bTop - 34);
  const step = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].find(s => yMax / s <= 4);
  textSize(12);
  for (let c = 0; c <= yMax; c += step) {
    stroke('gainsboro');
    strokeWeight(1);
    line(px, gy(c), px + pw, gy(c));
    noStroke();
    fill('dimgray');
    textAlign(RIGHT, CENTER);
    text(c, px - 5, gy(c));
  }
  noStroke();
  fill('steelblue');
  const barW = binW / 100 * pw;
  bins.forEach((c, i) => { if (c > 0) rect(gx(i * binW) + 0.5, gy(c), max(1, barW - 1), bAxis - gy(c)); });
  if (count > 0) {
    noFill();
    stroke('crimson');
    strokeWeight(2.5);
    beginShape();
    for (let x = 0; x <= 100; x += 0.25) vertex(gx(x), gy(count * binW * npdf(x, pop.mu, se)));
    endShape();
  }
  noStroke();
  textSize(ts);
  textStyle(BOLD);
  fill('black');
  textAlign(LEFT, TOP);
  text((narrow ? 'Sample means' : 'Distribution of sample means') + ' (n = ' + n + ')', margin + 10, bTop + 7);
  fill('crimson');
  textAlign(RIGHT, TOP);
  text(narrow ? 'red: normal, SD σ/√n' : 'red curve: normal with mean μ and SD σ/√n', margin + w - 10, bTop + 7);
  textStyle(NORMAL);

  // ---- latest sample: its values on the top chart and its mean carried down to the histogram ----
  if (lastSample.length) {
    stroke('saddlebrown');
    strokeWeight(1);
    fill('orange');
    lastSample.forEach((v, i) => circle(gx(v), aAxis - 5 - (i * 7) % 19, 6));
    stroke('chocolate');
    strokeWeight(2);
    drawingContext.setLineDash([5, 4]);
    line(gx(lastMean), aAxis - 30, gx(lastMean), bAxis);
    drawingContext.setLineDash([]);
    noStroke();
    fill('chocolate');
    triangle(gx(lastMean), aAxis - 22, gx(lastMean) - 6, aAxis - 34, gx(lastMean) + 6, aAxis - 34);
    textStyle(BOLD);
    textSize(ts);
    if (narrow) {             // short label beside the marker, on the side with more room
      const right = gx(lastMean) < px + pw * 0.6;
      textAlign(right ? LEFT : RIGHT, BOTTOM);
      text('mean = ' + nf(lastMean, 1, 1), gx(lastMean) + (right ? 9 : -9), aAxis - 26);
    } else {
      textAlign(RIGHT, TOP);
      text('▼ mean of the ' + n + ' orange dots = ' + nf(lastMean, 1, 1), margin + w - 10, aTop + 7);
    }
    textStyle(NORMAL);
  }

  drawStats(margin, 366, w, drawHeight - 8 - 366, pop, n, se, narrow, ts);
}

// White chart area with a 0..100 axis; both charts share px and pw so their x positions line up
function chartFrame(top, axisY, px, pw, gx, ts) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(margin, top, canvasWidth - 2 * margin, axisY - top + 22, 10);
  textSize(12);
  for (let v = 0; v <= 100; v += 10) {
    stroke('gainsboro');
    line(gx(v), top + 30, gx(v), axisY);
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    text(v, gx(v), axisY + 5);
  }
  stroke('gray');
  strokeWeight(1.5);
  line(px, axisY, px + pw, axisY);
}

// Observed statistics of the sample means next to what the theorem predicts
function drawStats(x, y, w, h, pop, n, se, narrow, ts) {
  fill('lightyellow');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const lh = narrow ? 16 : 20;
  const mean = count ? sum / count : 0;
  const sd = count > 1 ? Math.sqrt(max(0, sumSq - count * mean * mean) / (count - 1)) : 0;
  const predicted = narrow ? 'σ/√n = ' + nf(se, 1, 2) : 'σ/√n = ' + nf(pop.sigma, 1, 2) + '/√' + n + ' = ' + nf(se, 1, 2);
  const rows = [['Samples taken: ' + count, 'Observed', 'CLT predicts'],
    ['Mean of sample means', count ? nf(mean, 1, 2) : '–', 'μ = ' + nf(pop.mu, 1, 2)],
    ['SD of sample means', count > 1 ? nf(sd, 1, 2) : '–', predicted]];
  noStroke();
  textSize(ts);
  rows.forEach((r, i) => {
    textStyle(i === 0 ? BOLD : NORMAL);
    fill('black');
    textAlign(LEFT, TOP);
    text(r[0], x + 12, y + 8 + i * lh);
    textAlign(CENTER, TOP);
    text(r[1], x + w * 0.56, y + 8 + i * lh);
    textAlign(RIGHT, TOP);
    text(r[2], x + w - 12, y + 8 + i * lh);
  });
  fill('dimgray');
  textAlign(LEFT, TOP);
  textWrap(WORD);
  text(count === 0 ? 'Press Take 1 Sample to draw n values and plot their mean. Press Start to keep sampling.'
    : 'Compare the histogram with the red curve. Try n = 2, then n = 30. When does the bell shape appear?',
    x + 12, y + 12 + 3 * lh, w - 24, h - 14 - 3 * lh);
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  sizeRow.slider.size(max(60, canvasWidth - labelWidth - valueWidth - 40));
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
