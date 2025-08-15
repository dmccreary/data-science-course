/**
 * MicroSim: Bell Curve (Normal Distribution) Sampler
 * - Press buttons to drop 1, 10, or 100 samples from N(0,1).
 * - Samples animate from the top, fall into their bin, and increment the histogram.
 * - The theoretical normal PDF is overlaid for visual comparison.
 *
 * Layout: responsive width (fixed height), controls below draw area.
 */

// ---------- Core layout (Standard Rules) ----------
let canvasWidth = 800;                 // updated responsively
let drawHeight = 400;                  // simulation / chart area
let controlHeight = 50;                // controls area
let canvasHeight = drawHeight + controlHeight;
let margin = 25;
let sliderLeftMargin = 105;            // (unused here but kept for consistency)
let defaultTextSize = 16;

let containerWidth;
let containerHeight = canvasHeight;

// ---------- Simulation parameters ----------
const BIN_COUNT = 51;                  // number of histogram bins
const X_MIN = -3;                      // show ±3σ
const X_MAX = 3;
let bins = new Array(BIN_COUNT).fill(0);
let totalSamples = 0;

let particles = [];                    // active falling samples
const g = 0.45;                        // gravity for the falling animation
const particleRadius = 5;

// Controls (buttons)
let drop1Btn, drop10Btn, drop100Btn, resetBtn;

// Cached for drawing
let binWidthPx, plotLeft, plotRight, plotTop, plotBottom;

// ---------- p5 setup ----------
function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textFont('sans-serif');
  rectMode(CORNER);
  noStroke();

  // Buttons in controls area
  drop1Btn = createButton('Drop 1');
  drop1Btn.position(10, drawHeight + 15);
  drop1Btn.mousePressed(() => enqueueDrops(1));

  drop10Btn = createButton('Drop 10');
  drop10Btn.position(80, drawHeight + 15);
  drop10Btn.mousePressed(() => enqueueDrops(10));

  drop100Btn = createButton('Drop 100');
  drop100Btn.position(160, drawHeight + 15);
  drop100Btn.mousePressed(() => enqueueDrops(100));

  resetBtn = createButton('Reset');
  resetBtn.position(250, drawHeight + 15);
  resetBtn.mousePressed(resetSimulation);

  computePlotGeometry();

  // Accessibility
  describe('Bell curve MicroSim: repeatedly sample from a normal distribution. Each sample appears as a dot falling from the top into one of 51 bins spanning minus three to plus three standard deviations. Bars form a histogram as counts accumulate. The theoretical normal curve is overlaid for comparison.', LABEL);
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  computePlotGeometry();
  redraw();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}

// ---------- Main draw loop ----------
function draw() {
  background(255);

  // Draw areas
  drawPanels();

  // Title
  drawTitle();

  // Update & draw particles
  updateParticles();
  drawParticles();

  // Draw histogram & axes
  drawHistogram();

  // Overlay theoretical normal curve
  drawNormalCurveOverlay();

  // Draw controls labels
  drawControlLabels();
}

// ---------- Geometry helpers ----------
function computePlotGeometry() {
  // Plot area inside draw region (leave margins for labels)
  plotLeft   = margin * 1.5;
  plotRight  = canvasWidth - margin;
  plotTop    = margin * 2.25 + 10; // below title
  plotBottom = drawHeight - margin * 1.25;

  binWidthPx = (plotRight - plotLeft) / BIN_COUNT;
}

// ---------- Panels & UI text ----------
function drawPanels() {
  // Draw area
  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);

  // Controls area
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(0, drawHeight, canvasWidth, controlHeight);
}

function drawTitle() {
  fill(0);
  noStroke();
  textAlign(CENTER, TOP);
  textSize(getResponsiveTextSize(24));
  text('Bell Curve MicroSim: Sampling from a Normal Distribution', canvasWidth / 2, margin);

  textAlign(LEFT, TOP);
  textSize(getResponsiveTextSize(12));
  const subtitle = `Bins: ${BIN_COUNT} spanning [${X_MIN}σ, ${X_MAX}σ]   |   Samples: ${totalSamples}`;
  text(subtitle, plotLeft, margin + 30);
}

function drawControlLabels() {
  fill(0);
  noStroke();
  textSize(getResponsiveTextSize(defaultTextSize));
  textAlign(LEFT, CENTER);
  // text('Controls:', 10, drawHeight + 50);
}

// ---------- Particles (falling samples) ----------
function enqueueDrops(n) {
  for (let i = 0; i < n; i++) {
    const z = gaussian01();                // ~N(0,1)
    const xPx = xToPixel(z);
    particles.push({
      x: xPx,
      y: plotTop - 40,                     // start above the plot
      vy: random(0.5, 1.0),
      binIndex: clampBin(zToBin(z)),
      settled: false,
    });
  }
}

function updateParticles() {
  // Compute current bar tops so particles can "land" neatly
  const barHeights = bins.map(c => c); // counts; height scaling happens when drawing
  const maxCount = max(1, ...barHeights);
  const pxPerCount = (plotBottom - plotTop) / maxCount;

  for (let p of particles) {
    if (p.settled) continue;

    // Simple gravity
    p.vy += g * (deltaTime / 1000 * 60);
    p.y += p.vy;

    // Landing threshold: top of this bin's bar (grow upward from bottom)
    const binLeft = plotLeft + p.binIndex * binWidthPx;
    const binRight = binLeft + binWidthPx;
    // current bar height in pixels for this bin
    const currentCount = bins[p.binIndex];
    const barTopY = plotBottom - (currentCount + 1) * pxPerCount * 0.95; // leave a little headroom

    // Consider a particle "landed" when it reaches its bar area
    if (p.y >= barTopY - particleRadius) {
      p.y = barTopY - particleRadius;
      p.settled = true;

      // Increment bin count and total, then remove particle later
      bins[p.binIndex]++;
      totalSamples++;
    }

    // Keep x within its bin visually
    p.x = constrain(p.x, binLeft + 6, binRight - 6);
  }

  // Remove settled particles to keep array small
  particles = particles.filter(p => !p.settled);
}

function drawParticles() {
  fill(30, 90, 200);
  noStroke();
  for (let p of particles) {
    circle(p.x, p.y, particleRadius * 2);
  }
}

// ---------- Histogram & curve ----------
function drawHistogram() {
  // Axes
  stroke(0);
  strokeWeight(1);
  // X axis
  line(plotLeft, plotBottom, plotRight, plotBottom);
  // Y axis
  line(plotLeft, plotTop + 50, plotLeft, plotBottom);

  // X ticks at integer sigmas
  textAlign(CENTER, TOP);
  textSize(getResponsiveTextSize(12));
  noStroke();
  fill(0);
  for (let s = X_MIN; s <= X_MAX; s++) {
    const x = xToPixel(s);
    stroke(0, 80);
    line(x, plotBottom, x, plotBottom + 6);
    noStroke();
    text(`${s}`, x, plotBottom + 8);
  }
  // Y label
  textAlign(LEFT, BOTTOM);
  text('Count Per Bin', plotLeft + 6, plotTop + 55);

  // Bars
  const maxCount = max(1, ...bins);
  const scaleY = (plotBottom - plotTop) / maxCount;

  for (let i = 0; i < BIN_COUNT; i++) {
    const x = plotLeft + i * binWidthPx;
    const h = bins[i] * scaleY * 0.95; // small top gap
    const y = plotBottom - h;

    // Bin rectangle
    noStroke();
    fill(70, 130, 180, 180); // steelblue-ish
    rect(x + 1, y, binWidthPx - 2, h, 3);

    // Optional thin outline
    stroke(255);
    strokeWeight(1);
    line(x + 1, y, x + binWidthPx - 1, y);
  }
}

function drawNormalCurveOverlay() {
  // If no samples yet, draw a faint reference curve scaled to the plot height
  const maxCount = max(1, ...bins);
  const targetPeak = maxCount * 0.95;

  // Compute scaling: PDF peak for N(0,1) at x=0 is 1/sqrt(2π)
  const pdfPeak = 1 / sqrt(TWO_PI); // ≈ 0.39894
  const scaleY = (plotBottom - plotTop) / (pdfPeak === 0 ? 1 : pdfPeak);
  const desiredScale = (targetPeak * (plotBottom - plotTop) / max(1, maxCount)) / pdfPeak;

  stroke('blue');
  strokeWeight(2);
  noFill();
  beginShape();
  const steps = 400;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const xVal = lerp(X_MIN, X_MAX, t);
    const pdf = normalPDF(xVal, 0, 1);     // height in probability units
    // Map pdf to pixels: scale so its peak aligns roughly with current histogram peak
    const yPx = map(pdf * desiredScale, 0, (plotBottom - plotTop), plotBottom, plotTop);
    const xPx = xToPixel(xVal);
    vertex(xPx, yPx);
  }
  endShape();

  // Legend
  noStroke();
  fill('blue');
  textSize(getResponsiveTextSize(12));
  textAlign(LEFT, CENTER);
  text('Theoretical Normal Curve (scaled to histogram)', plotLeft, plotTop + 15);
}

// ---------- Utilities ----------
function getResponsiveTextSize(base) {
  return constrain(base * (containerWidth / 800), base * 0.8, base * 1.5);
}

function gaussian01() {
  // Box–Muller transform: returns ~N(0,1)
  let u = 0, v = 0;
  while (u === 0) u = random(); // avoid 0
  while (v === 0) v = random();
  return sqrt(-2.0 * Math.log(u)) * cos(TWO_PI * v);
}

function normalPDF(x, mu = 0, sigma = 1) {
  const z = (x - mu) / sigma;
  return (1 / (sigma * sqrt(TWO_PI))) * exp(-0.5 * z * z);
}

function xToPixel(x) {
  const t = (x - X_MIN) / (X_MAX - X_MIN);
  return plotLeft + t * (plotRight - plotLeft);
}

function zToBin(z) {
  const t = (z - X_MIN) / (X_MAX - X_MIN);
  return floor(t * BIN_COUNT);
}

function clampBin(i) {
  return constrain(i, 0, BIN_COUNT - 1);
}

// ---------- Reset ----------
function resetSimulation() {
  bins = new Array(BIN_COUNT).fill(0);
  totalSamples = 0;
  particles = [];
}

// ---------- End ----------
