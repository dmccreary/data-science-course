/* Two-Dice Sum Distribution MicroSim (p5.js)
   - 12 slots labeled 1..12 (pairs land in 2..12)
   - Each event is a PAIR of dice moving together from top-left
   - Packing: two pairs per row per slot (so 4 dice per row), bottom-up
   - Buttons: Roll 1/10/100 Pairs, Pause/Start, Reset
   - Responsive width; fixed vertical layout per standard rules
*/

// ===== Layout / standard rules =====
let canvasWidth = 900;          // Will be updated responsively
let drawHeight = 450;           // Tall area for sim
let controlHeight = 90;         // Controls region height
let canvasHeight = drawHeight + controlHeight;
let containerWidth;
let containerHeight = canvasHeight;

let margin = 30;
let defaultTextSize = 16;

// ===== Controls =====
let roll1Btn, roll10Btn, roll100Btn, startBtn, resetBtn;
let isRunning = true;

// ===== Simulation state =====
const NUM_SLOTS = 12;                      // 1..12 displayed; 2..12 used
let slotSettledPairs = new Array(NUM_SLOTS).fill(0);   // settled PAIRS per slot
let slotReservedPairs = new Array(NUM_SLOTS).fill(0);  // reservations as pairs
let pairs = [];                                           // active + settled pairs

// Slot geometry (recomputed on resize)
let slotCenters = []; // x centers for labels
let slotLefts = [];   // left x per slot
let slotWidth;
let innerPad = 8;     // inner padding inside slot band
let groundY;          // baseline center for bottom row

// Die visuals & motion
let dieSize = 22;     // computed to fit 4 dice per row (2 pairs)
let gap = 6;          // general spacing
let pairGap = 6;      // spacing between the two dice in a pair
let seatGap = 12;     // spacing between the two pair-seats in a row
let gravity = 0.55;
let slideSpeed = 9;

// ===== Dice face drawing =====
function drawDieFace(val, s) {
  rectMode(CENTER);
  stroke(40);
  fill(250);
  strokeWeight(2);
  const r = 6;
  rect(0, 0, s, s, r);

  // pip positions
  const d = s * 0.25;
  const pips = {
    1: [[0, 0]],
    2: [[-d, -d], [d, d]],
    3: [[-d, -d], [0, 0], [d, d]],
    4: [[-d, -d], [d, -d], [-d, d], [d, d]],
    5: [[-d, -d], [d, -d], [0, 0], [-d, d], [d, d]],
    6: [[-d, -d], [d, -d], [-d, 0], [d, 0], [-d, d], [d, d]]
  };
  noStroke();
  fill(20);
  for (const [px, py] of pips[val]) {
    circle(px, py, s * 0.14);
  }
}

// ===== Pair class (two dice moving together) =====
class Pair {
  constructor(v1, v2, slotIdx) {
    this.v1 = v1;
    this.v2 = v2;
    this.sum = v1 + v2;
    this.slotIdx = slotIdx;

    // Start off-canvas at top-left; two dice side-by-side
    this.baseX = -2 * dieSize;           // anchor/center for pair motion
    this.y = margin + 85;                // travel lane
    this.vy = 0;
    this.phase = 'slide';                // 'slide' -> 'fall' -> 'settled'
    this.settled = false;

    // Landing seat (as PAIRS). Each seat holds exactly one pair.
    this.seatIndex = null;   // 0-based across this slot counting pairs
    this.targetX = null;     // x-center for the *pair* (midpoint between dice)
    this.targetY = null;     // y-center for the pair row (dice share same y)
  }

  // Given seat index (0..N), returns the pair center (x,y).
  // Two seats per row => col = index % 2; row = floor(index / 2).
  static seatCenterXY(slotIdx, index) {
    const col = index % 2;                 // 0 or 1 (left seat, right seat)
    const row = Math.floor(index / 2);     // 0.. up from bottom
    const left = slotLefts[slotIdx];

    // Total width per row with two pairs:
    // width = 4*dieSize + (pair gaps inside pairs: 2*pairGap) + seatGap
    // We'll place the left pair center and right pair center accordingly.
    const innerLeft = left + innerPad;
    const innerRight = left + slotWidth - innerPad;
    const innerWidth = innerRight - innerLeft;

    // Compute seat centers evenly: left seat center and right seat center.
    // Left pair center x:
    const rowTotalWidth = 4*dieSize + 2*pairGap + seatGap;
    let leftCenterX = innerLeft + (rowTotalWidth / 2) - (rowTotalWidth / 2) + (2*dieSize + pairGap)/2;
    // Align row horizontally centered within inner area:
    const leftover = innerWidth - rowTotalWidth;
    leftCenterX = innerLeft + leftover/2 + (2*dieSize + pairGap)/2;

    const rightCenterX = leftCenterX + (2*dieSize + pairGap) + seatGap;

    const pairCenterX = (col === 0) ? leftCenterX : rightCenterX;

    // Vertical: centers are spaced by dieSize per row
    const y = groundY - row * dieSize;

    return { x: pairCenterX, y };
  }

  // Given pair center (x), return the individual die centers (x1, x2) side-by-side
  static diceCentersFromPairCenter(px) {
    const halfSpacing = (dieSize + pairGap) / 2;
    return [px - halfSpacing, px + halfSpacing];
  }

  update() {
    if (this.settled) return;

    if (this.phase === 'slide') {
      // Aim for the *next* open seat's x (pair center)
      const peekSeat = slotReservedPairs[this.slotIdx];
      const peekXY = Pair.seatCenterXY(this.slotIdx, peekSeat);
      const dx = peekXY.x - this.baseX;

      const step = constrain(dx, -slideSpeed, slideSpeed);
      this.baseX += step;

      if (abs(dx) <= 0.8) {
        // Reserve our seat (as a PAIR)
        this.seatIndex = slotReservedPairs[this.slotIdx];
        slotReservedPairs[this.slotIdx] += 1;

        const { x, y } = Pair.seatCenterXY(this.slotIdx, this.seatIndex);
        this.targetX = x;
        this.targetY = y;

        // Lock x and start falling
        this.baseX = this.targetX;
        this.phase = 'fall';
        this.vy = 0;
      }
    } else if (this.phase === 'fall') {
      this.vy += gravity * (deltaTime / 16.67);
      this.y += this.vy;

      if (this.y >= this.targetY) {
        this.y = this.targetY;
        this.vy = 0;
        this.phase = 'settled';
        this.settled = true;
        slotSettledPairs[this.slotIdx] += 1; // 1 more PAIR in this slot
      }
    }
  }

  draw() {
    // Draw the two dice at their horizontal offsets from baseX
    const [x1, x2] = Pair.diceCentersFromPairCenter(this.baseX);
    push();
      translate(x1, this.y);
      drawDieFace(this.v1, dieSize);
    pop();
    push();
      translate(x2, this.y);
      drawDieFace(this.v2, dieSize);
    pop();
  }
}

// ===== Layout helpers =====
function updateCanvasSize() {
  const container = document.querySelector('main')?.getBoundingClientRect() || { width: canvasWidth };
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;

  slotWidth = (canvasWidth - 2 * margin) / NUM_SLOTS;

  // Compute dieSize to fit 2 pairs (4 dice) horizontally within inner area:
  // rowTotalWidth = 4*dieSize + 2*pairGap + seatGap  <= innerWidth
  const innerWidth = slotWidth - 2 * innerPad;
  const maxDie = Math.floor((innerWidth - (2*pairGap + seatGap)) / 4);
  dieSize = constrain(maxDie, 14, 34);

  // Centers & lefts for each slot
  slotCenters = [];
  slotLefts = [];
  for (let i = 0; i < NUM_SLOTS; i++) {
    const left = margin + i * slotWidth;
    slotLefts.push(left);
    slotCenters.push(left + slotWidth / 2);
  }

  // Baseline for the first (bottom) row center
  groundY = drawHeight - margin - 10 - dieSize / 2;
}

// ===== Init / UI =====
function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textFont('sans-serif');

  // Buttons
  roll1Btn = createButton('Roll 1 Pair');
  roll1Btn.position(10, drawHeight + 12);
  roll1Btn.mousePressed(() => rollPairs(1));

  roll10Btn = createButton('Roll 10 Pairs');
  roll10Btn.position(110, drawHeight + 12);
  roll10Btn.mousePressed(() => rollPairs(10));

  roll100Btn = createButton('Roll 100 Pairs');
  roll100Btn.position(230, drawHeight + 12);
  roll100Btn.mousePressed(() => rollPairs(100));

  startBtn = createButton('Pause');
  startBtn.position(390, drawHeight + 12);
  startBtn.mousePressed(toggleRun);

  resetBtn = createButton('Reset');
  resetBtn.position(460, drawHeight + 12);
  resetBtn.mousePressed(resetSimulation);

  // Accessibility
  const description = 'Dice distribution simulator with twelve labeled slots. Two dice enter side-by-side from the top left, slide to the slot matching their sum, then drop as a pair. Each slot packs two pairs per row, forming a triangular distribution peaking at 7.';
  describe(description, p5.LABEL);
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  redraw();
}

// ===== Main draw loop =====
function draw() {
  background(255);

  // Panels
  drawPanels();

  // Title
  drawTitle('Sum of Two Dice — Distribution');

  // Slots & labels
  drawSlots();

  // Stats
  drawStats();

  // Update & draw pairs
  if (isRunning) {
    for (const p of pairs) p.update();
  }
  for (const p of pairs) p.draw();
}

// ===== Panels / framing =====
function drawPanels() {
  // Drawing area
  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);

  // Controls
  fill('white');
  stroke('silver');
  rect(0, drawHeight, canvasWidth, controlHeight);
}

function drawTitle(t) {
  noStroke();
  fill(0);
  textAlign(CENTER, TOP);
  textSize(getResponsiveTextSize(24));
  text(t, canvasWidth / 2, margin * 0.6);
  textAlign(LEFT, CENTER);
  textSize(getResponsiveTextSize(defaultTextSize));
}

function drawSlots() {
  stroke(160);
  strokeWeight(1);

  // Baseline
  line(margin, groundY + dieSize / 2 + 6, canvasWidth - margin, groundY + dieSize / 2 + 6);

  // Slot bands, boundaries, and labels
  for (let i = 0; i < NUM_SLOTS; i++) {
    const xLeft = slotLefts[i];
    const xRight = xLeft + slotWidth;

    // left boundary
    stroke(210);
    line(xLeft, margin + 70, xLeft, groundY + dieSize / 2 + 6);

    // subtle band
    noFill();
    stroke(230);
    rectMode(CORNER);
    rect(xLeft + innerPad, margin + 80, slotWidth - 2 * innerPad, groundY - (margin + 80) + dieSize / 2 - 6);

    // label (centered under baseline)
    noStroke();
    fill(30);
    textAlign(CENTER, TOP);
    textSize(getResponsiveTextSize(14));
    text(String(i + 1), (xLeft + xRight) / 2, groundY + dieSize / 2 + 10);
  }
}

function drawStats() {
  // Show counts (pairs) above slots and relative freqs by PAIRS
  textAlign(CENTER, BOTTOM);
  textSize(getResponsiveTextSize(12));
  noStroke();
  fill(50);

  const totalPairs = slotSettledPairs.reduce((a, b) => a + b, 0);
  for (let i = 0; i < NUM_SLOTS; i++) {
    const cx = slotCenters[i];
    const countPairs = slotSettledPairs[i];

    text(countPairs, cx, margin + 60);
    if (totalPairs > 0) {
      const p = (countPairs / totalPairs) * 100;
      text(p.toFixed(0) + '%', cx, margin + 80);
    }
  }

  // Summary (left)
  textAlign(LEFT, CENTER);
  textSize(getResponsiveTextSize(14));
  const summary = `Total pairs: ${totalPairs}`;
  text(summary, margin, margin + 25);

  // Expectation note (right)
  textAlign(RIGHT, CENTER);
  text('Expected triangular distribution: peak at 7 (6/36 ≈ 16.7%)', canvasWidth - margin, margin + 25);
}

// ===== Roll logic (generate PAIRS) =====
function rollPairs(n) {
  if (!isRunning) toggleRun();
  for (let i = 0; i < n; i++) {
    const v1 = 1 + Math.floor(Math.random() * 6);
    const v2 = 1 + Math.floor(Math.random() * 6);
    const sum = v1 + v2;          // 2..12
    const slotIdx = sum - 1;      // 1..11 mapped to 1..11; index 0 is slot "1" (unused)
    pairs.push(new Pair(v1, v2, slotIdx));
  }
}

// ===== Controls =====
function toggleRun() {
  isRunning = !isRunning;
  startBtn.html(isRunning ? 'Pause' : 'Start');
}

function resetSimulation() {
  pairs = [];
  slotSettledPairs = new Array(NUM_SLOTS).fill(0);
  slotReservedPairs = new Array(NUM_SLOTS).fill(0);
  isRunning = true;
  startBtn.html('Pause');
}

// ===== Utilities =====
function getResponsiveTextSize(baseSize) {
  return constrain(baseSize * (canvasWidth / 900), baseSize * 0.8, baseSize * 1.5);
}
