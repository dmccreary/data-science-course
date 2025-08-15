/* Dice Distribution MicroSim (p5.js) — Left-In, 5-Per-Row Stacking
   - All dice spawn at the left edge and slide right before falling.
   - Each slot stacks dice in rows of 5 per row (left→right, bottom→up).
   - Buttons: Roll 1, 10, 100 + Start/Pause + Reset.
   - Responsive width, fixed vertical layout (per standard rules).
*/

// ===== Layout / standard rules =====
let canvasWidth = 800;          // Will be updated responsively
let drawHeight = 420;           // Tall area for sim
let controlHeight = 90;         // Controls region height
let canvasHeight = drawHeight + controlHeight;
let containerWidth;
let containerHeight = canvasHeight;

let margin = 30;
let defaultTextSize = 16;
let sliderLeftMargin = 105; // (not used but kept for consistency)

// ===== Controls =====
let roll1Btn, roll10Btn, roll100Btn, startBtn, resetBtn;
let isRunning = true;

// ===== Simulation state =====
const NUM_SLOTS = 6;
let slotSettledCounts = new Array(NUM_SLOTS).fill(0);   // how many are fully settled
let slotReservedCounts = new Array(NUM_SLOTS).fill(0);  // "seat reservations" when a die begins to fall
let dice = []; // active + settled dice

// Slot geometry (recomputed on resize)
let slotCenters = []; // x centers for labels
let slotLefts = [];   // left-inner x for each slot (for packing)
let slotWidth;
let innerPad = 6;     // inner padding within slot band
let groundY;          // baseline for the first row (bottom row)

// Die visuals & motion
let dieSize = 24;     // will be adapted so 5 columns fit
let gap = 4;          // horizontal gap between dice in-row
let gravity = 0.5;
let slideSpeed = 8;

// Accessibility text
const description = 'Dice distribution simulator with six labeled slots. All dice enter from the left, slide into the correct slot by face value, then drop into a grid that packs five dice per row, forming a histogram.';

// ===== Die class =====
class Die {
  constructor(value, slotIdx) {
    this.value = value;
    this.slotIdx = slotIdx;

    // Start off-canvas at the left edge
    this.x = -dieSize;
    this.y = margin + 80;     // travel lane near the top
    this.vx = slideSpeed;
    this.vy = 0;

    this.phase = 'slide';     // 'slide' → 'fall' → 'settled'
    this.settled = false;

    // Will be computed when we begin to fall
    this.landingIndex = null; // 0-based index within this slot
    this.targetX = null;
    this.targetY = null;
  }

  // Compute the packed (x,y) center for a given slot + landing index
  static landingXYForIndex(slotIdx, index) {
    const col = index % 5;                // 0..4
    const row = Math.floor(index / 5);    // 0..∞ from bottom
    const left = slotLefts[slotIdx];

    // Horizontal position: left inner + half die + col steps
    const x = left + innerPad + (dieSize / 2) + col * (dieSize + gap);

    // Vertical position: groundY - row*dieSize (centers)
    const y = groundY - row * dieSize;

    return { x, y };
  }

  update() {
    if (this.settled) return;

    if (this.phase === 'slide') {
      // If we haven't reserved a landing spot yet, point the slide toward
      // the current "next" open seat's x position (row/col computed at fall time).
      // We slide horizontally until we're roughly aligned with our eventual column.
      // We use the *current* reserved count as a proxy for which column we'll occupy.
      const peekIndex = slotReservedCounts[this.slotIdx]; // not yet claimed
      const peekXY = Die.landingXYForIndex(this.slotIdx, peekIndex);
      const dx = peekXY.x - this.x;

      // Move rightwards until we're aligned horizontally
      const step = constrain(dx, -slideSpeed, slideSpeed);
      this.x += step;

      // When close enough in x, transition to falling
      if (abs(dx) <= 0.8) {
        // Lock in our landing "seat" now to avoid collisions with concurrent dice
        this.landingIndex = slotReservedCounts[this.slotIdx];
        slotReservedCounts[this.slotIdx] += 1;

        const { x, y } = Die.landingXYForIndex(this.slotIdx, this.landingIndex);
        this.targetX = x;   // already aligned, but store for completeness
        this.targetY = y;

        // Begin fall
        this.phase = 'fall';
        this.vy = 0;
        // Keep x fixed during fall (looks neat like dropping into the column)
        this.x = this.targetX;
      }
    } else if (this.phase === 'fall') {
      // Gravity-based drop until we hit our row's surface
      this.vy += gravity * (deltaTime / 16.67);
      this.y += this.vy;

      if (this.y >= this.targetY) {
        this.y = this.targetY;
        this.vy = 0;
        this.phase = 'settled';
        this.settled = true;
        slotSettledCounts[this.slotIdx] += 1;
      }
    }
  }

  draw() {
    push();
    translate(this.x, this.y);
    drawDieFace(this.value, dieSize);
    pop();
  }
}

// ===== Die face drawing =====
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

// ===== Layout helpers =====
function updateCanvasSize() {
  const container = document.querySelector('main')?.getBoundingClientRect() || { width: canvasWidth };
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;

  slotWidth = (canvasWidth - 2 * margin) / NUM_SLOTS;

  // Compute inner layout for 5 dice per row
  // Ensure dieSize fits five across with small gaps
  const innerWidth = slotWidth - 2 * innerPad;
  const maxDie = Math.floor((innerWidth - 4 * gap) / 5); // 5 dice, 4 gaps
  dieSize = constrain(maxDie, 14, 32);                   // keep readable & neat

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
  roll1Btn = createButton('Roll 1 Die');
  roll1Btn.position(10, drawHeight + 12);
  roll1Btn.mousePressed(() => rollDice(1));

  roll10Btn = createButton('Roll 10 Die');
  roll10Btn.position(110, drawHeight + 12);
  roll10Btn.mousePressed(() => rollDice(10));

  roll100Btn = createButton('Roll 100 Die');
  roll100Btn.position(230, drawHeight + 12);
  roll100Btn.mousePressed(() => rollDice(100));

  startBtn = createButton('Pause');
  startBtn.position(370, drawHeight + 12);
  startBtn.mousePressed(toggleRun);

  resetBtn = createButton('Reset');
  resetBtn.position(440, drawHeight + 12);
  resetBtn.mousePressed(resetSimulation);

  // Accessibility
  describe(description, LABEL);
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
  drawTitle('Dice Distribution Simulator');

  // Slots & labels
  drawSlots();

  // Stats
  drawStats();

  // Update & draw dice
  if (isRunning) {
    for (const d of dice) d.update();
  }
  for (const d of dice) d.draw();
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

  // Baseline line
  line(margin, groundY + dieSize / 2 + 6, canvasWidth - margin, groundY + dieSize / 2 + 6);

  // Vertical guides & slot bands
  for (let i = 0; i < NUM_SLOTS; i++) {
    const xLeft = slotLefts[i];
    const xRight = xLeft + slotWidth;

    // slot boundary
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
  // Counts above slot; percentages below the count
  textAlign(CENTER, BOTTOM);
  textSize(getResponsiveTextSize(12));
  noStroke();
  fill(50);

  const total = slotSettledCounts.reduce((a, b) => a + b, 0);
  for (let i = 0; i < NUM_SLOTS; i++) {
    const cx = slotCenters[i];
    const count = slotSettledCounts[i];

    text(count, cx, margin + 60);
    if (total > 0) {
      const p = (count / total) * 100;
      text(p.toFixed(0) + '%', cx, margin + 80);
    }
  }

  // Summary (left)
  textAlign(LEFT, CENTER);
  textSize(getResponsiveTextSize(14));
  const summary = `Total dice: ${total}`;
  text(summary, margin, margin + 25);

  // Expectation note (right)
  textAlign(RIGHT, CENTER);
  text('Expected uniform distribution (≈16.7% each)', canvasWidth - margin, margin + 25);
}

// ===== Roll logic =====
function rollDice(n) {
  if (!isRunning) toggleRun();
  for (let i = 0; i < n; i++) {
    const val = 1 + Math.floor(Math.random() * 6);
    const slotIdx = val - 1;
    const d = new Die(val, slotIdx);
    dice.push(d);
  }
}

// ===== Controls =====
function toggleRun() {
  isRunning = !isRunning;
  startBtn.html(isRunning ? 'Pause' : 'Start');
}

function resetSimulation() {
  dice = [];
  slotSettledCounts = new Array(NUM_SLOTS).fill(0);
  slotReservedCounts = new Array(NUM_SLOTS).fill(0);
  isRunning = true;
  startBtn.html('Pause');
}

// ===== Utilities =====
function getResponsiveTextSize(baseSize) {
  return constrain(baseSize * (canvasWidth / 800), baseSize * 0.8, baseSize * 1.5);
}
