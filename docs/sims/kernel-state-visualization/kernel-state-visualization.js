// Kernel State Visualization
// Step-through showing how the kernel accumulates variables
// Bloom Level: Understand (L2) - step-through pattern
// MicroSim template version 2026.02

let canvasWidth = 640;
let drawHeight = 420;
let controlHeight = 45;
let canvasHeight = drawHeight + controlHeight;
let containerWidth;
let containerHeight = canvasHeight;

let margin = 18;
let defaultTextSize = 14;

// Steps: each step corresponds to running one cell
let steps = [
  {
    cellCode: 'name = "Alice"',
    newVar: { name: 'name', value: '"Alice"', type: 'str' },
    desc: 'Cell 1 runs. Python evaluates name = "Alice" and stores it in kernel memory.'
  },
  {
    cellCode: 'age = 25',
    newVar: { name: 'age', value: '25', type: 'int' },
    desc: 'Cell 2 runs. Python stores age = 25. Both name and age now exist in kernel memory simultaneously.'
  },
  {
    cellCode: 'greeting = f"Hello {name}, you are {age}"',
    newVar: { name: 'greeting', value: '"Hello Alice, you are 25"', type: 'str' },
    desc: 'Cell 3 uses both name and age from memory to build a new string and stores it as greeting.'
  },
  {
    cellCode: 'print(greeting)',
    newVar: null,
    output: 'Hello Alice, you are 25',
    desc: 'Cell 4 reads greeting from memory and prints it. The kernel already has all the variables it needs!'
  }
];

let currentStep = -1; // -1 = initial state, no cells run
let kernelVars = []; // accumulated variables
let flashVar = null; // highlight newly added var
let flashTimer = 0;
let stepButton, resetButton;

let allCells = [
  'name = "Alice"',
  'age = 25',
  'greeting = f"Hello {name}, you are {age}"',
  'print(greeting)'
];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);
  textSize(defaultTextSize);

  stepButton = createButton('▶ Run Next Cell');
  stepButton.position(margin, drawHeight + 8);
  stepButton.mousePressed(runNextStep);

  resetButton = createButton('⟳ Restart Kernel');
  resetButton.position(margin + 140, drawHeight + 8);
  resetButton.mousePressed(restartKernel);

  describe('Step-through visualization of Jupyter kernel memory. Run each cell and watch variables accumulate in the kernel memory bank on the right side. Restart Kernel clears all variables.', LABEL);
}

function runNextStep() {
  if (currentStep >= steps.length - 1) {
    stepButton.html('All cells run! Press Restart Kernel to reset.');
    return;
  }
  currentStep++;
  let s = steps[currentStep];
  if (s.newVar) {
    kernelVars.push({ ...s.newVar });
    flashVar = s.newVar.name;
    flashTimer = 60;
  }
}

function restartKernel() {
  currentStep = -1;
  kernelVars = [];
  flashVar = null;
  flashTimer = 0;
  stepButton.html('▶ Run Next Cell');
}

function draw() {
  updateCanvasSize();
  if (flashTimer > 0) flashTimer--;

  background(248);

  // Drawing region border
  fill(248);
  stroke(200);
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);

  // Control region
  fill(255);
  rect(0, drawHeight, canvasWidth, controlHeight);

  // Title
  noStroke();
  fill(30);
  textAlign(CENTER, TOP);
  textSize(17);
  text('How the Kernel Remembers Variables', canvasWidth / 2, margin - 4);

  // Layout: left=notebook cells, right=kernel memory
  let divX = canvasWidth * 0.52;
  let panelTop = margin + 26;
  let panelH = drawHeight - panelTop - margin;
  let cellW = divX - margin - 8;
  let memX = divX + 8;
  let memW = canvasWidth - memX - margin;

  // --- Notebook panel header ---
  noStroke();
  fill(60, 90, 160);
  rect(margin, panelTop, cellW, 24, 4, 4, 0, 0);
  fill(255);
  textAlign(CENTER, CENTER);
  textSize(12);
  text('Notebook Cells', margin + cellW / 2, panelTop + 12);

  // --- Notebook cells ---
  let cellAreaTop = panelTop + 28;
  let singleCellH = (panelH - 28) / allCells.length - 4;

  for (let i = 0; i < allCells.length; i++) {
    let cy = cellAreaTop + i * (singleCellH + 4);
    let isRun = (i <= currentStep);
    let isCurrent = (i === currentStep);

    if (isCurrent) {
      fill(255, 250, 220);
      stroke(220, 160, 0);
      strokeWeight(2);
    } else if (isRun) {
      fill(240, 250, 240);
      stroke(120, 180, 120);
      strokeWeight(1);
    } else {
      fill(252, 252, 252);
      stroke(210);
      strokeWeight(1);
    }
    rect(margin, cy, cellW, singleCellH, 4);

    // Execution number
    noStroke();
    fill(isRun ? color(60, 100, 60) : color(160));
    textAlign(LEFT, TOP);
    textSize(11);
    text(isRun ? 'In [' + (i + 1) + ']:' : 'In [ ]:', margin + 4, cy + 4);

    // Code
    fill(isRun ? color(30) : color(120));
    textSize(12);
    textAlign(LEFT, TOP);
    textWrap(WORD);
    text(allCells[i], margin + 58, cy + 4, cellW - 64);

    // Output for print cell
    if (i === 3 && i <= currentStep && steps[3].output) {
      fill(0, 140, 0);
      textSize(11);
      text('Out: ' + steps[3].output, margin + 58, cy + singleCellH - 18, cellW - 64);
    }

    // Arrow indicator if current
    if (isCurrent) {
      fill(220, 160, 0);
      noStroke();
      triangle(margin - 2, cy + singleCellH / 2, margin - 10, cy + singleCellH / 2 - 6, margin - 10, cy + singleCellH / 2 + 6);
    }
  }

  // --- Kernel memory panel ---
  fill(30, 30, 50);
  noStroke();
  rect(memX, panelTop, memW, 24, 4, 4, 0, 0);
  fill(255);
  textAlign(CENTER, CENTER);
  textSize(12);
  text('Kernel Memory', memX + memW / 2, panelTop + 12);

  // Memory slots
  fill(38, 38, 55);
  noStroke();
  rect(memX, panelTop + 24, memW, panelH - 24, 0, 0, 4, 4);

  let slotH = 42;
  let slotMargin = 8;
  let maxSlots = Math.floor((panelH - 24 - slotMargin * 2) / (slotH + 6));

  if (kernelVars.length === 0) {
    fill(100);
    textAlign(CENTER, CENTER);
    textSize(12);
    text('(empty — run a cell to add variables)', memX + memW / 2, panelTop + 24 + (panelH - 24) / 2);
  } else {
    for (let j = 0; j < kernelVars.length; j++) {
      let v = kernelVars[j];
      let vy = panelTop + 24 + slotMargin + j * (slotH + 6);
      let isNew = (v.name === flashVar && flashTimer > 0);

      let slotAlpha = isNew ? map(flashTimer, 0, 60, 200, 255) : 200;
      fill(isNew ? color(80, 200, 120, slotAlpha) : color(60, 80, 130, 180));
      rect(memX + slotMargin, vy, memW - slotMargin * 2, slotH, 6);

      noStroke();
      fill(255);
      textAlign(LEFT, TOP);
      textSize(11);
      text(v.name, memX + slotMargin + 8, vy + 6);
      fill(220, 220, 100);
      textAlign(LEFT, TOP);
      textSize(13);
      text(v.value, memX + slotMargin + 8, vy + 22);
      fill(160);
      textAlign(RIGHT, TOP);
      textSize(10);
      text(v.type, memX + memW - slotMargin - 6, vy + 6);

      // Flash arrow from cell to memory
      if (isNew && currentStep >= 0 && currentStep < allCells.length) {
        let cellIdx = currentStep;
        let cy2 = cellAreaTop + cellIdx * (singleCellH + 4) + singleCellH / 2;
        let arrowAlpha = map(flashTimer, 0, 60, 0, 255);
        stroke(255, 220, 50, arrowAlpha);
        strokeWeight(2);
        line(margin + cellW, cy2, memX + slotMargin, vy + slotH / 2);
        noStroke();
        fill(255, 220, 50, arrowAlpha);
        let ax = memX + slotMargin - 1;
        let ay = vy + slotH / 2;
        triangle(ax, ay, ax - 8, ay - 5, ax - 8, ay + 5);
      }
    }
  }

  // Description / status
  noStroke();
  let stepDesc = currentStep >= 0 ? steps[currentStep].desc : 'Press "Run Next Cell" to step through the notebook and watch variables accumulate in kernel memory.';
  fill(50);
  textAlign(LEFT, CENTER);
  textSize(12);
  textWrap(WORD);
  text(stepDesc, margin + 300, drawHeight + 4, canvasWidth - margin * 2 - 310, controlHeight - 8);
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  stepButton.position(margin, drawHeight + 8);
  resetButton.position(margin + 140, drawHeight + 8);
  redraw();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
