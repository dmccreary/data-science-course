// Notebook Cell Types Interactive Demo
// Simulates a Jupyter notebook: code cells and markdown cells
// Bloom Level: Apply (L3) - hands-on cell interaction
// MicroSim template version 2026.02

let canvasWidth = 640;
let drawHeight = 480;
let controlHeight = 50;
let canvasHeight = drawHeight + controlHeight;
let containerWidth;
let containerHeight = canvasHeight;

let margin = 14;
let defaultTextSize = 13;

// Cell types
const MARKDOWN = 'Markdown';
const CODE = 'Code';

let cells = [
  { type: MARKDOWN, source: '# My First Notebook\nWelcome to **data science!**', output: '', executed: false, execNum: null },
  { type: CODE, source: 'x = 42\nprint(f"The answer is {x}")', output: 'The answer is 42', executed: false, execNum: null },
  { type: CODE, source: '# Try editing me!\ny = x * 2\nprint(f"Double: {y}")', output: 'Double: 84', executed: false, execNum: null }
];

let selectedCell = 0;
let execCounter = 0;
let statusMsg = 'Click a cell to select it, then press Run Cell or Shift+Click to execute.';

// Scroll offset for notebook
let scrollY = 0;
let cellH = 90;
let cellSpacing = 8;
let nbLeft, nbRight, nbTop;

// Buttons
let runButton, addCodeButton, addMdButton, deleteButton, changTypeButton;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);
  textSize(defaultTextSize);

  let by = drawHeight + 8;
  let bx = margin;

  runButton = createButton('▶ Run Cell');
  runButton.position(bx, by);
  runButton.mousePressed(runSelectedCell);
  bx += 90;

  addCodeButton = createButton('+ Code Cell');
  addCodeButton.position(bx, by);
  addCodeButton.mousePressed(() => addCell(CODE));
  bx += 100;

  addMdButton = createButton('+ Markdown Cell');
  addMdButton.position(bx, by);
  addMdButton.mousePressed(() => addCell(MARKDOWN));
  bx += 130;

  deleteButton = createButton('✕ Delete');
  deleteButton.position(bx, by);
  deleteButton.mousePressed(deleteSelectedCell);
  bx += 80;

  changTypeButton = createButton('⇄ Change Type');
  changTypeButton.position(bx, by);
  changTypeButton.mousePressed(changeType);

  describe('Interactive Jupyter notebook simulator with code and markdown cells. Click a cell to select, run cells to see output, and add or delete cells.', LABEL);
}

function runSelectedCell() {
  if (selectedCell < 0 || selectedCell >= cells.length) return;
  execCounter++;
  cells[selectedCell].executed = true;
  cells[selectedCell].execNum = execCounter;
  if (cells[selectedCell].type === CODE) {
    statusMsg = '[' + execCounter + '] Code cell executed. Output shown below cell.';
  } else {
    statusMsg = '[' + execCounter + '] Markdown cell rendered.';
  }
}

function addCell(type) {
  let newCell = { type: type, source: type === CODE ? '# New code here\n' : '**New markdown cell**', output: '', executed: false, execNum: null };
  cells.splice(selectedCell + 1, 0, newCell);
  selectedCell = selectedCell + 1;
  statusMsg = 'Added new ' + type + ' cell below selected cell.';
}

function deleteSelectedCell() {
  if (cells.length <= 1) { statusMsg = 'Cannot delete the last cell.'; return; }
  cells.splice(selectedCell, 1);
  selectedCell = Math.min(selectedCell, cells.length - 1);
  statusMsg = 'Cell deleted.';
}

function changeType() {
  if (selectedCell < 0 || selectedCell >= cells.length) return;
  cells[selectedCell].type = cells[selectedCell].type === CODE ? MARKDOWN : CODE;
  cells[selectedCell].executed = false;
  cells[selectedCell].execNum = null;
  statusMsg = 'Changed to ' + cells[selectedCell].type + ' cell. Run to execute.';
}

function mousePressed() {
  // Detect cell click
  let cy = nbTop - scrollY;
  for (let i = 0; i < cells.length; i++) {
    let ch = getCellHeight(cells[i]);
    if (mouseX >= nbLeft && mouseX <= nbRight && mouseY >= cy && mouseY <= cy + ch) {
      selectedCell = i;
      statusMsg = 'Selected ' + cells[i].type + ' cell ' + (i + 1) + '. Press Run Cell to execute.';
      return;
    }
    cy += ch + cellSpacing;
  }
}

function getCellHeight(cell) {
  let lines = cell.source.split('\n').length;
  let h = 16 + lines * 16 + 10;
  if (cell.executed) {
    let outLines = cell.output ? cell.output.split('\n').length : 1;
    h += outLines * 15 + 18;
  }
  return Math.max(h, 60);
}

function draw() {
  updateCanvasSize();
  background(245);

  // Drawing region
  fill(245);
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
  textSize(16);
  text('Jupyter Notebook Cell Types', canvasWidth / 2, margin - 2);

  // Notebook area
  nbLeft = margin + 4;
  nbRight = canvasWidth - margin;
  nbTop = margin + 26;
  let nbWidth = nbRight - nbLeft;

  // Clip to drawing area
  let cellY = nbTop - scrollY;

  for (let i = 0; i < cells.length; i++) {
    let cell = cells[i];
    let ch = getCellHeight(cell);
    let isSelected = (i === selectedCell);

    if (cellY + ch < 0 || cellY > drawHeight - margin) {
      cellY += ch + cellSpacing;
      continue;
    }

    // Cell frame
    if (isSelected) {
      fill(255);
      stroke(70, 130, 200);
      strokeWeight(2.5);
    } else {
      fill(255);
      stroke(210);
      strokeWeight(1);
    }
    rect(nbLeft, cellY, nbWidth, ch, 4);

    // Cell type badge
    noStroke();
    if (cell.type === CODE) {
      fill(240, 240, 255);
    } else {
      fill(255, 245, 230);
    }
    rect(nbLeft, cellY, 60, 18, 4, 0, 0, 0);
    fill(cell.type === CODE ? color(60, 80, 180) : color(180, 100, 0));
    textSize(10);
    textAlign(LEFT, TOP);
    text(cell.type, nbLeft + 4, cellY + 4);

    // Execution number
    if (cell.execNum !== null) {
      fill(100);
      textAlign(RIGHT, TOP);
      textSize(11);
      text('In [' + cell.execNum + ']:', nbLeft + 58, cellY + 4);
    } else {
      fill(160);
      textAlign(RIGHT, TOP);
      textSize(11);
      text('In [ ]:', nbLeft + 58, cellY + 4);
    }

    // Cell source
    fill(30);
    textSize(12);
    textAlign(LEFT, TOP);
    textWrap(WORD);
    let srcLines = cell.source.split('\n');
    let srcY = cellY + 20;
    for (let ln of srcLines) {
      // Simple markdown highlight
      if (cell.type === MARKDOWN && ln.startsWith('#')) fill(20, 60, 180);
      else if (cell.type === CODE && ln.trim().startsWith('#')) fill(100, 140, 100);
      else fill(30);
      text(ln, nbLeft + 68, srcY, nbWidth - 72);
      srcY += 16;
    }

    // Output
    if (cell.executed) {
      let outY = cellY + ch - (cell.output ? cell.output.split('\n').length * 15 + 18 : 20);
      stroke(220);
      strokeWeight(1);
      line(nbLeft + 68, outY - 2, nbRight - 4, outY - 2);
      noStroke();

      if (cell.type === MARKDOWN) {
        // Render "formatted" markdown
        fill(20, 60, 180);
        textSize(14);
        let rendered = cell.source.replace(/\*\*(.*?)\*\*/g, '$1').replace(/# /g, '').replace(/\n/, ' ');
        text(rendered.substring(0, 80), nbLeft + 68, outY + 2, nbWidth - 72);
      } else if (cell.output) {
        fill(0, 140, 0);
        textSize(12);
        textWrap(WORD);
        text(cell.output, nbLeft + 68, outY + 4, nbWidth - 72);
      }
    }

    cellY += ch + cellSpacing;
  }

  // Status bar in control region
  noStroke();
  fill(60);
  textAlign(LEFT, CENTER);
  textSize(12);
  textWrap(WORD);
  text(statusMsg, margin + 5, drawHeight + 8, canvasWidth - margin * 2 - 10, controlHeight - 12);
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  let by = drawHeight + 8;
  let bx = margin;
  runButton.position(bx, by); bx += 90;
  addCodeButton.position(bx, by); bx += 100;
  addMdButton.position(bx, by); bx += 130;
  deleteButton.position(bx, by); bx += 80;
  changTypeButton.position(bx, by);
  redraw();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
