// VS Code Interface Tour
// Interactive labeled diagram of VS Code interface components
// Bloom Level: Remember (L1) - hover labels + quiz mode
// MicroSim template version 2026.02

let canvasWidth = 700;
let drawHeight = 460;
let controlHeight = 40;
let canvasHeight = drawHeight + controlHeight;
let containerWidth;
let containerHeight = canvasHeight;

let margin = 16;
let defaultTextSize = 14;

let quizMode = false;
let quizButton;
let hoveredRegion = -1;

// VS Code color palette
const bg = [30, 30, 30];
const activityBarColor = [51, 51, 51];
const sideBarColor = [37, 37, 38];
const editorColor = [30, 30, 30];
const tabBarColor = [44, 44, 44];
const panelColor = [30, 30, 30];
const statusBarColor = [0, 122, 204];
const minimapColor = [38, 38, 38];
const lineNumColor = [90, 90, 90];

// Regions: {label, desc, color, x,y,w,h} - proportional, filled in draw()
let regions = [];

// Region specs as fractions of canvas [xF, yF, wF, hF]
// Will be computed in draw based on canvas size
let regionSpecs = [
  { id: 0, label: '1 Activity Bar', shortLabel: '?', desc: 'Activity Bar: Quick access icons for Explorer, Search, Source Control, Debug, and Extensions. Click to switch side bar views.', calloutColor: [70, 130, 220] },
  { id: 1, label: '2 Side Bar', shortLabel: '?', desc: 'Side Bar (Explorer): Shows your project file tree. Open folders, create files, and navigate your codebase from here.', calloutColor: [80, 170, 100] },
  { id: 2, label: '3 Tabs', shortLabel: '?', desc: 'Editor Tabs: Each open file gets a tab. Click to switch files; middle-click or × to close. A dot indicates unsaved changes.', calloutColor: [200, 130, 50] },
  { id: 3, label: '4 Editor Area', shortLabel: '?', desc: 'Editor Area: The main coding canvas. Features syntax highlighting, IntelliSense autocomplete, and error underlining.', calloutColor: [170, 80, 200] },
  { id: 4, label: '5 Minimap', shortLabel: '?', desc: 'Minimap: A zoomed-out preview of your entire file. Click to jump to any section. Helpful in large files.', calloutColor: [80, 190, 190] },
  { id: 5, label: '6 Terminal', shortLabel: '?', desc: 'Integrated Terminal: Run commands without leaving VS Code. Use it to run Python scripts, install packages, and manage git.', calloutColor: [210, 70, 70] },
  { id: 6, label: '7 Status Bar', shortLabel: '?', desc: 'Status Bar: Shows current Python interpreter, file encoding, line/column position, and git branch. Click Python version to switch environments.', calloutColor: [0, 160, 220] }
];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);
  textSize(defaultTextSize);

  quizButton = createButton('Quiz Mode: OFF');
  quizButton.position(margin, drawHeight + 8);
  quizButton.mousePressed(() => {
    quizMode = !quizMode;
    quizButton.html(quizMode ? 'Quiz Mode: ON — can you name each region?' : 'Quiz Mode: OFF');
  });

  describe('Interactive VS Code interface tour. Hover over each interface region to learn its name and purpose. Toggle quiz mode to test your recall.', LABEL);
}

function draw() {
  updateCanvasSize();
  background(bg[0], bg[1], bg[2]);

  // Drawing + control regions
  stroke(60);
  strokeWeight(1);
  fill(bg[0], bg[1], bg[2]);
  rect(0, 0, canvasWidth, drawHeight);
  fill(255);
  rect(0, drawHeight, canvasWidth, controlHeight);

  // Title
  noStroke();
  fill(220);
  textAlign(CENTER, TOP);
  textSize(16);
  text('VS Code Interface Tour', canvasWidth / 2, margin - 4);

  // Compute layout
  let vsTop = margin + 24;
  let vsH = drawHeight - vsTop - margin;
  let vsW = canvasWidth - margin * 2;
  let vsLeft = margin;

  let actW = 36;
  let sideW = Math.min(160, vsW * 0.22);
  let mmW = 38;
  let statusH = 22;
  let tabH = 28;
  let termH = Math.min(100, vsH * 0.26);

  let edLeft = vsLeft + actW + sideW;
  let edW = vsW - actW - sideW - mmW;
  let edTop = vsTop + tabH;
  let edH = vsH - statusH - tabH - termH;

  // Store region rects for hover detection
  regions = [
    { ...regionSpecs[0], x: vsLeft, y: vsTop, w: actW, h: vsH - statusH },
    { ...regionSpecs[1], x: vsLeft + actW, y: vsTop, w: sideW, h: vsH - statusH },
    { ...regionSpecs[2], x: edLeft, y: vsTop, w: edW + mmW, h: tabH },
    { ...regionSpecs[3], x: edLeft, y: edTop, w: edW, h: edH },
    { ...regionSpecs[4], x: edLeft + edW, y: edTop, w: mmW, h: edH },
    { ...regionSpecs[5], x: vsLeft + actW, y: edTop + edH, w: vsW - actW, h: termH },
    { ...regionSpecs[6], x: vsLeft, y: vsTop + vsH - statusH, w: vsW, h: statusH }
  ];

  // Detect hover
  hoveredRegion = -1;
  for (let i = 0; i < regions.length; i++) {
    let r = regions[i];
    if (mouseX >= r.x && mouseX <= r.x + r.w && mouseY >= r.y && mouseY <= r.y + r.h) {
      hoveredRegion = i;
    }
  }

  // Draw background panels
  function drawPanel(r, bgColor) {
    let isHov = (hoveredRegion === r.id);
    if (isHov) {
      fill(bgColor[0] + 25, bgColor[1] + 25, bgColor[2] + 25);
      stroke(r.calloutColor[0], r.calloutColor[1], r.calloutColor[2]);
      strokeWeight(2);
    } else {
      fill(bgColor[0], bgColor[1], bgColor[2]);
      stroke(55);
      strokeWeight(0.5);
    }
    rect(r.x, r.y, r.w, r.h);
  }

  // Activity Bar
  let actSpec = regions[0];
  drawPanel(actSpec, activityBarColor);
  // Activity bar icons
  noStroke();
  fill(150);
  let icons = ['📁', '🔍', '🔀', '🐛', '⚙'];
  textSize(14);
  textAlign(CENTER, CENTER);
  for (let i = 0; i < icons.length; i++) {
    text(icons[i], vsLeft + actW / 2, vsTop + 20 + i * 36);
  }

  // Side Bar
  drawPanel(regions[1], sideBarColor);
  fill(170);
  noStroke();
  textSize(10);
  textAlign(LEFT, TOP);
  let files = ['📂 my-project', '  📄 analysis.py', '  📄 data.csv', '  📂 notebooks', '    📓 explore.ipynb'];
  for (let i = 0; i < files.length; i++) {
    text(files[i], vsLeft + actW + 6, vsTop + 8 + i * 17);
  }

  // Tabs
  drawPanel(regions[2], tabBarColor);
  noStroke();
  fill(220, 220, 220);
  rect(edLeft, vsTop + 2, 110, tabH - 2, 4, 4, 0, 0);
  fill(170);
  textSize(11);
  textAlign(LEFT, CENTER);
  text('analysis.py ●', edLeft + 8, vsTop + tabH / 2);
  fill(120);
  text('data.csv ×', edLeft + 128, vsTop + tabH / 2);

  // Editor Area
  drawPanel(regions[3], editorColor);
  // Simulated code lines
  let codeLines = [
    { num: 1, code: 'import pandas as pd', color: [200, 200, 200] },
    { num: 2, code: 'import numpy as np', color: [200, 200, 200] },
    { num: 3, code: '', color: [200, 200, 200] },
    { num: 4, code: 'df = pd.read_csv("data.csv")', color: [200, 200, 200] },
    { num: 5, code: 'print(df.head())', color: [200, 200, 200] },
    { num: 6, code: '', color: [200, 200, 200] },
    { num: 7, code: '# Show column info', color: [100, 150, 100] },
    { num: 8, code: 'df.info()', color: [200, 200, 200] }
  ];
  noStroke();
  textSize(11);
  for (let i = 0; i < codeLines.length; i++) {
    let ly = edTop + 8 + i * 18;
    fill(lineNumColor[0], lineNumColor[1], lineNumColor[2]);
    textAlign(RIGHT, TOP);
    text(codeLines[i].num, edLeft + 24, ly);
    fill(codeLines[i].color[0], codeLines[i].color[1], codeLines[i].color[2]);
    textAlign(LEFT, TOP);
    text(codeLines[i].code, edLeft + 30, ly);
  }

  // Minimap
  drawPanel(regions[4], minimapColor);
  noStroke();
  fill(90);
  textAlign(CENTER, CENTER);
  textSize(9);
  for (let row = 0; row < 14; row++) {
    let lw = random(10, mmW - 8);
    fill(80 + row * 3, 80 + row, 90);
    rect(edLeft + edW + 4, edTop + 8 + row * 12, lw, 4, 1);
  }

  // Terminal
  drawPanel(regions[5], panelColor);
  noStroke();
  fill(0, 200, 0);
  textSize(11);
  textAlign(LEFT, TOP);
  text('$ python analysis.py', vsLeft + actW + 8, edTop + edH + 8);
  fill(200);
  text('   RangeIndex: 150 entries', vsLeft + actW + 8, edTop + edH + 24);
  fill(0, 200, 0);
  text('$', vsLeft + actW + 8, edTop + edH + 40);

  // Status Bar
  drawPanel(regions[6], statusBarColor);
  noStroke();
  fill(255);
  textSize(10);
  textAlign(LEFT, CENTER);
  text('⎇ main', vsLeft + 8, vsTop + vsH - statusH / 2);
  textAlign(CENTER, CENTER);
  text('Python 3.11 (data-science)', vsLeft + vsW / 2, vsTop + vsH - statusH / 2);
  textAlign(RIGHT, CENTER);
  text('Ln 5, Col 12  UTF-8', vsLeft + vsW - 6, vsTop + vsH - statusH / 2);

  // Labels / callouts
  for (let i = 0; i < regions.length; i++) {
    let r = regions[i];
    let isHov = (i === hoveredRegion);
    let labelText = quizMode ? r.shortLabel : r.label;

    // Callout badge
    let bx = r.x + r.w + 4;
    let by = r.y + r.h / 2;
    // Clamp to canvas
    if (bx + 120 > canvasWidth) bx = r.x - 124;

    fill(r.calloutColor[0], r.calloutColor[1], r.calloutColor[2]);
    noStroke();
    rect(bx, by - 10, quizMode ? 22 : 110, 20, 4);
    fill(255);
    textSize(10);
    textAlign(LEFT, CENTER);
    text(labelText, bx + 4, by);
  }

  // Description in control area
  noStroke();
  if (hoveredRegion >= 0) {
    fill(40);
    textAlign(LEFT, CENTER);
    textSize(12);
    textWrap(WORD);
    text(regions[hoveredRegion].desc, margin + 160, drawHeight + 4, canvasWidth - margin * 2 - 170, controlHeight - 8);
  } else {
    fill(120);
    textAlign(CENTER, CENTER);
    textSize(12);
    text('Hover over any region to learn what it does. Toggle Quiz Mode to test your recall!', canvasWidth / 2, drawHeight + controlHeight / 2);
  }
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  redraw();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
