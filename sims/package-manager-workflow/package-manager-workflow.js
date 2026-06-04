// Package Manager Workflow
// Horizontal flowchart showing how pip/conda installs packages
// Bloom Level: Understand (L2) - hover-to-reveal pattern
// MicroSim template version 2026.02

let canvasWidth = 520;
let drawHeight = 360;
let controlHeight = 50;
let canvasHeight = drawHeight + controlHeight;
let containerWidth;
let containerHeight = canvasHeight;

let margin = 18;
let defaultTextSize = 15;

let steps = [
  {
    icon: '⌨',
    title: 'Type Command',
    code: 'pip install pandas',
    desc: 'You type a pip or conda command in the terminal. This triggers the package manager to start working.',
    color: [60, 120, 200],
    errorNote: ''
  },
  {
    icon: '🔍',
    title: 'Search Registry',
    code: 'Searching PyPI...',
    desc: 'The package manager searches PyPI (pip) or Anaconda Cloud (conda) for the package you requested.',
    color: [200, 120, 40],
    errorNote: 'Error: "Package not found" means the name is misspelled or the package doesn\'t exist on PyPI.'
  },
  {
    icon: '⬇',
    title: 'Download Package',
    code: 'Downloading pandas-2.x...',
    desc: 'Downloads the package AND all its dependencies. pandas needs NumPy, so NumPy downloads too!',
    color: [50, 160, 90],
    errorNote: 'Error: Check your internet connection if downloads fail.'
  },
  {
    icon: '📁',
    title: 'Install to Env',
    code: 'Installing collected packages...',
    desc: 'Files are extracted and placed into your active virtual environment\'s site-packages folder.',
    color: [140, 70, 180],
    errorNote: 'Error: "Permission denied" usually means you need to activate your virtual environment first.'
  },
  {
    icon: '✓',
    title: 'Ready to Import',
    code: 'import pandas as pd',
    desc: 'Success! The package is now importable in any Python script or notebook in this environment.',
    color: [180, 150, 0],
    errorNote: ''
  }
];

let hoveredStep = -1;
let stepW, stepH, startX, stepY;
let arrowW = 22;
let showErrors = false;
let errorButton;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);
  textSize(defaultTextSize);

  errorButton = createButton('Show Common Errors');
  errorButton.position(margin, drawHeight + 10);
  errorButton.mousePressed(() => {
    showErrors = !showErrors;
    errorButton.html(showErrors ? 'Hide Common Errors' : 'Show Common Errors');
  });

  describe('Horizontal flowchart showing five steps of package installation: Type Command, Search Registry, Download Package, Install to Environment, Ready to Import. Hover each step to see details.', LABEL);
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
  textSize(17);
  text('Package Manager Workflow', canvasWidth / 2, margin - 2);

  // Geometry
  let n = steps.length;
  let availW = canvasWidth - margin * 2;
  let totalArrowW = arrowW * (n - 1);
  stepW = (availW - totalArrowW) / n;
  stepH = drawHeight - margin * 2 - 36;
  stepY = margin + 34;
  startX = margin;

  // Detect hover
  hoveredStep = -1;
  for (let i = 0; i < n; i++) {
    let sx = startX + i * (stepW + arrowW);
    if (mouseX >= sx && mouseX <= sx + stepW && mouseY >= stepY && mouseY <= stepY + stepH) {
      hoveredStep = i;
    }
  }

  // Draw steps and arrows
  for (let i = 0; i < n; i++) {
    let sx = startX + i * (stepW + arrowW);
    let s = steps[i];
    let isHovered = (i === hoveredStep);
    let c = s.color;
    let bright = isHovered ? 25 : 0;

    // Box
    fill(c[0] + bright, c[1] + bright, c[2] + bright);
    if (isHovered) {
      stroke(255, 220, 50);
      strokeWeight(3);
    } else {
      stroke(160);
      strokeWeight(1);
    }
    rect(sx, stepY, stepW, stepH, 8);

    // Icon
    noStroke();
    fill(255, 255, 255, 200);
    textAlign(CENTER, TOP);
    textSize(28);
    text(s.icon, sx + stepW / 2, stepY + 10);

    // Step number
    fill(255, 255, 255, 130);
    textSize(10);
    text('STEP ' + (i + 1), sx + stepW / 2, stepY + 46);

    // Title
    fill(255);
    textSize(13);
    text(s.title, sx + stepW / 2, stepY + 60);

    // Code snippet
    fill(255, 255, 200, 200);
    textSize(10);
    textWrap(WORD);
    text(s.code, sx + 6, stepY + 80, stepW - 12);

    // Arrow to next step
    if (i < n - 1) {
      let ax = sx + stepW + 1;
      let ay = stepY + stepH / 2;
      fill(100);
      noStroke();
      triangle(ax + arrowW - 2, ay, ax + 4, ay - 7, ax + 4, ay + 7);
      stroke(100);
      strokeWeight(2);
      line(ax, ay, ax + arrowW - 6, ay);
      noStroke();
    }
  }

  // Description / error panel
  if (hoveredStep >= 0) {
    let s = steps[hoveredStep];
    fill(40);
    noStroke();
    textAlign(LEFT, CENTER);
    textSize(12);
    let msg = (showErrors && s.errorNote) ? s.errorNote : s.desc;
    textWrap(WORD);
    text(msg, margin, drawHeight + 4, canvasWidth - margin * 2 - 160, controlHeight - 8);
  } else {
    fill(120);
    noStroke();
    textAlign(CENTER, CENTER);
    textSize(12);
    text('Hover each step to learn what happens during package installation', canvasWidth / 2, drawHeight + controlHeight / 2);
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
