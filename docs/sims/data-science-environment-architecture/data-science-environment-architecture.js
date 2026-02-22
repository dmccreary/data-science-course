// Data Science Environment Architecture
// Shows layered architecture of a Python data science environment
// Bloom Level: Understand (L2) - hover-to-reveal pattern
// MicroSim template version 2026.02

let canvasWidth = 520;
let drawHeight = 420;
let controlHeight = 40;
let canvasHeight = drawHeight + controlHeight;
let containerWidth;
let containerHeight = canvasHeight;

let margin = 20;
let defaultTextSize = 16;

// Layer data (bottom to top)
let layers = [
  {
    label: "Operating System",
    sublabel: "Windows / macOS / Linux",
    desc: "The foundation. Your OS manages hardware resources and provides the environment where all software runs.",
    color: [70, 70, 80],
    textColor: [255, 255, 255]
  },
  {
    label: "Python Installation",
    sublabel: "Python Interpreter",
    desc: "The Python engine that reads and runs your code. Think of it as the translator between your instructions and the computer.",
    color: [55, 118, 171],
    textColor: [255, 255, 255]
  },
  {
    label: "Package Manager",
    sublabel: "pip / conda",
    desc: "Your supply chain. Downloads and installs Python libraries from the internet. 'pip install pandas' uses this layer.",
    color: [220, 120, 40],
    textColor: [255, 255, 255]
  },
  {
    label: "Virtual Environment",
    sublabel: "Isolated Workspace",
    desc: "A clean room that keeps each project's packages separate. Prevents version conflicts between different projects.",
    color: [60, 140, 80],
    textColor: [255, 255, 255]
  },
  {
    label: "Data Science Libraries",
    sublabel: "NumPy · pandas · matplotlib · scikit-learn",
    desc: "Your power tools! Pre-built code that handles arrays, data tables, charts, and machine learning algorithms.",
    color: [140, 80, 180],
    textColor: [255, 255, 255]
  },
  {
    label: "IDE / Jupyter Notebook",
    sublabel: "VS Code · JupyterLab",
    desc: "The cockpit where you work! Combines code writing, running, and visualization in one place. This is where you spend your time.",
    color: [60, 130, 200],
    textColor: [255, 255, 255]
  }
];

let hoveredLayer = -1;
let layerH, layerGap, stackTop;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);
  textSize(defaultTextSize);
  describe('Vertical stack diagram showing the six layers of a Python data science environment, from Operating System at the bottom to IDE/Jupyter at the top. Hover each layer to see its description.', LABEL);
}

function draw() {
  updateCanvasSize();
  background(245);

  // Drawing region border
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
  textSize(18);
  text('Data Science Environment Architecture', canvasWidth / 2, margin - 4);

  // Compute layer geometry
  let numLayers = layers.length;
  layerGap = 4;
  stackTop = margin + 28;
  let totalGap = layerGap * (numLayers - 1);
  let availH = drawHeight - stackTop - margin;
  layerH = (availH - totalGap) / numLayers;

  let lx = margin;
  let lw = canvasWidth - margin * 2;

  // Detect hover
  hoveredLayer = -1;
  for (let i = 0; i < numLayers; i++) {
    // layers rendered bottom-to-top visually: index 0 = bottom
    let visualIdx = numLayers - 1 - i;
    let ly = stackTop + visualIdx * (layerH + layerGap);
    if (mouseX >= lx && mouseX <= lx + lw && mouseY >= ly && mouseY <= ly + layerH) {
      hoveredLayer = i;
    }
  }

  // Draw layers
  for (let i = 0; i < numLayers; i++) {
    let visualIdx = numLayers - 1 - i; // bottom layer drawn last (idx 0 at bottom)
    let ly = stackTop + visualIdx * (layerH + layerGap);
    let layer = layers[i];
    let c = layer.color;
    let bright = (hoveredLayer === i) ? 30 : 0;

    // Layer rectangle
    let r = color(c[0] + bright, c[1] + bright, c[2] + bright);
    fill(r);
    if (hoveredLayer === i) {
      stroke(255, 220, 50);
      strokeWeight(3);
    } else {
      stroke(180);
      strokeWeight(1);
    }
    let rounding = 8;
    rect(lx, ly, lw, layerH, rounding);

    // Layer text
    noStroke();
    fill(layer.textColor[0], layer.textColor[1], layer.textColor[2]);
    textAlign(LEFT, CENTER);
    textSize(15);
    text(layer.label, lx + 14, ly + layerH * 0.35);
    textSize(11);
    fill(layer.textColor[0], layer.textColor[1], layer.textColor[2], 200);
    text(layer.sublabel, lx + 14, ly + layerH * 0.72);

    // Layer number badge (1=bottom OS)
    fill(255, 255, 255, 60);
    noStroke();
    ellipse(lx + lw - 20, ly + layerH / 2, 26, 26);
    fill(layer.textColor[0], layer.textColor[1], layer.textColor[2]);
    textAlign(CENTER, CENTER);
    textSize(12);
    text(i + 1, lx + lw - 20, ly + layerH / 2);
  }

  // Description tooltip for hovered layer
  if (hoveredLayer >= 0) {
    let desc = layers[hoveredLayer].desc;
    let tipX = margin;
    let tipW = canvasWidth - margin * 2;

    // Draw description in control area
    fill(50);
    noStroke();
    textAlign(LEFT, CENTER);
    textSize(13);
    textWrap(WORD);
    text(desc, tipX + 8, drawHeight + 4, tipW - 16, controlHeight - 8);
  } else {
    // Default hint
    fill(120);
    noStroke();
    textAlign(CENTER, CENTER);
    textSize(13);
    text('Hover over a layer to learn its role in your data science setup', canvasWidth / 2, drawHeight + controlHeight / 2);
  }

  // "You work here" arrow annotation on top layer
  noStroke();
  fill(60, 130, 200);
  textAlign(CENTER, TOP);
  textSize(11);
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
