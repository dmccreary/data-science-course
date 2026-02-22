// Data Structure Hierarchy
// Tree diagram showing Python native and pandas data structures
// Bloom Level: Understand (L2) - hover-to-reveal pattern
// MicroSim template version 2026.02

let canvasWidth = 640;
let drawHeight = 440;
let controlHeight = 40;
let canvasHeight = drawHeight + controlHeight;
let containerWidth;
let containerHeight = canvasHeight;

let margin = 20;
let defaultTextSize = 14;

// Tree nodes with positions computed in draw()
let nodes = [
  // Root
  { id: 0, label: 'Python Data Structures', sublabel: '', parent: -1,
    desc: 'Python offers several built-in container types and the pandas library adds powerful tabular structures for data science.',
    color: [50, 80, 160], textColor: [255, 255, 255], x: 0, y: 0, w: 0, h: 0 },

  // Level 1
  { id: 1, label: 'Native Python', sublabel: 'Built into Python', parent: 0,
    desc: 'These structures come with Python — no import needed. Each is optimized for different use cases.',
    color: [60, 120, 200], textColor: [255, 255, 255], x: 0, y: 0, w: 0, h: 0 },
  { id: 2, label: 'pandas Library', sublabel: 'import pandas as pd', parent: 0,
    desc: 'pandas extends Python with tabular data structures designed specifically for data analysis and manipulation.',
    color: [150, 80, 40], textColor: [255, 255, 255], x: 0, y: 0, w: 0, h: 0 },

  // Level 2 — Native Python children
  { id: 3, label: 'List', sublabel: '[ ]', parent: 1,
    desc: 'Ordered, mutable sequence. Use for collections of items where order matters and items may change.\nExample: [1, 2, 3, "hello"]',
    color: [80, 150, 220], textColor: [255, 255, 255], x: 0, y: 0, w: 0, h: 0 },
  { id: 4, label: 'Dictionary', sublabel: '{ key: value }', parent: 1,
    desc: 'Key-value pairs. Use when you need fast lookup by name. Like a real dictionary: look up a word, get a definition.\nExample: {"name": "Alice", "age": 25}',
    color: [80, 150, 220], textColor: [255, 255, 255], x: 0, y: 0, w: 0, h: 0 },
  { id: 5, label: 'Tuple', sublabel: '( )', parent: 1,
    desc: 'Ordered, immutable sequence. Use when data should not change (e.g., coordinates, RGB colors).\nExample: (40.7128, -74.0060)',
    color: [80, 150, 220], textColor: [255, 255, 255], x: 0, y: 0, w: 0, h: 0 },

  // Level 2 — pandas children
  { id: 6, label: 'Series', sublabel: 'pd.Series()', parent: 2,
    desc: '1D labeled array. Like a single column of data with an index. Built on top of a NumPy array.\nExample: pd.Series([10, 20, 30], index=["a","b","c"])',
    color: [200, 120, 60], textColor: [255, 255, 255], x: 0, y: 0, w: 0, h: 0 },
  { id: 7, label: 'DataFrame', sublabel: 'pd.DataFrame()', parent: 2,
    desc: '2D labeled table with rows and columns. The most important data structure in data science — like a spreadsheet.\nExample: pd.read_csv("data.csv")',
    color: [200, 120, 60], textColor: [255, 255, 255], x: 0, y: 0, w: 0, h: 0 }
];

let hoveredNode = -1;
let nodePositions = []; // {x, y, w, h} for each node

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);
  textSize(defaultTextSize);
  describe('Hierarchical tree diagram of Python data structures. Root node splits into Native Python (List, Dictionary, Tuple) and pandas (Series, DataFrame). Hover a node to learn about each structure.', LABEL);
}

function draw() {
  updateCanvasSize();
  background(248);

  fill(248);
  stroke(200);
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill(255);
  rect(0, drawHeight, canvasWidth, controlHeight);

  // Title
  noStroke();
  fill(30);
  textAlign(CENTER, TOP);
  textSize(17);
  text('Python Data Structure Hierarchy', canvasWidth / 2, margin - 4);

  // Layout constants
  let treeTop = margin + 28;
  let nodeW = min(140, (canvasWidth - margin * 2) * 0.22);
  let nodeH = 48;
  let levelGap = 90;

  // Row 0: root — centered
  let r0x = canvasWidth / 2 - nodeW / 2;
  let r0y = treeTop;

  // Row 1: two children evenly
  let spacing1 = (canvasWidth - margin * 2) / 2;
  let r1y = r0y + nodeH + levelGap;
  let n1x = [margin + spacing1 * 0.17, margin + spacing1 * 0.67 + spacing1 * 0.05];

  // Row 2: three under L1, two under L2
  let r2y = r1y + nodeH + levelGap;
  let colGap2 = (canvasWidth - margin * 2) / 5;
  let nativeXs = [margin + colGap2 * 0.1, margin + colGap2 * 1.1, margin + colGap2 * 2.1];
  let pandasXs = [margin + colGap2 * 3.1, margin + colGap2 * 4.1];

  // Positions
  nodePositions = [
    { x: r0x, y: r0y, w: nodeW, h: nodeH },                  // 0 root
    { x: n1x[0], y: r1y, w: nodeW, h: nodeH },                // 1 native
    { x: n1x[1], y: r1y, w: nodeW, h: nodeH },                // 2 pandas
    { x: nativeXs[0], y: r2y, w: nodeW, h: nodeH },           // 3 list
    { x: nativeXs[1], y: r2y, w: nodeW, h: nodeH },           // 4 dict
    { x: nativeXs[2], y: r2y, w: nodeW, h: nodeH },           // 5 tuple
    { x: pandasXs[0], y: r2y, w: nodeW, h: nodeH },           // 6 series
    { x: pandasXs[1], y: r2y, w: nodeW, h: nodeH }            // 7 dataframe
  ];

  // Detect hover
  hoveredNode = -1;
  for (let i = 0; i < nodePositions.length; i++) {
    let p = nodePositions[i];
    if (mouseX >= p.x && mouseX <= p.x + p.w && mouseY >= p.y && mouseY <= p.y + p.h) {
      hoveredNode = i;
    }
  }

  // Draw edges first
  stroke(160);
  strokeWeight(1.5);
  for (let i = 1; i < nodes.length; i++) {
    let child = nodePositions[i];
    let parent = nodePositions[nodes[i].parent];
    let cx = child.x + child.w / 2;
    let cy = child.y;
    let px = parent.x + parent.w / 2;
    let py = parent.y + parent.h;
    // Elbow connector
    let midY = (cy + py) / 2;
    line(px, py, px, midY);
    line(px, midY, cx, midY);
    line(cx, midY, cx, cy);
  }

  // Draw nodes
  for (let i = 0; i < nodes.length; i++) {
    let n = nodes[i];
    let p = nodePositions[i];
    let isHov = (i === hoveredNode);
    let c = n.color;

    fill(isHov ? color(c[0] + 30, c[1] + 30, c[2] + 30) : color(c[0], c[1], c[2]));
    if (isHov) {
      stroke(255, 220, 50);
      strokeWeight(3);
    } else {
      stroke(130);
      strokeWeight(1);
    }
    rect(p.x, p.y, p.w, p.h, 8);

    noStroke();
    fill(n.textColor[0], n.textColor[1], n.textColor[2]);
    textAlign(CENTER, CENTER);
    textSize(12);
    text(n.label, p.x + p.w / 2, p.y + p.h / 2 - 6);
    textSize(10);
    fill(n.textColor[0], n.textColor[1], n.textColor[2], 180);
    text(n.sublabel, p.x + p.w / 2, p.y + p.h / 2 + 10);
  }

  // Description in control area
  noStroke();
  if (hoveredNode >= 0) {
    fill(40);
    textAlign(LEFT, CENTER);
    textSize(12);
    textWrap(WORD);
    text(nodes[hoveredNode].desc, margin, drawHeight + 4, canvasWidth - margin * 2, controlHeight - 8);
  } else {
    fill(120);
    textAlign(CENTER, CENTER);
    textSize(12);
    text('Hover any node to learn when to use each data structure', canvasWidth / 2, drawHeight + controlHeight / 2);
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
