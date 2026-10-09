// Python Visualization Library Landscape
// CANVAS_HEIGHT: 620
// Bloom L2 (Understand): students compare Matplotlib, Seaborn, and Plotly. Three cards list each
// library's strengths, best uses, learning curve, and interactivity. Choosing a need from the
// "I need" list marks the library that fits and explains why. Clicking a library shows a typical
// chart of the same 40 restaurant bills in that library's default look, with the code for it.
//
// Model: tip = 0.92 + 0.105 * bill + noise (seeded). The Seaborn chart adds the least-squares
// line y = a + b x and its 95% confidence band y +/- t * s * sqrt(1/n + (x - mean)^2 / Sxx),
// with s the residual standard error and t = 2.024 (38 degrees of freedom).
// Hex colors are the real default colors of each library, so the three looks are faithful.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 575;
let controlHeight = 45;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

// ---- seeded data (plain JavaScript so it is ready before p5 starts) ----
let lcg = 42;
function rnd() { lcg = (lcg * 1664525 + 1013904223) % 4294967296; return lcg / 4294967296; }
function gauss(mean, sd) { return mean + sd * Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd()); }
const TIPS = Array.from({ length: 40 }, () => { const bill = 8 + 40 * rnd(); return { bill, tip: Math.max(1, 0.92 + 0.105 * bill + gauss(0, 0.9)) }; });
// least-squares fit and the pieces of its confidence band
const N = TIPS.length, MEAN_X = TIPS.reduce((s, d) => s + d.bill, 0) / N, MEAN_Y = TIPS.reduce((s, d) => s + d.tip, 0) / N;
const SXX = TIPS.reduce((s, d) => s + (d.bill - MEAN_X) ** 2, 0);
const SLOPE = TIPS.reduce((s, d) => s + (d.bill - MEAN_X) * (d.tip - MEAN_Y), 0) / SXX, INTERCEPT = MEAN_Y - SLOPE * MEAN_X;
const RESID_SE = Math.sqrt(TIPS.reduce((s, d) => s + (d.tip - INTERCEPT - SLOPE * d.bill) ** 2, 0) / (N - 2));
const halfBand = x => 2.024 * RESID_SE * Math.sqrt(1 / N + (x - MEAN_X) ** 2 / SXX);

const LIBS = [
  { name: 'Matplotlib', tag: 'The Foundation', color: 'royalblue', ink: 'mediumblue',
    strengths: ['Complete control over every element', 'Publication-quality static images', 'Huge community and documentation',
      'Seaborn and pandas build on it'],
    best: ['Academic papers', 'Print publications', 'Maximum customization'],
    curve: 'Medium-High', inter: 'Limited (static image)',
    code: ['import matplotlib.pyplot as plt', "plt.scatter(tips['total_bill'], tips['tip'])", "plt.title('Tips')",
      "plt.xlabel('Total bill ($)')", "plt.ylabel('Tip ($)')", 'plt.show()'],
    caption: 'A static image by default. You add the title and axis labels yourself and can change every detail.',
    look: { bg: 'white', frame: 'black', dot: '#1F77B4', xLabel: 'Total bill ($)', yLabel: 'Tip ($)', title: 'Tips' } },
  { name: 'Seaborn', tag: 'Beautiful Statistics', color: 'teal', ink: 'darkslategray',
    strengths: ['Attractive default styles', 'Built-in statistical plots', 'Works with pandas DataFrames', 'Less code for common plots'],
    best: ['Statistical analysis', 'Exploratory data analysis', 'Quick, good-looking plots'],
    curve: 'Low-Medium', inter: 'Limited (uses Matplotlib)',
    code: ['import seaborn as sns', 'sns.set_theme()', "sns.regplot(data=tips, x='total_bill', y='tip')"],
    caption: 'One call fits the regression line and shades its 95% confidence band. The labels come from the column names.',
    look: { bg: '#EAEAF2', grid: 'white', dot: '#4C72B0', xLabel: 'total_bill', yLabel: 'tip', fit: true } },
  { name: 'Plotly', tag: 'Interactive & Modern', color: 'mediumpurple', ink: 'indigo',
    strengths: ['Zoom, pan, and hover built in', 'Web-ready HTML output', '3D charts', 'Dashboards with Dash'],
    best: ['Web applications', 'Presentations', 'Data exploration'],
    curve: 'Low-Medium', inter: 'Full (built in)',
    code: ['import plotly.express as px', "fig = px.scatter(tips, x='total_bill', y='tip')", 'fig.show()'],
    caption: 'Interactive. Point at a dot here to see its values. A real Plotly chart also zooms and pans.',
    look: { bg: '#E5ECF6', grid: 'white', dot: '#636EFA', xLabel: 'total_bill', yLabel: 'tip', hover: true } }
];

// The decision guide: each need, the libraries that fit it, and the reason
const NEEDS = [
  ['a figure for a printed paper or PDF', [0], 'Matplotlib gives exact control over every element and saves sharp static files such as PDF and SVG.'],
  ['statistics on the chart, such as a regression fit', [1], 'Seaborn computes and draws the statistics for you, straight from a DataFrame.'],
  ['a chart people can explore on a web page', [2], 'Plotly charts zoom, pan, and show tooltips with no extra code, and they save as HTML.'],
  ['a quick first look at a new dataset', [1, 2], 'Both make a clear chart from a DataFrame in one line. Pick Plotly when you want to hover and zoom.'],
  ['full control over every detail', [0], 'Matplotlib exposes every line, tick, and label. Seaborn charts can be fine-tuned with Matplotlib too.']
];

let selected = 0;                 // library whose example is shown
let need = 0;                     // index into NEEDS
let needSelect;
let cardBoxes = [];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  needSelect = createSelect();
  needSelect.parent(mainElement);
  needSelect.position(68, drawHeight + 11);
  needSelect.style('font-size', '14px');
  NEEDS.forEach((n, i) => needSelect.option(n[0], i));
  needSelect.changed(() => { need = Number(needSelect.value()); selected = NEEDS[need][1][0]; });

  describe('Three cards compare the Python libraries Matplotlib, Seaborn, and Plotly by strengths, best uses, ' +
    'learning curve, and interactivity. A drop-down list of needs marks the library that fits and gives the reason. ' +
    'The selected library shows a typical scatter chart of tips against total bill in its default style with its code.', LABEL);
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600, fullW = canvasWidth - 2 * margin;
  needSelect.style('max-width', (canvasWidth - 80) + 'px');
  textWrap(WORD);
  textStyle(NORMAL);
  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('I need', 10, drawHeight + 22);
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Python Visualization Libraries', canvasWidth / 2, 8);

  // three library cards. When narrow they shrink to tabs and the facts move to a panel below.
  const gap = narrow ? 6 : 10, cardW = (fullW - 2 * gap) / 3, headH = narrow ? 44 : 46;
  const cardY = narrow ? 32 : 42, cardH = narrow ? headH : 277, fits = NEEDS[need][1];
  cardBoxes = LIBS.map((lib, i) => ({ x: margin + i * (cardW + gap), y: cardY, w: cardW, h: cardH, index: i }));
  cardBoxes.forEach(b => {
    const lib = LIBS[b.index], on = b.index === selected;
    fill('white');
    stroke(lib.color);
    strokeWeight(on ? 3 : 1);
    rect(b.x, b.y, b.w, b.h, 8);
    noStroke();
    fill(lib.color);
    if (!narrow) rect(b.x, b.y, b.w, headH, 8, 8, 0, 0);
    else if (on) rect(b.x, b.y, b.w, headH, 8);
    fill(narrow && !on ? lib.ink : 'white');
    textAlign(LEFT, TOP);
    textStyle(BOLD);
    textSize(narrow ? 14 : 19);
    text(lib.name + (narrow && fits.includes(b.index) ? '  ✓' : ''), b.x + 8, b.y + 6);
    textStyle(ITALIC);
    textSize(narrow ? 11 : 13);
    text(lib.tag, b.x + 8, b.y + (narrow ? 25 : 28));
    textStyle(NORMAL);
    if (narrow) return;
    drawOutput(lib, b.x + b.w - 66, b.y + 5, 58, 36, true);
    drawFacts(lib, b.x + 10, b.y + headH + 8, b.w - 20, false);
    if (fits.includes(b.index)) {
      noStroke();
      fill('seagreen');
      rect(b.x + 8, b.y + b.h - 26, b.w - 16, 20, 5);
      fill('white');
      textStyle(BOLD);
      textSize(13);
      textAlign(CENTER, CENTER);
      text('✓  Fits what you need', b.x + b.w / 2, b.y + b.h - 15);
      textStyle(NORMAL);
    }
  });
  cursor(cardBoxes.some(mouseOver) ? HAND : ARROW);

  let y = cardY + cardH + (narrow ? 6 : 8);
  const lib = LIBS[selected];
  if (narrow) {
    fill('white');
    stroke(lib.color);
    strokeWeight(1.5);
    rect(margin, y, fullW, 128, 8);
    drawFacts(lib, margin + 8, y + 7, fullW - 16, true);
    y += 134;
  }
  const reasonH = narrow ? 62 : 46, reasonY = drawHeight - 8 - reasonH;
  drawExample(lib, margin, y, fullW, reasonY - 8 - y, narrow);

  // the reason for the current need
  fill('honeydew');
  stroke('seagreen');
  strokeWeight(1);
  rect(margin, reasonY, fullW, reasonH, 8);
  noStroke();
  fill('black');
  textSize(narrow ? 12 : 14);
  textLeading(narrow ? 16 : 18);
  textAlign(LEFT, TOP);
  text('Best fit: ' + fits.map(i => LIBS[i].name).join(' or ') + '.  ' + NEEDS[need][2], margin + 10, reasonY + 6, fullW - 20, reasonH - 8);
}

// Strengths, best uses, learning curve, and interactivity of one library
function drawFacts(lib, x, y, w, twoCol) {
  const ts = twoCol ? 11 : 13, lh = ts + 4, colW = twoCol ? w * 0.58 : w;
  const list = (title, items, lx, ly) => {
    noStroke();
    textAlign(LEFT, TOP);
    textSize(ts);
    textStyle(BOLD);
    fill(lib.ink);
    text(title, lx, ly);
    textStyle(NORMAL);
    fill('black');
    items.forEach((it, k) => text('•  ' + it, lx, ly + (k + 1) * lh));
    return ly + (items.length + 1) * lh;
  };
  let bottom = list('Strengths', lib.strengths, x, y);
  bottom = twoCol ? max(bottom, list('Best for', lib.best, x + colW, y)) : list('Best for', lib.best, x, bottom + 6);
  fill('dimgray');
  textSize(twoCol ? 11 : 12);
  text('Learning curve: ' + lib.curve, x, bottom + 6);
  text('Interactivity: ' + lib.inter, x, bottom + 6 + lh - 1);
}

// The selected library's code and a typical chart in its default look
function drawExample(lib, x, y, w, h, narrow) {
  fill('white');
  stroke(lib.color);
  strokeWeight(2);
  rect(x, y, w, h, 10);
  const pad = narrow ? 8 : 12, ts = narrow ? 11 : 13, lh = ts + 4;
  const codeW = narrow ? w - 2 * pad : w * 0.52, codeH = lib.code.length * lh + 10;
  noStroke();
  fill(lib.ink);
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(narrow ? 12 : 14);
  text('A typical ' + lib.name + ' chart: ' + lib.code.length + ' lines of code', x + pad, y + pad - 2);
  const cy = y + pad + (narrow ? 16 : 20);
  fill('whitesmoke');
  stroke('silver');
  strokeWeight(1);
  rect(x + pad, cy, codeW, codeH, 5);
  noStroke();
  fill('darkslateblue');
  textSize(ts);
  lib.code.forEach((ln, i) => text(ln, x + pad + 8, cy + 6 + i * lh));
  textStyle(NORMAL);
  fill('black');
  textSize(narrow ? 12 : 13);
  textLeading(narrow ? 15 : 17);
  // the caption sits under the code when wide and under the chart when narrow
  if (narrow) {
    const oy = cy + codeH + 6, oh = y + h - pad - 32 - oy;
    drawOutput(lib, x + pad, oy, w - 2 * pad, oh, false);
    noStroke();
    fill('black');
    textAlign(LEFT, TOP);
    text(lib.caption, x + pad, oy + oh + 4, w - 2 * pad, 32);
  } else {
    text(lib.caption, x + pad, cy + codeH + 8, codeW, y + h - pad - cy - codeH);
    drawOutput(lib, x + pad + codeW + 14, y + pad, w - 2 * pad - codeW - 14, h - 2 * pad, false);
  }
}

// The chart as the library draws it by default. The small version is the icon in a card header.
function drawOutput(lib, x, y, w, h, small) {
  const k = lib.look;
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h);
  const l = x + (small ? 4 : 48), t = y + (small ? 4 : k.title ? 24 : 10);
  const pw = x + w - (small ? 4 : 12) - l, ph = y + h - (small ? 4 : 38) - t;
  const X = bill => l + (bill - 4) / 48 * pw, Y = tip => t + ph - tip / 8.5 * ph;   // axes: bill 4 to 52, tip 0 to 8.5
  fill(k.bg);
  if (k.frame) stroke(k.frame); else noStroke();
  rect(l, t, pw, ph);
  textSize(11);
  textStyle(NORMAL);
  for (let v = 10; v <= 50; v += 10) {
    stroke(k.grid || k.frame);
    if (k.grid) line(X(v), t, X(v), t + ph); else line(X(v), t + ph, X(v), t + ph + (small ? 0 : 4));
    if (small) continue;
    noStroke();
    fill('dimgray');
    textAlign(CENTER, TOP);
    text(v, X(v), t + ph + 6);
  }
  for (let v = 2; v <= 8; v += 2) {
    stroke(k.grid || k.frame);
    if (k.grid) line(l, Y(v), l + pw, Y(v)); else line(l - (small ? 0 : 4), Y(v), l, Y(v));
    if (small) continue;
    noStroke();
    fill('dimgray');
    textAlign(RIGHT, CENTER);
    text(v, l - 7, Y(v));
  }
  const dot = color(k.dot);
  if (k.fit) {
    // regression line with its 95% confidence band
    const xs = Array.from({ length: 13 }, (_, i) => 6 + i * 44 / 12), line95 = v => INTERCEPT + SLOPE * v;
    dot.setAlpha(60);
    noStroke();
    fill(dot);
    beginShape();
    xs.forEach(v => vertex(X(v), Y(line95(v) + halfBand(v))));
    xs.slice().reverse().forEach(v => vertex(X(v), Y(line95(v) - halfBand(v))));
    endShape(CLOSE);
    stroke(k.dot);
    strokeWeight(small ? 1.5 : 2.5);
    line(X(6), Y(line95(6)), X(50), Y(line95(50)));
    dot.setAlpha(200);
  }
  noStroke();
  fill(dot);
  let hover = null;
  TIPS.forEach(d => {
    circle(X(d.bill), Y(d.tip), small ? 3 : 7);
    if (k.hover && !small && dist(mouseX, mouseY, X(d.bill), Y(d.tip)) < 7) hover = d;
  });
  if (small) return;
  // title and axis labels
  fill('black');
  textSize(12);
  textAlign(CENTER, TOP);
  if (k.title) text(k.title, l + pw / 2, y + 7);
  text(k.xLabel, l + pw / 2, t + ph + 21);
  push();
  translate(x + 12, t + ph / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, CENTER);
  text(k.yLabel, 0, 0);
  pop();
  if (!hover) return;
  // tooltip in the default Plotly Express format
  const tx = X(hover.bill) + 110 > x + w ? X(hover.bill) - 108 : X(hover.bill) + 10, ty = constrain(Y(hover.tip) - 20, y + 2, y + h - 42);
  fill(k.dot);
  stroke('white');
  rect(tx, ty, 98, 38, 4);
  noStroke();
  fill('white');
  textStyle(BOLD);
  textAlign(LEFT, TOP);
  text('total_bill=' + hover.bill.toFixed(2), tx + 6, ty + 5);
  text('tip=' + hover.tip.toFixed(2), tx + 6, ty + 21);
  textStyle(NORMAL);
}

// Clicking a library card shows that library's example.
function mouseOver(b) {
  return mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h;
}
function mousePressed() {
  const box = cardBoxes.find(mouseOver);
  if (box) selected = box.index;
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
