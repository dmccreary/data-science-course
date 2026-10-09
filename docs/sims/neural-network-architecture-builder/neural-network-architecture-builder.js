// Neural Network Architecture Builder
// CANVAS_HEIGHT: 630
// Bloom L3 and L6 (Apply, Create): students build a fully connected network by adding and
// removing hidden layers and setting the number of neurons in every layer. They apply the
// parameter formula to each layer and design networks that meet a parameter budget.
//
// Model: a fully connected layer nn.Linear(n_in, n_out) has n_in * n_out weights and n_out
// biases, so a network with layer sizes n_0, n_1, ..., n_L has
//   parameters = sum over layers of (n_(l-1) * n_l + n_l).
// ReLU has no parameters. The code panel is the nn.Sequential model for the drawn network:
// ReLU after every hidden layer and no activation after the output layer.
// Checked against PyTorch 2.10.0: sum(p.numel() for p in model.parameters()) on the shown code.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 550;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let sliderLeftMargin = 215;
let defaultTextSize = 16;

const MAX_HIDDEN = 5;               // at most five hidden layers
const MAX_NEURONS = 128;            // at most 128 neurons in a layer
const PRESETS = [
  { name: 'Chapter example', sizes: [4, 8, 4, 1] },
  { name: 'Simple', sizes: [4, 8, 1] },
  { name: 'Deep', sizes: [4, 32, 16, 8, 1] },
  { name: 'Wide', sizes: [4, 128, 1] },
  { name: 'Classification', sizes: [4, 16, 8, 3] },
  { name: 'Custom' }
];

let sizes = PRESETS[0].sizes.slice();   // neurons per layer: input, hidden layers, output
let sel = 1;                            // index of the selected layer
let presetSelect, addButton, removeButton, sizeSlider;
let hits = [];                          // clickable layer columns of the last frame: { i, x, y, w, h }

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  presetSelect = createSelect();
  presetSelect.parent(mainElement);
  presetSelect.position(10, drawHeight + 8);
  presetSelect.style('font-size', '15px');
  PRESETS.forEach(p => presetSelect.option(p.name));
  presetSelect.changed(() => {
    const p = PRESETS.find(q => q.name === presetSelect.value());
    if (!p.sizes) return;
    sizes = p.sizes.slice();
    sel = 1;
    syncControls(false);
  });

  addButton = createButton('Add Layer');
  addButton.parent(mainElement);
  addButton.position(150, drawHeight + 8);
  addButton.mousePressed(() => {                 // a new hidden layer goes after the selected layer
    if (sizes.length - 2 >= MAX_HIDDEN) return;
    sel = constrain(sel + 1, 1, sizes.length - 1);
    sizes.splice(sel, 0, 8);
    syncControls(true);
  });

  removeButton = createButton('Remove Layer');
  removeButton.parent(mainElement);
  removeButton.position(232, drawHeight + 8);
  removeButton.mousePressed(() => {              // removes the selected hidden layer (or the last one)
    if (sizes.length - 2 <= 1) return;
    sel = constrain(sel, 1, sizes.length - 2);
    sizes.splice(sel, 1);
    sel = min(sel, sizes.length - 2);
    syncControls(true);
  });

  sizeSlider = createSlider(1, MAX_NEURONS, sizes[sel], 1);
  sizeSlider.parent(mainElement);
  sizeSlider.position(sliderLeftMargin, drawHeight + 45);
  sizeSlider.size(canvasWidth - sliderLeftMargin - margin);
  sizeSlider.input(() => {
    sizes[sel] = sizeSlider.value();
    presetSelect.selected('Custom');
  });
  syncControls(false);

  describe('A diagram of a fully connected neural network with an input layer, one to five hidden layers, and an ' +
    'output layer. Buttons add and remove hidden layers, clicking a layer selects it, and a slider sets its number of ' +
    'neurons. Panels show the matching PyTorch nn.Sequential code with the parameter calculation for every layer, a ' +
    'bar for each layer, and the total number of weights, biases, and parameters.', LABEL);
}

// Keep the slider and the buttons in step with the network after it changes
function syncControls(custom) {
  sizeSlider.value(sizes[sel]);
  if (custom) presetSelect.selected('Custom');
  const hidden = sizes.length - 2;
  if (hidden >= MAX_HIDDEN) addButton.attribute('disabled', ''); else addButton.removeAttribute('disabled');
  if (hidden <= 1) removeButton.attribute('disabled', ''); else removeButton.removeAttribute('disabled');
}

function layerName(i) { return i === 0 ? 'Input' : i === sizes.length - 1 ? 'Output' : 'Hidden ' + i; }

function fmt(n) { return n.toLocaleString('en-US'); }

// One entry per nn.Linear: it joins layer i to layer i + 1
function linearLayers() {
  return sizes.slice(1).map((nout, i) => ({ nin: sizes[i], nout, weights: sizes[i] * nout, biases: nout,
    params: sizes[i] * nout + nout, hot: i === sel || i + 1 === sel }));
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600, w = canvasWidth - 2 * margin, layers = linearLayers();
  cursor(hits.some(r => mouseX >= r.x && mouseX <= r.x + r.w && mouseY >= r.y && mouseY <= r.y + r.h) ? HAND : ARROW);
  textWrap(WORD);

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Neural Network Architecture Builder', canvasWidth / 2, 8);

  if (narrow) {
    drawNetwork(margin, 36, w, 176, layers, true);
    drawCode(margin, 218, w, 160, layers, true);
    drawParams(margin, 384, w, 158, layers, true);
  } else {
    const codeW = Math.round(w * 0.6);
    drawNetwork(margin, 44, w, 280, layers, false);
    drawCode(margin, 332, codeW, 210, layers, false);
    drawParams(margin + codeW + 10, 332, w - codeW - 10, 210, layers, false);
  }

  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Neurons in ' + layerName(sel) + ': ' + sizes[sel], 10, drawHeight + 56);
}

function panel(x0, y0, w, h, bg) {
  fill(bg);
  stroke('silver');
  strokeWeight(1);
  rect(x0, y0, w, h, 10);
}

// The network: one column of circles per layer, every neuron joined to every neuron in the next layer.
// A layer with more neurons than fit is drawn with a gap of three dots; its label gives the true size.
function drawNetwork(x0, y0, w, h, layers, narrow) {
  panel(x0, y0, w, h, 'white');
  const L = sizes.length, pad = narrow ? 34 : 62, ts = narrow ? 11 : 13;
  const gap = (w - 2 * pad) / (L - 1), cx = i => x0 + pad + i * gap;
  const top = y0 + (narrow ? 36 : 48), bottom = y0 + h - (narrow ? 38 : 48), mid = (top + bottom) / 2;
  const maxShow = narrow ? 6 : 8, s = (bottom - top) / maxShow, d = Math.min(s - 4, 22);
  const cols = sizes.map(n => {
    const m = Math.min(n, maxShow), col = [];
    for (let k = 0; k < m; k++) col.push({ y: mid + (k - (m - 1) / 2) * s, dots: n > maxShow && k === Math.floor(m / 2) });
    return col;
  });

  // selected layer
  fill('lightyellow');
  stroke('goldenrod');
  strokeWeight(1.5);
  rect(cx(sel) - (narrow ? 25 : 34), y0 + 4, narrow ? 50 : 68, bottom - y0 + 2, 8);

  // connections, and under them the number of parameters in that nn.Linear
  layers.forEach((l, i) => {
    stroke(l.hot ? color(65, 105, 225, 130) : color(110, 110, 110, 70));
    strokeWeight(1);
    for (const a of cols[i]) for (const b of cols[i + 1]) if (!a.dots && !b.dots) line(cx(i), a.y, cx(i + 1), b.y);
    noStroke();
    fill(l.hot ? 'royalblue' : 'black');
    textAlign(CENTER, CENTER);
    textStyle(BOLD);
    textSize(ts);
    text(fmt(l.params) + (narrow ? '' : ' params'), cx(i) + gap / 2, bottom + (narrow ? 12 : 15));
    textStyle(NORMAL);
  });

  hits = [];
  sizes.forEach((n, i) => {
    const c = i === 0 ? 'mediumseagreen' : i === L - 1 ? 'darkorange' : 'cornflowerblue';
    for (const p of cols[i]) {
      if (p.dots) {
        noStroke();
        fill('dimgray');
        for (const dy of [-5, 0, 5]) circle(cx(i), p.y + dy, 3);
        continue;
      }
      stroke(i === sel ? 'black' : 'white');
      strokeWeight(1.5);
      fill(c);
      circle(cx(i), p.y, d);
    }
    noStroke();
    fill('black');
    textAlign(CENTER, TOP);
    textSize(ts);
    text(layerName(i), cx(i), y0 + 7);
    textStyle(BOLD);
    textSize(ts + 2);
    text(n, cx(i), y0 + ts + 10);
    textStyle(NORMAL);
    hits.push({ i, x: cx(i) - gap / 2, y: y0, w: gap, h });
  });

  noStroke();
  fill('dimgray');
  textAlign(CENTER, CENTER);
  textSize(narrow ? 11 : 13);
  text(narrow ? 'Click a layer to select it, then set its size with the slider.'
    : 'Click a layer to select it. The slider sets its number of neurons. Blue numbers change when it changes.', x0 + w / 2, y0 + h - 13);
}

// The nn.Sequential code for the network, with the parameter count of every layer as a comment
function drawCode(x0, y0, w, h, layers, narrow) {
  panel(x0, y0, w, h, 'white');
  const ts = narrow ? 12 : 14, headH = narrow ? 22 : 28, pitch = (h - headH - 6) / (MAX_HIDDEN + 4);
  const total = layers.reduce((sum, l) => sum + l.params, 0);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text('PyTorch code', x0 + 10, y0 + (narrow ? 5 : 8));
  textStyle(NORMAL);
  fill('dimgray');
  textSize(narrow ? 11 : 12);
  textAlign(RIGHT, TOP);
  text('needs: import torch.nn as nn', x0 + w - 10, y0 + (narrow ? 7 : 10));

  // [code, comment, indented, highlighted]
  const rows = [['model = nn.Sequential(', '', false, false]];
  layers.forEach((l, i) => rows.push(['nn.Linear(' + l.nin + ', ' + l.nout + ')' + (i < layers.length - 1 ? ', nn.ReLU(),' : ''),
    '# ' + l.nin + '×' + l.nout + ' + ' + l.nout + ' = ' + fmt(l.params), true, l.hot]));
  rows.push([')', '', false, false]);
  rows.push(['total = sum(p.numel() for p in model.parameters())', '# ' + fmt(total), false, false]);

  textSize(ts);
  rows.forEach(([code, comment, indented, hot], k) => {
    const y = y0 + headH + k * pitch, tx = x0 + (indented ? (narrow ? 26 : 32) : 10);
    if (hot) {
      fill('lightyellow');
      stroke('goldenrod');
      strokeWeight(1);
      rect(x0 + 5, y + 1, w - 10, pitch - 2, 4);
    }
    noStroke();
    fill('black');
    textAlign(LEFT, CENTER);
    text(code, tx, y + pitch / 2 + 1);
    const last = k === rows.length - 1;
    fill('seagreen');
    if (last) textStyle(BOLD);
    text(comment, last ? tx + textWidth(code) + 10 : x0 + w * 0.55, y + pitch / 2 + 1);
    textStyle(NORMAL);
  });
}

// A bar for the parameters of each nn.Linear, then the totals
function drawParams(x0, y0, w, h, layers, narrow) {
  panel(x0, y0, w, h, 'lightyellow');
  const ts = narrow ? 11 : 13, lh = narrow ? 14 : 19, headH = narrow ? 22 : 28;
  const weights = layers.reduce((sum, l) => sum + l.weights, 0), biases = layers.reduce((sum, l) => sum + l.biases, 0);
  const biggest = Math.max(...layers.map(l => l.params)), nOut = sizes[sizes.length - 1];
  const labelW = narrow ? 62 : 74, countW = narrow ? 44 : 52, barW = w - 20 - labelW - countW - 6;
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text('Parameters in each nn.Linear', x0 + 10, y0 + (narrow ? 5 : 8));
  textStyle(NORMAL);
  textSize(ts);
  layers.forEach((l, i) => {
    const y = y0 + headH + i * lh + lh / 2;
    noStroke();
    fill('black');
    textAlign(LEFT, CENTER);
    text(l.nin + ' → ' + l.nout, x0 + 10, y);
    fill(l.hot ? 'royalblue' : 'lightsteelblue');
    rect(x0 + 10 + labelW, y - lh / 2 + 3, Math.max(2, barW * l.params / biggest), lh - 6, 3);
    fill('black');
    textAlign(RIGHT, CENTER);
    text(fmt(l.params), x0 + w - 10, y);
  });

  // totals sit at the bottom of the panel
  let y = y0 + h - (narrow ? 44 : 60);
  stroke('silver');
  line(x0 + 10, y - 4, x0 + w - 10, y - 4);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  text('Weights ' + fmt(weights) + ' + biases ' + fmt(biases), x0 + 10, y);
  textStyle(BOLD);
  textAlign(RIGHT, TOP);
  text('= ' + fmt(weights + biases) + ' total', x0 + w - 10, y);
  textStyle(NORMAL);
  fill('dimgray');
  textAlign(LEFT, TOP);
  text(nOut === 1 ? 'Output of 1 neuron: one number, as in regression.'
    : 'Output of ' + nOut + ' neurons: one score per class (' + nOut + ' classes).', x0 + 10, y + lh + 2, w - 20, 2 * lh + 4);
}

function mousePressed() {
  for (const r of hits) {
    if (mouseX >= r.x && mouseX <= r.x + r.w && mouseY >= r.y && mouseY <= r.y + r.h) {
      sel = r.i;
      sizeSlider.value(sizes[sel]);
      return;
    }
  }
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  sizeSlider.size(canvasWidth - sliderLeftMargin - margin);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
