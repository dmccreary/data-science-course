// Array Shape Visualizer
// CANVAS_HEIGHT: 580
// Bloom L2-L3 (Understand, Apply): students type a new shape for a = np.arange(n), see the
// same numbers laid out as a row, a table, or a stack of tables, and find out which shapes
// NumPy accepts.
//
// Model: numpy.reshape. The sizes in the new shape must multiply to a.size. One size may be
// -1, and NumPy replaces it with a.size divided by the product of the others, when that is
// a whole number. Elements keep their order (row-major): position p of a lands at the index
// you get by repeated division, for example p = i * columns + j in a table. Error messages
// are worded as NumPy 2 words them.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 500;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let sliderLeftMargin = 170;
let defaultTextSize = 16;

// [label, n, text typed into reshape( )]
const EXAMPLES = [
  ['Examples', 0, ''],
  ['reshape(12): one row', 12, '12'],
  ['reshape(3, 4): table', 12, '3, 4'],
  ['reshape(2, 2, 3): 2 tables', 12, '2, 2, 3'],
  ['reshape(4, -1): auto size', 12, '4, -1'],
  ['reshape(-1, 1): one column', 12, '-1, 1'],
  ['reshape(2, 2, 2): 8 numbers', 8, '2, 2, 2'],
  ['reshape(5, 3): error', 12, '5, 3'],
  ['reshape(5, -1): error', 12, '5, -1']
];

let shapeInput, exampleSelect, sizeSlider;
let selected = 7;                 // position in a of the element being followed
let cellBoxes = [];               // clickable cells: { x, y, w, h, p }

// ---- NumPy behaviour ----
// "3, 4" or "(3, 4)" → [3, 4]. Returns null when the text is not a list of whole numbers.
function parseShape(str) {
  const parts = str.trim().replace(/^\(/, '').replace(/\)$/, '').replace(/,\s*$/, '').split(',').map(s => s.trim());
  return parts.every(s => /^-?\d+$/.test(s)) ? parts.map(Number) : null;
}
const product = list => list.reduce((p, d) => p * d, 1);
// How NumPy prints the requested shape in its error: leading unknowns are dropped, later ones read "newaxis".
function requestText(req) {
  let i = 0;
  while (i < req.length && req[i] < 0) i++;
  return '(' + req.slice(i).map(d => d < 0 ? 'newaxis' : d).join(',') + (req.length === 1 ? ',' : '') + ')';
}
// a.reshape(req) for an array of `size` elements: { shape } or { error }. Any negative size is the unknown.
function reshapeResult(size, req) {
  const unknown = req.filter(d => d < 0).length;
  if (unknown > 1) return { error: 'can only specify one unknown dimension' };
  const known = product(req.filter(d => d >= 0));
  const fits = unknown === 0 ? known === size : known > 0 && size % known === 0;
  if (!fits) return { error: 'cannot reshape array of size ' + size + ' into shape ' + requestText(req) };
  return { shape: req.map(d => d < 0 ? size / known : d) };
}
// position p in the flat array → index in an array of this shape (np.unravel_index)
function unravel(p, shape) {
  const index = [];
  for (let k = shape.length - 1; k >= 0; k--) { index.unshift(p % shape[k]); p = Math.floor(p / shape[k]); }
  return index;
}
// For a failed reshape, the grid the request describes, with an unknown size rounded up
function attemptShape(size, req) {
  if (req.filter(d => d < 0).length > 1 || req.includes(0)) return null;
  const known = product(req.filter(d => d >= 0));
  return req.map(d => d < 0 ? Math.ceil(size / known) : d);
}
const tupleText = shape => '(' + shape.join(', ') + (shape.length === 1 ? ',' : '') + ')';
const count = (n, word) => n + ' ' + word + (n === 1 ? '' : 's');

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  shapeInput = createInput('3, 4');
  shapeInput.parent(mainElement);
  shapeInput.position(90, drawHeight + 9);
  shapeInput.size(80);
  shapeInput.style('font-size', '16px');
  shapeInput.input(() => exampleSelect.selected(EXAMPLES[0][0]));

  exampleSelect = createSelect();
  exampleSelect.parent(mainElement);
  exampleSelect.position(200, drawHeight + 10);
  EXAMPLES.forEach(e => exampleSelect.option(e[0]));
  exampleSelect.changed(() => {
    const e = EXAMPLES.find(item => item[0] === exampleSelect.value());
    if (e[1] === 0) return;
    sizeSlider.value(e[1]);
    shapeInput.value(e[2]);
  });

  sizeSlider = createSlider(2, 12, 12, 1);
  sizeSlider.parent(mainElement);
  sizeSlider.position(sliderLeftMargin, drawHeight + 45);
  sizeSlider.size(canvasWidth - sliderLeftMargin - margin);

  describe('An array of whole numbers shown as a flat row and, below it, in the shape typed into a reshape box: ' +
    'a row, a table, or a stack of tables. Panels show the shape, the number of dimensions, the size, the index of ' +
    'a selected element, and the NumPy error message when the requested shape is impossible.', LABEL);
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600;
  const n = sizeSlider.value();
  selected = min(selected, n - 1);
  const req = parseShape(shapeInput.value());
  const res = req ? reshapeResult(n, req) : {};
  cellBoxes = [];

  textWrap(WORD);
  textStyle(NORMAL);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Array Shape Visualizer', canvasWidth / 2, 8);

  const fullW = canvasWidth - 2 * margin, top = 136;
  const view = res.shape || (res.error ? attemptShape(n, req) : null);
  drawFlat(margin, narrow ? 38 : 44, fullW, n, res.shape, narrow);
  if (narrow) {
    drawShaped(margin, top, fullW, 228, n, view, !res.shape, narrow);
    drawInfo(margin, top + 234, fullW, drawHeight - top - 242, n, req, res, narrow);
  } else {
    const leftW = fullW * 0.56;
    drawShaped(margin, top, leftW, drawHeight - top - 8, n, view, !res.shape, narrow);
    drawInfo(margin + leftW + 10, top, fullW - leftW - 10, drawHeight - top - 8, n, req, res, narrow);
  }
  cursor(cellBoxes.some(mouseOver) ? HAND : ARROW);

  // control labels
  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('a.reshape(', 10, drawHeight + 22);
  text(')', 180, drawHeight + 22);
  text('Elements in a: ' + n, 10, drawHeight + 56);
}

// One element. The color deepens with the value, and the followed element gets a red frame.
function valueCell(x, y, w, h, p, n) {
  if (p >= n) {                   // a slot with no number to put in it
    fill('mistyrose');
    stroke('firebrick');
    strokeWeight(1);
    rect(x, y, w, h);
    noStroke();
    fill('firebrick');
    textAlign(CENTER, CENTER);
    textSize(constrain(min(w * 0.5, h * 0.7), 11, 16));
    text('?', x + w / 2, y + h / 2 + 1);
    return;
  }
  fill(lerpColor(color('white'), color('deepskyblue'), n > 1 ? p / (n - 1) : 0));
  stroke('slategray');
  strokeWeight(1);
  rect(x, y, w, h);
  if (p === selected) {
    noFill();
    stroke('red');
    strokeWeight(3);
    rect(x + 1.5, y + 1.5, w - 3, h - 3);
  }
  noStroke();
  fill('black');
  textAlign(CENTER, CENTER);
  textSize(constrain(min(w * 0.5, h * 0.7), 11, 16));
  text(p, x + w / 2, y + h / 2 + 1);
  cellBoxes.push({ x, y, w, h, p });
}

// The array as NumPy stores it: one row of n numbers. Brackets mark the runs that fill the last axis of b.
function drawFlat(x, y, w, n, shape, narrow) {
  const ts = narrow ? 12 : 15;
  noStroke();
  fill('black');
  textSize(ts);
  textAlign(LEFT, TOP);
  text('a = np.arange(' + n + ')', x, y);
  fill('dimgray');
  textAlign(RIGHT, TOP);
  text('a.shape → (' + n + ',)    a.ndim → 1    a.size → ' + n, x + w, y);

  const cw = min(48, w / n), sx = x + (w - cw * n) / 2, cy = y + 22, ch = 30;
  for (let p = 0; p < n; p++) valueCell(sx + p * cw, cy, cw, ch, p, n);
  if (!shape || shape.length > 3) return;
  const run = shape[shape.length - 1], lead = shape.slice(0, -1);
  for (let c = 0; c < n / run; c++) {
    const bx = sx + c * run * cw + 2, bw = run * cw - 4, by = cy + ch + 4;
    const ink = c % 2 ? 'chocolate' : 'seagreen';
    stroke(ink);
    strokeWeight(2);
    line(bx, by, bx, by + 5);
    line(bx, by + 5, bx + bw, by + 5);
    line(bx + bw, by, bx + bw, by + 5);
    const label = lead.length === 0 ? 'b: all ' + n + ' in one row' : 'b[' + unravel(c, lead).join(', ') + ']';
    noStroke();
    fill(ink);
    textSize(narrow ? 11 : 13);
    textAlign(CENTER, TOP);
    if (textWidth(label) < bw + 4) text(label, bx + bw / 2, by + 8);
  }
}

// The same numbers in the new shape: 1 axis is a row, 2 axes a table, 3 axes a stack of tables.
// When the reshape fails, the requested grid is drawn anyway so the mismatch can be seen.
function drawShaped(x, y, w, h, n, shape, failed, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = narrow ? 12 : 15;
  const [blocks, rows, cols] = shape ? [1, 1, 1].slice(shape.length).concat(shape) : [0, 0, 0];
  noStroke();
  textSize(ts);
  if (!shape || shape.length > 3 || rows > 12 || blocks * cols > 16) {
    fill('dimgray');
    textAlign(CENTER, CENTER);
    text(!failed ? 'b has ' + shape.length + ' axes. This view draws up to 3.'
      : shapeInput.value().trim() === '' ? 'Type a shape to see the array.' : 'No array to draw. This reshape fails.',
    x + 10, y, w - 20, h);
    return;
  }
  const slots = product(shape);
  fill(failed ? 'firebrick' : 'black');
  textAlign(CENTER, TOP);
  text(failed ? (slots > n ? count(slots, 'slot') + ' but only ' + n + ' numbers: ' + (slots - n) + ' stay empty'
    : 'Only ' + count(slots, 'slot') + ' for ' + n + ' numbers: ' + (n - slots) + ' left over')
    : shape.length === 1 ? 'axis 0: ' + count(cols, 'element')
      : (shape.length === 3 ? 'axis 0: ' + count(blocks, 'table') + '   ·   ' : '') +
        'axis ' + (shape.length - 2) + ': ' + count(rows, 'row') + '   ·   axis ' + (shape.length - 1) + ': ' +
        count(cols, 'column'), x + w / 2, y + 9);

  const labelW = 20, gap = 12, headH = shape.length === 3 ? 34 : 18, availH = h - 34 - headH - 10;
  const cw = min(narrow ? 48 : 64, (w - 24 - labelW - (blocks - 1) * gap) / (blocks * cols));
  const ch = min(narrow ? 38 : 50, availH / rows);
  const gx = x + (w - blocks * cols * cw - (blocks - 1) * gap - labelW) / 2 + labelW;
  const gy = y + 34 + headH + (availH - rows * ch) / 3;
  for (let k = 0; k < blocks; k++) {
    const bx = gx + k * (cols * cw + gap);
    noStroke();
    textAlign(CENTER, BOTTOM);
    if (shape.length === 3) {
      fill('black');
      textSize(ts);
      text((cols * cw > 34 ? 'b[' : '[') + k + ']', bx + cols * cw / 2, gy - 17);
    }
    fill('dimgray');
    textSize(narrow ? 11 : 12);
    for (let j = 0; j < cols && shape.length > 1; j++) text(j, bx + (j + 0.5) * cw, gy - 2);
    textAlign(RIGHT, CENTER);
    for (let i = 0; i < rows && k === 0 && shape.length > 1; i++) text(i, bx - 5, gy + (i + 0.5) * ch);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const p = (k * rows + i) * cols + j;
        if (p < n || failed) valueCell(bx + j * cw, gy + i * ch, cw, ch, p, n);
      }
    }
  }
}

// What NumPy reports for b, or the error it raises, and where the followed element went
function drawInfo(x, y, w, h, n, req, res, narrow) {
  fill('lightyellow');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = narrow ? 12 : 16, lh = ts + 4, pad = narrow ? 10 : 14;
  let cy = y + (narrow ? 6 : 14);
  const say = (str, lines, ink, style) => {
    noStroke();
    fill(ink);
    textStyle(style || NORMAL);
    textSize(ts);
    textLeading(lh);
    textAlign(LEFT, TOP);
    text(str, x + pad, cy, w - 2 * pad, lines * lh + 3);
    cy += lines * lh + (narrow ? 2 : 10);
    textStyle(NORMAL);
  };

  say('b = a.reshape(' + (req ? req.join(', ') : ' ? ') + ')', 1, 'black', BOLD);
  if (!req) {
    say('Type the new shape as whole numbers separated by commas, for example 3, 4 or 2, -1.', 3, 'dimgray');
    return;
  }
  const known = product(req.filter(d => d >= 0)), unknown = req.filter(d => d < 0).length;
  if (res.error) {
    say('ValueError: ' + res.error, narrow ? 2 : 3, 'firebrick', BOLD);
    say(unknown > 1 ? 'Only one size can be -1. NumPy cannot work out two unknown sizes.'
      : unknown === 1 ? (known === 0 ? 'A size of 0 leaves no room for ' + n + ' elements.'
        : n + ' is not a multiple of ' + known + ', so no whole number can replace -1.')
        : req.join(' × ') + (req.length > 1 ? ' = ' + known : '') + ', but a has ' + n + ' elements. Reshaping never adds ' +
          'or drops elements, so the sizes must multiply to ' + n + '.', narrow ? 3 : 4, 'black');
    return;
  }
  const shape = res.shape;
  if (narrow) say('b.shape → ' + tupleText(shape) + '     b.ndim → ' + shape.length + '     b.size → ' + n, 1, 'black');
  else {
    say('b.shape → ' + tupleText(shape), 1, 'black');
    say('b.ndim → ' + shape.length + '        b.size → ' + n, 1, 'black');
  }
  say(unknown === 1 ? 'The negative size is worked out for you: ' + n + ' ÷ ' + known + ' = ' + n / known + '.'
    : shape.join(' × ') + ' = ' + n + ', the size of a, so this shape works.', narrow ? 1 : 2, 'darkgreen');
  say('Same numbers, same order. Each run of ' + shape[shape.length - 1] + ' fills the last axis.', narrow ? 1 : 2, 'black');
  // where the followed element lands: strides are the products of the sizes to the right
  const index = unravel(selected, shape);
  const terms = index.map((v, k) => k === shape.length - 1 ? String(v) : v + ' × ' + product(shape.slice(k + 1)));
  say('a[' + selected + ']  is  b[' + index.join(', ') + ']', 1, 'red', BOLD);
  say((shape.length > 1 ? 'because ' + selected + ' = ' + terms.join(' + ') + '. ' : '') + 'Click any cell to follow it.',
    2, 'dimgray');
}

// Clicking a cell in either view selects that element.
function mouseOver(b) {
  return mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h;
}
function mousePressed() {
  const box = cellBoxes.find(mouseOver);
  if (box) selected = box.p;
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
