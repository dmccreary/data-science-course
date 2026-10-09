// Slicing Playground
// CANVAS_HEIGHT: 590
// Bloom L3-L4 (Apply, Analyze): students type a row slice and a column slice for a 6 x 8
// array, see which elements a[rows, columns] selects and what the result looks like, and
// work out the slices that solve three challenges.
//
// Model: NumPy basic indexing. Each axis takes start:stop:step or a single index.
//   - missing start/stop mean "from the end you are leaving" and "to the far end"
//   - a negative start or stop counts from the end (n is added), then both are clipped
//   - stop is never included; a negative step walks backwards; step 0 is an error
//   - a single index must be in range (IndexError) and removes that axis from the result
// The result of basic indexing is a view: writing through it changes the original array.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 510;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const ROWS = 6, COLS = 8;
// [label, row slice, column slice, challenge text]. A challenge sets a target and clears the boxes.
const EXAMPLES = [
  ['Examples and challenges'],
  ['a[1:4, 2:6]  a block', '1:4', '2:6'],
  ['a[:3, :]  first 3 rows', ':3', ':'],
  ['a[:, -1]  last column', ':', '-1'],
  ['a[::2, ::2]  every other', '::2', '::2'],
  ['a[::-1, :]  rows reversed', '::-1', ':'],
  ['a[-2:, -3:]  bottom right', '-2:', '-3:'],
  ['Challenge: the 4 corners', '::5', '::7', 'select only the four corners of a'],
  ['Challenge: last 2 rows, reversed', '-1:-3:-1', ':', 'select the last two rows, bottom row first'],
  ['Challenge: row 2, every 3rd column', '2', '::3', 'select every third column of row 2 as a 1-D array']
];

let data = [];                    // the array a, row by row
let rowInput, colInput, exampleSelect, zeroButton, resetButton;
let challenge = null;             // the EXAMPLES entry being attempted
let written = false;              // true after b[:] = 0 until a is reset

// ---- NumPy behaviour ----
// Python's slice.indices(n): fill in missing values, wrap negatives, clip, then walk.
function sliceIndices(start, stop, step, n) {
  const lower = step > 0 ? 0 : -1, upper = step > 0 ? n : n - 1;
  const fix = (v, missing) => v === null ? missing : v < 0 ? Math.max(v + n, lower) : Math.min(v, upper);
  const first = fix(start, step > 0 ? lower : upper), end = fix(stop, step > 0 ? upper : lower);
  const idx = [];
  for (let i = first; step > 0 ? i < end : i > end; i += step) idx.push(i);
  return { idx, first, end, step };
}
// One axis of a[rows, cols] for an axis of length n. Returns the selected positions in idx,
// or error (what NumPy raises), or hint when the text is not an index or a slice at all.
function parseAxis(str, n, axis) {
  const parts = str.trim().split(':').map(s => s.trim());
  if (str.trim() === '' || parts.length > 3 || !parts.every(s => /^(-?\d+)?$/.test(s))) return { hint: true };
  if (parts.length === 1) {
    const i = Number(parts[0]);
    if (i < -n || i >= n) return { error: 'IndexError: index ' + i + ' is out of bounds for axis ' + axis + ' with size ' + n };
    return { idx: [i < 0 ? i + n : i], scalar: true };
  }
  const [start, stop, step] = [0, 1, 2].map(k => parts[k] === undefined || parts[k] === '' ? null : Number(parts[k]));
  if (step === 0) return { error: 'ValueError: slice step cannot be zero' };
  return sliceIndices(start, stop, step === null ? 1 : step, n);
}
// b.shape for a[rows, cols]: an axis indexed with a single number disappears
function resultShape(rowSel, colSel) {
  return (rowSel.scalar ? [] : [rowSel.idx.length]).concat(colSel.scalar ? [] : [colSel.idx.length]);
}
const tupleText = shape => '(' + shape.join(', ') + (shape.length === 1 ? ',' : '') + ')';

function resetData() {
  data = [];
  for (let r = 0; r < ROWS; r++) data.push(Array.from({ length: COLS }, (v, c) => r * COLS + c));
  written = false;
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);
  resetData();

  rowInput = createInput('1:4');
  colInput = createInput('::2');
  [rowInput, colInput].forEach((input, k) => {
    input.parent(mainElement);
    input.position(32 + k * 88, drawHeight + 9);
    input.size(64);
    input.style('font-size', '16px');
  });

  exampleSelect = createSelect();
  exampleSelect.parent(mainElement);
  exampleSelect.position(210, drawHeight + 10);
  EXAMPLES.forEach(e => exampleSelect.option(e[0]));
  exampleSelect.changed(() => {
    const e = EXAMPLES.find(item => item[0] === exampleSelect.value());
    if (e.length === 1) return;
    challenge = e[3] ? e : null;
    rowInput.value(challenge ? ':' : e[1]);
    colInput.value(challenge ? ':' : e[2]);
  });

  zeroButton = createButton('b[:] = 0');
  zeroButton.parent(mainElement);
  zeroButton.position(10, drawHeight + 45);
  zeroButton.mousePressed(() => {
    const rowSel = parseAxis(rowInput.value(), ROWS, 0), colSel = parseAxis(colInput.value(), COLS, 1);
    if (!rowSel.idx || !colSel.idx) return;
    for (const r of rowSel.idx) for (const c of colSel.idx) data[r][c] = 0;
    written = true;
  });

  resetButton = createButton('Reset a');
  resetButton.parent(mainElement);
  resetButton.position(90, drawHeight + 45);
  resetButton.mousePressed(resetData);
  resizeControls();

  describe('A 6 by 8 array of the numbers 0 to 47. Two text boxes hold a row slice and a column slice. ' +
    'The selected elements are highlighted in the array, the resulting array and its shape are shown beside it, ' +
    'and each slice is explained as a start, a stop and a step. A button writes zeros through the slice to show ' +
    'that a slice is a view of the original array.', LABEL);
}

function resizeControls() { exampleSelect.size(min(250, canvasWidth - 220)); }

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600;
  const rowSel = parseAxis(rowInput.value(), ROWS, 0), colSel = parseAxis(colInput.value(), COLS, 1);
  const valid = Boolean(rowSel.idx && colSel.idx);
  const shape = valid ? resultShape(rowSel, colSel) : null;
  const count = valid ? rowSel.idx.length * colSel.idx.length : 0;
  rowInput.style('outline', rowSel.idx ? 'none' : '2px solid red');
  colInput.style('outline', colSel.idx ? 'none' : '2px solid red');
  zeroButton.elt.disabled = !valid || count === 0 || shape.length === 0;

  textWrap(WORD);
  textStyle(NORMAL);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Slicing Playground', canvasWidth / 2, 8);

  const fullW = canvasWidth - 2 * margin, top = narrow ? 34 : 44;
  const target = challenge ? [parseAxis(challenge[1], ROWS, 0), parseAxis(challenge[2], COLS, 1)] : null;
  if (narrow) {
    drawArray(margin, top, fullW, 208, rowSel, colSel, valid, target, narrow);
    drawResult(margin, top + 214, fullW, drawHeight - top - 222, rowSel, colSel, shape, target, narrow);
  } else {
    const leftW = fullW * 0.54;
    drawArray(margin, top, leftW, drawHeight - top - 8, rowSel, colSel, valid, target, narrow);
    drawResult(margin + leftW + 10, top, fullW - leftW - 10, drawHeight - top - 8, rowSel, colSel, shape, target, narrow);
  }

  // control labels: the boxes sit inside a[ , ]
  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('a[', 10, drawHeight + 22);
  text(',', 106, drawHeight + 22);
  text(']', 194, drawHeight + 22);
  fill('dimgray');
  textSize(narrow ? 12 : 14);
  text(narrow ? 'writes zeros through the slice' : 'writes zeros through the slice b, then look at a', 168, drawHeight + 57);
}

// The array a with its positive indices (top and left) and negative indices (bottom and right)
function drawArray(x, y, w, h, rowSel, colSel, valid, target, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = narrow ? 12 : 15, labelW = narrow ? 22 : 28, labelH = narrow ? 14 : 20;
  noStroke();
  fill('black');
  textStyle(BOLD);
  textSize(ts);
  textAlign(LEFT, TOP);
  text('a = np.arange(48).reshape(6, 8)', x + 10, y + 7);
  textStyle(NORMAL);

  const cw = (w - 16 - 2 * labelW) / COLS, ch = min(cw * 1.25, (h - 30 - 2 * labelH - (narrow ? 4 : 44)) / ROWS);
  const gx = x + 8 + labelW, gy = y + 28 + labelH;
  const rowOn = r => valid && rowSel.idx.includes(r), colOn = c => valid && colSel.idx.includes(c);
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const on = rowOn(r) && colOn(c);
      fill(on ? 'royalblue' : 'whitesmoke');
      stroke('silver');
      strokeWeight(1);
      rect(gx + c * cw, gy + r * ch, cw, ch);
      noStroke();
      fill(on ? 'white' : 'gray');
      textStyle(on ? BOLD : NORMAL);
      textSize(min(ts + 1, ch * 0.62));
      textAlign(CENTER, CENTER);
      text(data[r][c], gx + (c + 0.5) * cw, gy + (r + 0.5) * ch + 1);
    }
  }
  // challenge target: dashed orange frames
  if (target) {
    noFill();
    stroke('darkorange');
    strokeWeight(3);
    drawingContext.setLineDash([5, 4]);
    for (const r of target[0].idx) for (const c of target[1].idx) rect(gx + c * cw + 2, gy + r * ch + 2, cw - 4, ch - 4);
    drawingContext.setLineDash([]);
  }
  // index labels
  noStroke();
  textSize(narrow ? 11 : 13);
  for (let c = 0; c < COLS; c++) {
    textStyle(colOn(c) ? BOLD : NORMAL);
    fill(colOn(c) ? 'mediumblue' : 'black');
    textAlign(CENTER, BOTTOM);
    text(c, gx + (c + 0.5) * cw, gy - 3);
    fill('gray');
    textStyle(NORMAL);
    textAlign(CENTER, TOP);
    text(c - COLS, gx + (c + 0.5) * cw, gy + ROWS * ch + 3);
  }
  for (let r = 0; r < ROWS; r++) {
    textStyle(rowOn(r) ? BOLD : NORMAL);
    fill(rowOn(r) ? 'mediumblue' : 'black');
    textAlign(RIGHT, CENTER);
    text(r, gx - 6, gy + (r + 0.5) * ch);
    fill('gray');
    textStyle(NORMAL);
    textAlign(LEFT, CENTER);
    text(r - ROWS, gx + COLS * cw + 5, gy + (r + 0.5) * ch);
  }
  if (!narrow) {
    fill('dimgray');
    textSize(13);
    textAlign(LEFT, TOP);
    text('Black labels are indices. Gray labels are the same positions counted from the end.', x + 10,
      gy + ROWS * ch + labelH + 6, w - 20, 36);
  }
}

// One sentence for one axis: how the typed text becomes a list of positions
function axisText(sel, typed, word) {
  const label = word[0].toUpperCase() + word.slice(1) + 's  ' + typed.trim() + '  →  ';
  if (sel.hint) return label + 'not a slice. Type start:stop:step or one index, for example 1:4, ::2 or -1.';
  if (sel.error) return label + sel.error;
  if (sel.scalar) return label + 'the single ' + word + ' ' + sel.idx[0] + '. One index, not a slice, so this axis disappears.';
  return label + 'start ' + sel.first + ', ' + (sel.end < 0 ? 'run back past 0' : 'stop before ' + sel.end) + ', step ' +
    sel.step + ': ' + (sel.idx.length ? word + (sel.idx.length > 1 ? 's ' : ' ') + sel.idx.join(', ') : 'no ' + word + 's') + '.';
}

// The code, the meaning of each slice, the result b, and what a view is
function drawResult(x, y, w, h, rowSel, colSel, shape, target, narrow) {
  fill('lightyellow');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = narrow ? 12 : 15, lh = ts + (narrow ? 4 : 5), pad = narrow ? 10 : 12, iw = w - 2 * pad;
  let cy = y + (narrow ? 6 : 10);
  const say = (str, lines, ink, style, sx, sw) => {
    noStroke();
    fill(ink);
    textStyle(style || NORMAL);
    textSize(ts);
    textLeading(lh);
    textAlign(LEFT, TOP);
    text(str, sx || x + pad, cy, sw || iw, lines * lh + 3);
    cy += lines * lh + (narrow ? 3 : 9);
    textStyle(NORMAL);
  };
  const rowText = rowInput.value().trim(), colText = colInput.value().trim();
  say('b = a[' + rowText + ', ' + colText + ']', 1, 'black', BOLD);
  say(axisText(rowSel, rowText, 'row'), 2, rowSel.idx ? 'black' : 'firebrick');
  say(axisText(colSel, colText, 'column'), 2, colSel.idx ? 'black' : 'firebrick');
  if (!shape) return;

  // b as NumPy would print it: a 1-D result is a single row, whichever axis it came from
  const flat = [];
  for (const r of rowSel.idx) for (const c of colSel.idx) flat.push(data[r][c]);
  const cols = shape.length === 0 ? 1 : shape[shape.length - 1], rows = cols ? flat.length / cols : 0;
  const gridW = narrow ? iw * 0.52 : iw, gridTop = cy + lh + 2;
  say('b.shape → ' + tupleText(shape) + (shape.length === 0 ? '   one number' : shape.length === 1 ? '   1-D' : ''), 1,
    'mediumblue', BOLD);
  const cw = min(40, gridW / max(cols, 1)), ch = narrow ? 17 : 24;
  for (let i = 0; i < flat.length; i++) {
    const bx = x + pad + (i % cols) * cw, by = gridTop + floor(i / cols) * ch;
    fill('royalblue');
    stroke('white');
    strokeWeight(1);
    rect(bx, by, cw, ch);
    noStroke();
    fill('white');
    textStyle(BOLD);
    textSize(min(ts, 13));
    textAlign(CENTER, CENTER);
    text(flat[i], bx + cw / 2, by + ch / 2 + 1);
  }
  textStyle(NORMAL);

  // notes: beside the grid when narrow, under it when wide
  const nx = narrow ? x + pad + gridW + 8 : x + pad, nw = narrow ? iw - gridW - 8 : iw;
  cy = narrow ? gridTop - lh : gridTop + max(rows, 1) * ch + 10;
  const view = shape.length === 0 ? 'A single number is a copy, not a view.'
    : flat.length === 0 ? 'b is empty: no elements were selected. Nothing is wrong, NumPy just returns an empty array.'
    : written ? 'The zeros in a were written through b. A slice is a view: it uses a\'s memory. Use .copy() for a separate array.'
      : 'b is a view of a: nothing was copied. Press b[:] = 0 and watch a.';
  say(view, narrow ? 5 : 3, written ? 'firebrick' : 'black', NORMAL, nx, nw);
  if (!target) return;
  const same = (p, q) => p.idx.join() === q.idx.join();
  const cells = same(rowSel, target[0]) && same(colSel, target[1]);
  const solved = cells && tupleText(shape) === tupleText(resultShape(target[0], target[1]));
  say(solved ? 'Challenge solved: b is exactly the target.'
    : cells ? 'Right cells, but the target shape is ' + tupleText(resultShape(target[0], target[1])) +
      '. A single index, not a slice, drops an axis.'
      : 'Challenge: ' + challenge[3] + ' (orange frames).', narrow ? 5 : 3, solved ? 'darkgreen' : 'chocolate', BOLD, nx, nw);
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  resizeControls();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
