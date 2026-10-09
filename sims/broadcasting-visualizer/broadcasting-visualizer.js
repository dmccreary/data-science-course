// Broadcasting Visualizer
// CANVAS_HEIGHT: 550
// Bloom L2-L3 (Understand, Apply): students pick the shapes of two arrays and an operation,
// see how NumPy stretches the smaller array to fit the larger one, and learn to predict
// which pairs of shapes work.
//
// Model: NumPy broadcasting. Write the two shapes right-aligned. A missing size counts as 1.
// Each pair of sizes must be equal, or one of them must be 1; the result takes the larger.
// An axis of size 1 supplies its single element at every position along that axis. If any
// pair fails, NumPy raises
//   ValueError: operands could not be broadcast together with shapes (3,4) (2,4)

let containerWidth;
let canvasWidth = 400;
let drawHeight = 470;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const A_SHAPES = [[3, 3], [3, 4], [3, 1], [3]];
const B_SHAPES = [[], [3], [4], [3, 1], [1, 4], [2, 4], [3, 3]];
const OPS = { '+': (a, b) => a + b, '-': (a, b) => a - b, '*': (a, b) => a * b, '/': (a, b) => a / b };
// [label, index into A_SHAPES, index into B_SHAPES]
const EXAMPLES = [
  ['Examples'],
  ['Scalar and matrix', 0, 0],
  ['Row and matrix', 0, 1],
  ['Column and matrix', 0, 3],
  ['Column and row: both stretch', 2, 4],
  ['Mismatch: (3, 4) and (3,)', 1, 1],
  ['Mismatch: (3, 4) and (2, 4)', 1, 5]
];

let aSelect, bSelect, opSelect, exampleSelect, stretchCheckbox;
let sel = [1, 2];                 // result cell whose calculation is spelled out
let cellBoxes = [];               // clickable cells: { x, y, w, h, i, j }

// ---- NumPy behaviour ----
const sizeOf = shape => shape.reduce((p, d) => p * d, 1);
const tupleText = shape => '(' + shape.join(', ') + (shape.length === 1 ? ',' : '') + ')';
const errorShape = shape => '(' + shape.join(',') + (shape.length === 1 ? ',' : '') + ')';   // as NumPy prints it
const padTo = (shape, nd) => Array(nd - shape.length).fill(1).concat(shape);
// Right-align the shapes and compare them size by size. shape is null when they do not fit.
function broadcast(sa, sb) {
  const nd = Math.max(sa.length, sb.length), pa = padTo(sa, nd), pb = padTo(sb, nd);
  const ok = pa.map((d, k) => d === pb[k] || d === 1 || pb[k] === 1);
  return { pa, pb, ok, shape: ok.every(Boolean) ? pa.map((d, k) => Math.max(d, pb[k])) : null,
    error: 'operands could not be broadcast together with shapes ' + errorShape(sa) + ' ' + errorShape(sb) };
}
// Example data: A counts 1, 2, 3, ... and B counts in tens (hundreds for a column, as in the chapter)
function valuesA(shape) { return Array.from({ length: sizeOf(shape) }, (v, k) => k + 1); }
function valuesB(shape) {
  const unit = shape.length === 2 && shape[1] === 1 ? 100 : 10;
  return Array.from({ length: sizeOf(shape) }, (v, k) => unit * (k + 1));
}
// The element an array of this shape supplies at row i, column j of the result
function pick(values, shape, i, j) {
  const p = padTo(shape, 2);
  return values[(p[0] === 1 ? 0 : i) * p[1] + (p[1] === 1 ? 0 : j)];
}
// The whole result as rows, or null when the shapes do not broadcast
function resultRows(sa, sb, op) {
  const b = broadcast(sa, sb);
  if (!b.shape) return null;
  const [rows, cols] = padTo(b.shape, 2), va = valuesA(sa), vb = valuesB(sb);
  return Array.from({ length: rows }, (r, i) => Array.from({ length: cols }, (c, j) => OPS[op](pick(va, sa, i, j), pick(vb, sb, i, j))));
}
const show = v => String(Number(v.toFixed(3)));         // quotients are shown to 3 decimals

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);
  const place = (control, x, row) => { control.parent(mainElement); control.position(x, drawHeight + 10 + row * 35); };

  aSelect = createSelect();
  place(aSelect, 30, 0);
  A_SHAPES.forEach(s => aSelect.option(tupleText(s)));
  opSelect = createSelect();
  place(opSelect, 112, 0);
  Object.keys(OPS).forEach(op => opSelect.option(op));
  bSelect = createSelect();
  place(bSelect, 186, 0);
  B_SHAPES.forEach(s => bSelect.option(s.length ? tupleText(s) : '()  scalar'));
  bSelect.selected('(3,)');
  aSelect.changed(() => exampleSelect.selected(EXAMPLES[0][0]));
  bSelect.changed(() => exampleSelect.selected(EXAMPLES[0][0]));

  exampleSelect = createSelect();
  place(exampleSelect, 10, 1);
  EXAMPLES.forEach(e => exampleSelect.option(e[0]));
  exampleSelect.changed(() => {
    const e = EXAMPLES.find(item => item[0] === exampleSelect.value());
    if (e.length === 1) return;
    aSelect.elt.selectedIndex = e[1];
    bSelect.elt.selectedIndex = e[2];
  });

  stretchCheckbox = createCheckbox(' Show stretched copies', true);
  place(stretchCheckbox, 224, 1);
  stretchCheckbox.style('font-size', '16px');

  describe('Two arrays A and B drawn as grids of numbers with an operation between them and the result on the right. ' +
    'The smaller array is shown stretched with faded copies so that it matches the larger one. A table below lines up ' +
    'the two shapes from the right, marks each pair of sizes as compatible or not, and shows the NumPy error ' +
    'message when the shapes cannot be broadcast.', LABEL);
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
  const sa = A_SHAPES[aSelect.elt.selectedIndex], sb = B_SHAPES[bSelect.elt.selectedIndex], op = opSelect.value();
  const b = broadcast(sa, sb), rows = resultRows(sa, sb, op);
  const full = b.shape ? padTo(b.shape, 2) : null;
  if (full) sel = [min(sel[0], full[0] - 1), min(sel[1], full[1] - 1)];
  cellBoxes = [];

  textWrap(WORD);
  textStyle(NORMAL);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Broadcasting Visualizer', canvasWidth / 2, 8);

  // three grids in a row: A  op  B  =  result. Rows line up so matching positions sit side by side.
  const ts = narrow ? 12 : 15, opW = narrow ? 16 : 34, top = narrow ? 76 : 94, side = narrow ? 8 : margin;
  const slotW = (canvasWidth - 2 * side - 2 * opW) / 3;
  const widest = max(padTo(sa, 2)[1], padTo(sb, 2)[1]);        // columns in the widest grid
  const cw = min(narrow ? 38 : 54, slotW / widest), ch = narrow ? 28 : 42;
  const stretch = stretchCheckbox.checked() && full;
  const va = valuesA(sa), vb = valuesB(sb);
  [[sa, va, 'A', 'lightskyblue'], [sb, vb, 'B', 'gold']].forEach(([shape, values, name, tint], k) => {
    const own = padTo(shape, 2), shown = stretch ? full : own;
    const cx = side + slotW / 2 + k * (slotW + opW);
    heading(cx, top, name, stretch && tupleText(shown) !== tupleText(shape) ? tupleText(shape) + ' acts like ' + tupleText(b.shape)
      : 'shape ' + tupleText(shape), ts);
    drawGrid(cx, top, cw, ch, shown, (i, j) => pick(values, shape, i, j), (i, j) => i >= own[0] || j >= own[1], tint, Boolean(full), ts);
  });
  const cxR = side + slotW / 2 + 2 * (slotW + opW);
  heading(cxR, top, 'A ' + op + ' B', full ? 'shape ' + tupleText(b.shape) + (op === '/' ? ', floats' : '') : 'no result', ts);
  if (full) drawGrid(cxR, top, cw, ch, full, (i, j) => rows[i][j], () => false, 'palegreen', true, ts);
  noStroke();
  textAlign(CENTER, CENTER);
  textStyle(BOLD);
  textSize(narrow ? 18 : 24);
  fill('black');
  text(op, side + slotW + opW / 2, top + ch * 1.5);
  text('=', side + 2 * slotW + 1.5 * opW, top + ch * 1.5);
  if (!full) {
    fill('firebrick');
    textSize(narrow ? 30 : 40);
    text('✗', cxR, top + ch * 1.5);
  }
  textStyle(NORMAL);

  // the calculation behind the framed cell
  const calcY = top + 3 * ch + (narrow ? 8 : 12);
  textSize(ts);
  textAlign(CENTER, TOP);
  if (full) {
    const [i, j] = sel, x = pick(va, sa, i, j), y = pick(vb, sb, i, j);
    const at = shape => shape.length === 0 ? '' : shape.length === 1 ? '[' + j + ']'
      : '[' + (shape[0] === 1 ? 0 : i) + ', ' + (shape[1] === 1 ? 0 : j) + ']';
    fill('red');
    text('Framed cell:  A' + at(sa) + ' ' + op + ' B' + at(sb) + '  =  ' + x + ' ' + op + ' ' + y + '  =  ' + show(rows[i][j]) +
      (narrow ? '' : '      (click any cell)'), canvasWidth / 2, calcY);
  } else {
    fill('firebrick');
    text('These shapes cannot be combined. The table shows why.', canvasWidth / 2, calcY);
  }
  cursor(cellBoxes.some(mouseOver) ? HAND : ARROW);

  const panelY = calcY + (narrow ? 24 : 30);
  drawAnalysis(margin, panelY, canvasWidth - 2 * margin, drawHeight - panelY - 8, sa, sb, b, narrow);

  // control labels
  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('A', 10, drawHeight + 22);
  text('B', 166, drawHeight + 22);
}

function heading(cx, top, name, detail, ts) {
  noStroke();
  textAlign(CENTER, BOTTOM);
  fill('black');
  textStyle(BOLD);
  textSize(ts + 2);
  text(name, cx, top - ts - 9);
  textStyle(NORMAL);
  fill('dimgray');
  textSize(ts);
  text(detail, cx, top - 5);
}

// A grid centered on cx. Cells outside the array's own shape are the stretched copies, drawn faded.
// The framed cell is the one that supplies the value: a size-1 axis always supplies its only element.
function drawGrid(cx, top, cw, ch, shown, valueAt, isCopy, tint, framed, ts) {
  const gx = cx - shown[1] * cw / 2;
  for (let i = 0; i < shown[0]; i++) {
    for (let j = 0; j < shown[1]; j++) {
      const x = gx + j * cw, y = top + i * ch, copy = isCopy(i, j);
      const c = color(tint);
      if (copy) c.setAlpha(70);
      fill(c);
      stroke(copy ? 'darkgray' : 'dimgray');
      strokeWeight(1);
      if (copy) drawingContext.setLineDash([3, 3]);
      rect(x, y, cw, ch);
      drawingContext.setLineDash([]);
      if (framed && i === (shown[0] === 1 ? 0 : sel[0]) && j === (shown[1] === 1 ? 0 : sel[1])) {
        noFill();
        stroke('red');
        strokeWeight(3);
        rect(x + 1.5, y + 1.5, cw - 3, ch - 3);
      }
      noStroke();
      fill(copy ? 'gray' : 'black');
      textAlign(CENTER, CENTER);
      textSize(min(ts, cw * 0.42));
      text(show(valueAt(i, j)), x + cw / 2, y + ch / 2 + 1);
      cellBoxes.push({ x, y, w: cw, h: ch, i, j });
    }
  }
}

// The shape check: sizes lined up from the right, one column per axis
function drawAnalysis(x, y, w, h, sa, sb, b, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = narrow ? 12 : 15, nd = b.pa.length, rowH = narrow ? 21 : 30;
  noStroke();
  fill('black');
  textStyle(BOLD);
  textSize(ts + 1);
  textAlign(LEFT, TOP);
  text('Shape check: line the shapes up from the right', x + 10, y + 7);
  textStyle(NORMAL);

  const labelW = narrow ? 86 : 110, colW = narrow ? 56 : 68, tx = x + 12, ty = y + (narrow ? 28 : 34);
  const tableRows = [['', null], ['A  ' + tupleText(sa), b.pa, sa], ['B  ' + (sb.length ? tupleText(sb) : 'scalar ()'), b.pb, sb],
    ['result', b.shape]];
  tableRows.forEach(([label, sizes, own], r) => {
    const ry = ty + r * rowH;
    noStroke();
    fill('black');
    textSize(ts);
    textAlign(LEFT, CENTER);
    text(label, tx, ry + rowH / 2);
    for (let k = 0; k < nd; k++) {
      const cx = tx + labelW + k * colW;
      textAlign(CENTER, CENTER);
      if (r === 0) {
        fill('dimgray');
        text('axis ' + k, cx + colW / 2, ry + rowH / 2);
        continue;
      }
      const missing = own && k < nd - own.length;        // an axis the array does not have: counts as 1
      fill(b.ok[k] ? 'honeydew' : 'mistyrose');
      stroke(b.ok[k] ? 'seagreen' : 'firebrick');
      strokeWeight(r === 3 ? 2 : 1);
      rect(cx + 3, ry + 2, colW - 6, rowH - 4, 4);
      noStroke();
      fill(missing ? 'gray' : b.ok[k] ? 'black' : 'firebrick');
      textStyle(r === 3 ? BOLD : missing ? ITALIC : NORMAL);
      text(sizes ? (missing ? '(1)' : sizes[k]) : b.ok[k] ? max(b.pa[k], b.pb[k]) : '✗', cx + colW / 2, ry + rowH / 2 + 1);
      textStyle(NORMAL);
    }
  });

  // one sentence per axis, then the outcome
  const notes = [];
  for (let k = 0; k < nd; k++) {
    const p = b.pa[k], q = b.pb[k], who = p === 1 ? 'A' : 'B', own = p === 1 ? sa : sb;
    const way = nd === 1 ? '' : k === nd - 2 ? ' down the rows' : ' across the columns';
    notes.push(['Axis ' + k + ': ' + (!b.ok[k] ? p + ' and ' + q + ' are different and neither is 1. They clash.'
      : p === q ? p + ' = ' + q + ', so the sizes already match.'
      : !b.shape ? p + ' and ' + q + ' would fit, because one of them is 1.'
        : who + (k < nd - own.length ? ' has no axis here, which counts as 1' : ' has size 1') + '. It is repeated ' + max(p, q) +
          ' times' + way + '.'), b.ok[k] ? 'black' : 'firebrick']);
  }
  notes.push(b.shape ? ['Result shape ' + tupleText(b.shape) + '. NumPy does not really copy the smaller array. It reuses its values.', 'darkgreen']
    : ['ValueError: ' + b.error, 'firebrick']);
  const nx = narrow ? x + 10 : tx + labelW + 2 * colW + 18, nw = x + w - 12 - nx, lh = ts + 4;
  let ny = narrow ? ty + 4 * rowH + 5 : y + 36;
  notes.forEach(([str, ink], k) => {
    const lines = k === notes.length - 1 ? 3 : 2;
    noStroke();
    fill(ink);
    textStyle(k === notes.length - 1 ? BOLD : NORMAL);
    textSize(ts);
    textLeading(lh);
    textAlign(LEFT, TOP);
    text(str, nx, ny, nw, lines * lh + 3);
    ny += lines * lh + (narrow ? 2 : 8);
  });
  textStyle(NORMAL);
}

// Clicking a cell in any grid frames that position in all three.
function mouseOver(b) {
  return mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h;
}
function mousePressed() {
  const box = cellBoxes.find(mouseOver);
  if (box) sel = [box.i, box.j];
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
