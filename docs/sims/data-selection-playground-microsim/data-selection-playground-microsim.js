// Data Selection Playground
// CANVAS_HEIGHT: 595
// Bloom L3 (Apply): students build the four kinds of pandas selection from the chapter and see
// at once which cells each one takes: column selection df[[...]], rows by position
// df.iloc[start:stop], rows by label df.loc[[...]], and a Boolean filter df[condition].
//
// Model: a small pandas-like table. The index holds the student names, so loc (labels) and
// iloc (positions) visibly differ. The rules are the ones pandas uses:
//   df[[cols]]          keeps the listed columns, in the order listed
//   df.iloc[a:b]        keeps positions a, a+1, ..., b-1 (the stop position is excluded)
//   df.loc[[labels]]    keeps the rows with those index labels, in the order listed
//   df[df[col] op v]    tests every row and keeps the rows where the test is True
// The code line, the result, and its shape are all computed from DATA and the controls.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 480;
let controlHeight = 115;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;
let labelWidth = 62;        // width of the label span in each slider row
let valueWidth = 38;        // width of the value span in each slider row

// The chapter's students and five more. The names are the index.
const COLUMNS = ['age', 'city', 'score', 'active'];
const NAMES = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'];
const DATA = [
  [25, 'New York', 85, true], [30, 'Los Angeles', 92, false], [22, 'Chicago', 78, true],
  [28, 'Houston', 95, true], [26, 'Boston', 88, false], [19, 'New York', 64, true],
  [31, 'Chicago', 91, true], [24, 'New York', 58, false], [27, 'Boston', 73, true],
  [21, 'Los Angeles', 69, false]
];
const N = DATA.length, M = COLUMNS.length;
const CELLS = DATA.map(row => row.map(v => (typeof v === 'boolean' ? (v ? 'True' : 'False') : String(v))));
const RANGES = { age: [18, 32], score: [50, 100] };       // slider range for each filter column
const OPS = ['>', '>=', '<', '<=', '==', '!='];
const quote = s => '"' + s + '"';

const NOTES = {
  columns: ['Double brackets return a DataFrame with just the listed columns, in the order you list them.',
    'Try: show only city and score.'],
  iloc: ['iloc counts positions from 0, and the stop position is not included.', 'Try: get the last three rows.'],
  loc: ['loc looks rows up by index label. The rows come back in the order you list them.',
    'Try: Diana first, then Bob.'],
  filter: ['The condition gives True or False for every row. Only the True rows are kept.',
    'Try: find everyone who passed (score >= 70).']
};

// selection state, one set of choices per method
let mode = 'columns';
let pickedCols = ['age', 'score'];          // in the order they were clicked
let pickedRows = ['Alice', 'Charlie'];
let ilocStart = 0, ilocStop = 3;
let filterCol = 'score', filterOp = '>=';
const filterValue = { age: 25, score: 90 };

let methodSelect, columnSelect, opSelect, startRow, stopRow;
let hitBoxes = [];            // clickable labels in the table: { x, y, w, h, kind, id }

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  methodSelect = createSelect();
  methodSelect.parent(mainElement);
  methodSelect.position(92, drawHeight + 8);
  methodSelect.option('Columns: df[[...]]', 'columns');
  methodSelect.option('Rows by position: df.iloc[a:b]', 'iloc');
  methodSelect.option('Rows by label: df.loc[[...]]', 'loc');
  methodSelect.option('Boolean filter: df[condition]', 'filter');
  methodSelect.style('font-size', '15px');
  methodSelect.changed(() => { mode = methodSelect.value(); configureControls(); });

  columnSelect = createSelect();
  columnSelect.parent(mainElement);
  columnSelect.position(92, drawHeight + 43);
  Object.keys(RANGES).forEach(name => columnSelect.option(name));
  columnSelect.selected(filterCol);
  columnSelect.style('font-size', '15px');
  columnSelect.changed(() => { filterCol = columnSelect.value(); configureControls(); });

  opSelect = createSelect();
  opSelect.parent(mainElement);
  opSelect.position(170, drawHeight + 43);
  OPS.forEach(op => opSelect.option(op));
  opSelect.selected(filterOp);
  opSelect.style('font-size', '15px');
  opSelect.changed(() => { filterOp = opSelect.value(); });

  startRow = makeSliderRow('start', 0, N, ilocStart, 1, 1);
  stopRow = makeSliderRow('stop', 0, N, ilocStop, 1, 2);
  resizeSliders();
  configureControls();

  describe('A practice area for selecting data from a pandas DataFrame of ten students. A menu chooses column ' +
    'selection, rows by position with iloc, rows by label with loc, or a Boolean filter. The selected cells are ' +
    'highlighted in the table, and a second panel shows the pandas code, the resulting table, and its shape.', LABEL);
}

// A control row built from a div: the label and value sit in fixed-width spans so that
// every slider in the control area starts at the same x position.
function makeSliderRow(label, minValue, maxValue, startValue, step, rowIndex) {
  const row = createDiv();
  row.parent(document.querySelector('main'));
  row.position(10, drawHeight + 8 + rowIndex * 35);
  row.style('font-size', '16px');
  const labelSpan = createSpan(label + ':');
  labelSpan.parent(row);
  labelSpan.style('display', 'inline-block');
  labelSpan.style('width', labelWidth + 'px');
  const valueSpan = createSpan('');
  valueSpan.parent(row);
  valueSpan.style('display', 'inline-block');
  valueSpan.style('width', valueWidth + 'px');
  valueSpan.style('font-weight', 'bold');
  const slider = createSlider(minValue, maxValue, startValue, step);
  slider.parent(row);
  slider.style('vertical-align', 'middle');
  return { row, labelSpan, valueSpan, slider };
}

function resizeSliders() {
  const w = max(60, canvasWidth - labelWidth - valueWidth - 40);
  startRow.slider.size(w);
  stopRow.slider.size(w);
}

// Show only the controls the chosen method uses. The second slider is the stop position
// for iloc and the comparison value for the Boolean filter.
function configureControls() {
  const isIloc = mode === 'iloc', isFilter = mode === 'filter';
  if (isIloc) startRow.row.show(); else startRow.row.hide();
  if (isIloc || isFilter) stopRow.row.show(); else stopRow.row.hide();
  if (isFilter) { columnSelect.show(); opSelect.show(); } else { columnSelect.hide(); opSelect.hide(); }
  const range = isFilter ? RANGES[filterCol] : [0, N];
  stopRow.labelSpan.html(isFilter ? 'value:' : 'stop:');
  stopRow.slider.elt.min = range[0];
  stopRow.slider.elt.max = range[1];
  stopRow.slider.value(isFilter ? filterValue[filterCol] : ilocStop);
  startRow.slider.value(ilocStart);
}

function compare(a, op, b) {
  switch (op) {
    case '>': return a > b;
    case '>=': return a >= b;
    case '<': return a < b;
    case '<=': return a <= b;
    case '==': return a === b;
    default: return a !== b;
  }
}

// Apply the chosen method: which rows and columns (by position) it returns, in result order
function currentSelection() {
  const range = (a, b) => Array.from({ length: max(0, b - a) }, (v, i) => a + i);
  let rows = range(0, N), cols = range(0, M), code, mask = null, empty = null;
  if (mode === 'columns') {
    cols = pickedCols.map(name => COLUMNS.indexOf(name));
    code = 'df[[' + pickedCols.map(quote).join(', ') + ']]';
    if (cols.length === 0) empty = 'No columns are listed, so the result has ' + N + ' rows and 0 columns. Click a column name in the table.';
  } else if (mode === 'iloc') {
    rows = range(ilocStart, ilocStop);
    code = 'df.iloc[' + ilocStart + ':' + ilocStop + ']';
    if (rows.length === 0) empty = 'Empty DataFrame: start (' + ilocStart + ') is not smaller than stop (' + ilocStop + '), so no positions are selected.';
  } else if (mode === 'loc') {
    rows = pickedRows.map(name => NAMES.indexOf(name));
    code = 'df.loc[[' + pickedRows.map(quote).join(', ') + ']]';
    if (rows.length === 0) empty = 'Empty DataFrame: no labels are listed. Click a name in the table.';
  } else {
    const c = COLUMNS.indexOf(filterCol), v = filterValue[filterCol];
    mask = DATA.map(row => compare(row[c], filterOp, v));
    rows = rows.filter(r => mask[r]);
    code = 'df[df[' + quote(filterCol) + '] ' + filterOp + ' ' + v + ']';
    if (rows.length === 0) empty = 'Empty DataFrame: the condition is False for every row.';
  }
  return { rows, cols, code, mask, empty, rowSet: new Set(rows), colSet: new Set(cols) };
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  // read the sliders into the state of the method that owns them
  if (mode === 'iloc') {
    ilocStart = startRow.slider.value();
    ilocStop = stopRow.slider.value();
    startRow.valueSpan.html(ilocStart);
    stopRow.valueSpan.html(ilocStop);
  } else if (mode === 'filter') {
    filterValue[filterCol] = stopRow.slider.value();
    stopRow.valueSpan.html(filterValue[filterCol]);
  }

  const narrow = canvasWidth < 600;
  const sel = currentSelection();
  const over = hitBoxes.some(b => mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h);
  cursor(over ? HAND : ARROW);
  hitBoxes = [];

  noStroke();
  fill('black');
  textStyle(NORMAL);
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Data Selection Playground', canvasWidth / 2, 8);

  const top = narrow ? 34 : 44, bottom = drawHeight - 8;
  if (narrow) {
    const srcH = 202;
    drawSource(margin, top, canvasWidth - 2 * margin, srcH, sel, narrow);
    drawResult(margin, top + srcH + 5, canvasWidth - 2 * margin, bottom - top - srcH - 5, sel, narrow);
  } else {
    const leftW = (canvasWidth - 2 * margin - 10) * 0.52;
    drawSource(margin, top, leftW, bottom - top, sel, narrow);
    drawResult(margin + leftW + 10, top, canvasWidth - 2 * margin - leftW - 10, bottom - top, sel, narrow);
  }

  // control labels and, for the two click-to-choose methods, a hint in place of sliders
  noStroke();
  fill('black');
  textStyle(NORMAL);
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Select by:', 10, drawHeight + 21);
  if (mode === 'filter') text('Test:', 10, drawHeight + 56);
  if (mode === 'columns' || mode === 'loc') {
    fill('dimgray');
    textSize(narrow ? 14 : 16);
    textAlign(LEFT, TOP);
    textWrap(WORD);
    textLeading(22);
    const what = mode === 'columns' ? 'a column name (outlined in blue)' : 'a row label (a name, outlined in blue)';
    text('Click ' + what + ' in the df table to add it to the list or remove it.', 10, drawHeight + 46, canvasWidth - 20, 60);
  }
}

// The full DataFrame. Gold cells are the ones the selection returns.
function drawSource(x, y, w, h, sel, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const pad = narrow ? 8 : 12, titleH = narrow ? 20 : 30;

  noStroke();
  fill('black');
  textStyle(BOLD);
  textSize(narrow ? 13 : 16);
  textAlign(LEFT, CENTER);
  text('df', x + pad, y + titleH / 2 + 2);
  const dfW = textWidth('df');
  textStyle(NORMAL);
  fill('dimgray');
  textSize(narrow ? 12 : 14);
  // legend on the right, and as much of the description as fits beside it
  const sw = textWidth('selected');
  const room = w - 2 * pad - dfW - sw - 40;
  const about = [N + ' rows × ' + M + ' columns, index = names', N + ' rows × ' + M + ' columns', ''].find(s => textWidth(s) <= room);
  text(about, x + pad + dfW + 8, y + titleH / 2 + 2);
  textAlign(RIGHT, CENTER);
  text('selected', x + w - pad, y + titleH / 2 + 2);
  fill('gold');
  stroke('goldenrod');
  strokeWeight(1);
  rect(x + w - pad - sw - 18, y + titleH / 2 - 5, 13, 13, 2);

  const gold = alpha => { const c = color('gold'); c.setAlpha(alpha); return c; };
  const inResult = (r, c) => sel.rowSet.has(r) && sel.colSet.has(c);
  const usesPositions = mode === 'iloc';
  const test = filterOp + ' ' + filterValue[filterCol];      // shown beside the outlined column
  const rowH = min((h - titleH - 8) / (N + 1), 34);
  const out = drawSheet(x + pad, y + titleH, w - 2 * pad, rowH, narrow ? 12 : 15, NAMES,
    COLUMNS.map((name, c) => ({ head: name, cells: CELLS.map(row => row[c]) })), {
      fit: true,
      // positions on the left, and on the right the True/False result of the filter test
      left: { head: '#', headInk: 'darkgray', cells: NAMES.map((name, r) => String(r)),
        ink: r => (usesPositions && sel.rowSet.has(r) ? 'royalblue' : 'darkgray'),
        bold: r => usesPositions && sel.rowSet.has(r) },
      right: sel.mask ? { head: test, widest: '>= 100', headInk: 'royalblue',
        cells: sel.mask.map(v => (v ? 'True' : 'False')),
        ink: r => (sel.mask[r] ? 'seagreen' : 'darkgray'), bold: r => sel.mask[r] } : null,
      headFill: c => (sel.colSet.has(c) && sel.rows.length > 0 ? gold(150) : null),
      cellFill: (r, c) => (inResult(r, c) ? gold(110) : null),
      indexFill: r => (sel.rowSet.has(r) ? gold(150) : null),
      ink: (r, c) => (inResult(r, c) ? 'black' : 'gray'),
      headHit: mode === 'columns',
      indexHit: mode === 'loc'
    });

  // the Boolean filter tests one column: outline it
  if (sel.mask) {
    const k = out.first + 1 + COLUMNS.indexOf(filterCol);
    noFill();
    stroke('royalblue');
    strokeWeight(2);
    rect(out.lefts[k], y + titleH, out.widths[k], rowH * (N + 1));
  }
}

// A table drawn like a spreadsheet: gray label cells, grid lines, bold index and column names,
// values right-aligned as pandas prints them. The text shrinks, down to 11px, until it fits.
// opt.fit stretches the table to the full width. opt.left and opt.right are extra columns
// outside the grid. opt.headHit and opt.indexHit make the labels clickable.
function drawSheet(x, y, w, rowH, ts, index, cols, opt) {
  const rows = index.length;
  const all = (opt.left ? [opt.left] : []).concat([{ head: '', cells: index }], cols, opt.right ? [opt.right] : []);
  let natural, pad;
  for (; ; ts--) {
    textSize(ts);
    textStyle(BOLD);
    natural = all.map(col => max([textWidth(col.widest || col.head)].concat(col.cells.map(s => textWidth(s)))));
    pad = (w - natural.reduce((a, b) => a + b, 0)) / all.length;
    if (pad >= 6 || ts <= 11) break;
  }
  pad = opt.fit ? max(pad, 2) : constrain(pad, 2, 22);
  const widths = natural.map(v => v + pad);
  const lefts = [];
  widths.reduce((acc, v, i) => { lefts[i] = acc; return acc + v; }, x);
  const first = opt.left ? 1 : 0;                       // the index column
  const gx = lefts[first], gh = rowH * (rows + 1);
  const gw = widths.slice(first, first + 1 + cols.length).reduce((a, b) => a + b, 0);

  // label cells, then highlight tints
  noStroke();
  fill('white');
  rect(gx, y, gw, gh);
  fill('gainsboro');
  rect(gx, y, gw, rowH);
  rect(gx, y, widths[first], gh);
  for (let c = 0; c < cols.length; c++) {
    const k = first + 1 + c;
    const hf = opt.headFill ? opt.headFill(c) : null;
    if (hf) { fill(hf); rect(lefts[k], y, widths[k], rowH); }
    for (let r = 0; r < rows; r++) {
      const cf = opt.cellFill ? opt.cellFill(r, c) : null;
      if (cf) { fill(cf); rect(lefts[k], y + (r + 1) * rowH, widths[k], rowH); }
    }
  }
  for (let r = 0; r < rows; r++) {
    const xf = opt.indexFill ? opt.indexFill(r) : null;
    if (xf) { fill(xf); rect(gx, y + (r + 1) * rowH, widths[first], rowH); }
  }

  stroke('silver');
  strokeWeight(1);
  noFill();
  rect(gx, y, gw, gh);
  for (let r = 1; r <= rows; r++) line(gx, y + r * rowH, gx + gw, y + r * rowH);
  for (let c = 0; c < cols.length; c++) line(lefts[first + 1 + c], y, lefts[first + 1 + c], y + gh);

  // clickable labels get a blue outline, and their boxes are remembered for mousePressed
  const clickable = [];
  if (opt.headHit) cols.forEach((col, c) => clickable.push({ x: lefts[first + 1 + c], y, w: widths[first + 1 + c], h: rowH, kind: 'col', id: col.head }));
  if (opt.indexHit) index.forEach((name, r) => clickable.push({ x: gx, y: y + (r + 1) * rowH, w: widths[first], h: rowH, kind: 'row', id: name }));
  stroke('royalblue');
  strokeWeight(1.5);
  for (const b of clickable) {
    rect(b.x + 2, b.y + 2, b.w - 4, b.h - 4, 4);
    hitBoxes.push(b);
  }

  noStroke();
  textAlign(RIGHT, CENTER);
  all.forEach((col, k) => {
    const tx = lefts[k] + widths[k] - min(pad * 0.45, 10);
    const outside = k < first || k > first + cols.length;
    fill(col.headInk || 'black');
    textStyle(BOLD);
    text(col.head, tx, y + rowH / 2 + 1);
    for (let r = 0; r < rows; r++) {
      const bold = k === first || (outside && col.bold(r));
      fill(outside ? col.ink(r) : (k === first || !opt.ink ? 'black' : opt.ink(r, k - first - 1)));
      textStyle(bold ? BOLD : NORMAL);
      text(col.cells[r], tx, y + (r + 1.5) * rowH + 1);
    }
  });
  textStyle(NORMAL);
  return { bottom: y + gh, lefts, widths, first };
}

// The code for the current selection, the DataFrame it returns, and a note.
function drawResult(x, y, w, h, sel, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = narrow ? 12 : 15, lh = ts + 6, pad = narrow ? 8 : 12;
  const ix = x + pad, iw = w - 2 * pad;

  // code line, wrapped if the list is long
  textSize(ts + 1);
  textStyle(NORMAL);
  textWrap(WORD);
  const codeLines = countLines(sel.code, iw - 16);
  const chipH = codeLines * (lh + 1) + (narrow ? 8 : 12);
  noStroke();
  fill('darkslategray');
  rect(ix, y + pad, iw, chipH, 6);
  fill('white');
  textAlign(LEFT, TOP);
  textLeading(lh + 1);
  text(sel.code, ix + 8, y + pad + (narrow ? 5 : 7), iw - 16, chipH);
  let cy = y + pad + chipH + (narrow ? 5 : 9);

  // result heading with the shape
  fill('black');
  textSize(ts);
  textStyle(BOLD);
  text('Result', ix, cy);
  const headW = textWidth('Result');
  fill('royalblue');
  const shape = 'shape (' + sel.rows.length + ', ' + sel.cols.length + ')';
  text(shape, ix + headW + 10, cy);
  const shapeW = textWidth(shape);
  textStyle(NORMAL);
  const counts = sel.rows.length + ' of ' + N + ' rows, ' + sel.cols.length + ' of ' + M + ' columns';
  textSize(ts - 1);
  if (headW + shapeW + textWidth(counts) + 26 <= iw) {         // spelled out, where there is room
    fill('dimgray');
    textAlign(RIGHT, TOP);
    text(counts, ix + iw, cy + 1);
  }
  cy += lh + 1;

  // note, anchored to the bottom: a warning when the result is empty
  const note = sel.empty || (NOTES[mode][0] + (narrow ? '' : ' ' + NOTES[mode][1]));
  textSize(ts);
  const noteH = countLines(note, iw - 16) * lh + (narrow ? 8 : 12);
  const ny = y + h - noteH - pad;
  fill(sel.empty ? 'mistyrose' : 'lightyellow');
  stroke(sel.empty ? 'indianred' : 'khaki');
  strokeWeight(1);
  rect(ix, ny, iw, noteH, 8);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textLeading(lh);
  text(note, ix + 8, ny + (narrow ? 4 : 6), iw - 16, noteH);

  // the returned DataFrame. A narrow panel shows the first rows only, as pandas does for long output.
  const maxRows = narrow ? constrain(floor((ny - cy - 4 - lh) / 15) - 1, 2, 5) : N;
  const shown = sel.rows.slice(0, maxRows), more = sel.rows.length - shown.length;
  const avail = ny - cy - 6 - (more > 0 ? lh - 2 : 0);
  const rowH = min(narrow ? 16 : 26, avail / (shown.length + 1));
  const out = drawSheet(ix, cy, iw, rowH, ts, shown.map(r => NAMES[r]),
    sel.cols.map(c => ({ head: COLUMNS[c], cells: shown.map(r => CELLS[r][c]) })), {});
  if (more > 0) {
    noStroke();
    fill('dimgray');
    textSize(ts);
    textStyle(ITALIC);
    textAlign(LEFT, TOP);
    text('... and ' + more + ' more ' + (more === 1 ? 'row' : 'rows'), ix + 4, out.bottom + 3);
    textStyle(NORMAL);
  }
}

// How many lines p5's word wrapping needs for str at the current text size. Used only to
// measure heights; the text itself is drawn with textWrap(WORD).
function countLines(str, w) {
  let lines = 1, line = '';
  for (const word of str.split(' ')) {
    const test = line + word + ' ';       // p5 measures each line with its trailing space
    if (textWidth(test) > w && line.length > 0) {
      lines++;
      line = word + ' ';
    } else {
      line = test;
    }
  }
  return lines;
}

// Clicking a column name or a row label adds it to that method's list, or removes it.
function mousePressed() {
  for (const b of hitBoxes) {
    if (mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h) {
      const list = b.kind === 'col' ? pickedCols : pickedRows;
      const at = list.indexOf(b.id);
      if (at >= 0) list.splice(at, 1); else list.push(b.id);
      return;
    }
  }
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  resizeSliders();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
