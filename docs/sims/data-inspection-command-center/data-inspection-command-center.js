// Data Inspection Command Center
// CANVAS_HEIGHT: 580
// Bloom L3 (Apply): students choose a pandas inspection command (head, tail, shape, info,
// describe, columns, dtypes), set n for head and tail, and read the result exactly as a
// notebook would print it. The DataFrame is highlighted to show what the command looked at.
//
// Model: every output is computed from DATA below the way pandas 2.x computes it.
//   dtype      text: object. True/False: bool. Numbers: int64, or float64 when a value is
//              missing or has a decimal part (NaN is a float).
//   describe() count of non-missing values, mean, sample standard deviation (divide by
//              n - 1), min, max, and quartiles by linear interpolation between sorted values.
//   info()     memory = 132 bytes for the RangeIndex + rows x (8 bytes per object, int64, or
//              float64 column and 1 byte per bool column).

let containerWidth;
let canvasWidth = 400;
let drawHeight = 500;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let sliderLeftMargin = 120;
let defaultTextSize = 16;

// The chapter's five students plus three more. One age and one score are missing.
const COLUMNS = ['name', 'age', 'city', 'score', 'active'];
const DATA = [
  ['Alice', 25, 'New York', 85, true],
  ['Bob', 30, 'Los Angeles', 92, false],
  ['Charlie', 22, 'Chicago', 78, true],
  ['Diana', 28, 'Houston', 95, true],
  ['Eve', 26, 'Boston', 88, false],
  ['Frank', null, 'Seattle', 71.5, true],
  ['Grace', 31, 'Denver', null, true],
  ['Henry', 24, 'Miami', 90.5, false]
];
const N = DATA.length, M = COLUMNS.length;
const columnValues = c => DATA.map(row => row[c]);

function dtypeOf(values) {
  const present = values.filter(v => v !== null);
  if (present.every(v => typeof v === 'boolean')) return 'bool';
  if (present.every(v => typeof v === 'number')) {
    return present.length < values.length || present.some(v => !Number.isInteger(v)) ? 'float64' : 'int64';
  }
  return 'object';
}
const DTYPES = COLUMNS.map((name, c) => dtypeOf(columnValues(c)));
const NON_NULL = COLUMNS.map((name, c) => columnValues(c).filter(v => v !== null).length);
const NUMERIC = COLUMNS.map((name, c) => c).filter(c => DTYPES[c] === 'int64' || DTYPES[c] === 'float64');
const TYPE_COLORS = { object: 'slateblue', int64: 'seagreen', float64: 'chocolate', bool: 'teal' };

// How pandas prints one value: floats of a column share a number of decimals, missing is NaN
function show(v, c) {
  if (v === null) return 'NaN';
  if (DTYPES[c] === 'float64') {
    const decimals = Math.max(1, ...columnValues(c).filter(x => x !== null).map(x => (String(x).split('.')[1] || '').length));
    return v.toFixed(decimals);
  }
  if (DTYPES[c] === 'bool') return v ? 'True' : 'False';
  return String(v);
}
const CELLS = DATA.map(row => row.map((v, c) => show(v, c)));
const INDEX = DATA.map((row, r) => String(r));

// describe(): the eight summary statistics of one numeric column
const STAT_NAMES = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'];
function describeColumn(values) {
  const v = values.filter(x => x !== null).sort((a, b) => a - b);
  const n = v.length;
  const mean = v.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (n - 1));
  const quantile = p => {
    const pos = (n - 1) * p, lo = Math.floor(pos);
    return lo + 1 < n ? v[lo] + (pos - lo) * (v[lo + 1] - v[lo]) : v[lo];
  };
  return [n, mean, sd, v[0], quantile(0.25), quantile(0.5), quantile(0.75), v[n - 1]];
}

const METHODS = [
  { id: 'head', label: 'df.head(n)', usesN: true },
  { id: 'tail', label: 'df.tail(n)', usesN: true },
  { id: 'shape', label: 'df.shape' },
  { id: 'info', label: 'df.info()' },
  { id: 'describe', label: 'df.describe()' },
  { id: 'columns', label: 'df.columns' },
  { id: 'dtypes', label: 'df.dtypes' }
];

let methodSelect, nSlider;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  methodSelect = createSelect();
  methodSelect.parent(mainElement);
  methodSelect.position(sliderLeftMargin, drawHeight + 10);
  METHODS.forEach(m => methodSelect.option(m.label, m.id));
  methodSelect.style('font-size', '15px');
  methodSelect.changed(updateSliderState);

  nSlider = createSlider(1, N, 5, 1);
  nSlider.parent(mainElement);
  nSlider.position(sliderLeftMargin, drawHeight + 44);
  nSlider.size(canvasWidth - sliderLeftMargin - margin);
  updateSliderState();

  describe('A notebook-style practice panel. A table shows a DataFrame of eight students with five columns and ' +
    'two missing values. A menu chooses one of seven pandas inspection commands and a slider sets n for head and ' +
    'tail. The command and its output are shown as a notebook cell, the part of the table the command used is ' +
    'highlighted, and a note explains how to read the output.', LABEL);
}

// n only matters for head and tail, so the slider is switched off for the other commands
function updateSliderState() {
  const m = METHODS.find(mm => mm.id === methodSelect.value());
  if (m.usesN) nSlider.removeAttribute('disabled'); else nSlider.attribute('disabled', '');
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
  const m = METHODS.find(mm => mm.id === methodSelect.value());
  const n = nSlider.value();

  noStroke();
  fill('black');
  textStyle(NORMAL);
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('Data Inspection Command Center', canvasWidth / 2, 8);

  const top = narrow ? 36 : 44, bottom = drawHeight - 8;
  if (narrow) {
    const dfH = 170;
    drawDataFrame(margin, top, canvasWidth - 2 * margin, dfH, m, n, narrow);
    drawNotebook(margin, top + dfH + 5, canvasWidth - 2 * margin, bottom - top - dfH - 5, m, n, narrow);
  } else {
    const leftW = (canvasWidth - 2 * margin - 10) * 0.47;
    drawDataFrame(margin, top, leftW, bottom - top, m, n, narrow);
    drawNotebook(margin + leftW + 10, top, canvasWidth - 2 * margin - leftW - 10, bottom - top, m, n, narrow);
  }

  // control labels
  noStroke();
  textStyle(NORMAL);
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  fill('black');
  text('Command:', 10, drawHeight + 22);
  fill(m.usesN ? 'black' : 'gray');
  text(m.usesN ? 'Rows n = ' + n : 'n (not used)', 10, drawHeight + 56);
}

// What the chosen command looks at: which cells to tint, which to fade, and a caption.
function highlight(id, n) {
  const tint = (name, alpha) => { const c = color(name); c.setAlpha(alpha); return c; };
  const none = () => null, never = () => false;
  switch (id) {
    case 'head':
      return { caption: 'the first ' + n + ' of ' + N + ' rows', ink: 'darkgoldenrod',
        cell: r => (r < n ? tint('gold', 95) : null), head: none, fade: r => r >= n };
    case 'tail':
      return { caption: 'the last ' + n + ' of ' + N + ' rows', ink: 'darkgoldenrod',
        cell: r => (r >= N - n ? tint('gold', 95) : null), head: none, fade: r => r < N - n };
    case 'shape':
      return { caption: N + ' rows × ' + M + ' columns', ink: 'royalblue',
        cell: () => tint('royalblue', 40), head: none, fade: never };
    case 'info':
      return { caption: (N * M - NON_NULL.reduce((a, b) => a + b, 0)) + ' missing values (NaN)', ink: 'firebrick',
        cell: (r, c) => (DATA[r][c] === null ? tint('crimson', 85) : null), head: none, fade: never };
    case 'describe':
      return { caption: 'numeric columns: ' + NUMERIC.map(c => COLUMNS[c]).join(', '), ink: 'seagreen',
        cell: (r, c) => (NUMERIC.includes(c) ? tint('mediumseagreen', 70) : null),
        head: c => (NUMERIC.includes(c) ? tint('mediumseagreen', 130) : null), fade: (r, c) => !NUMERIC.includes(c) };
    case 'columns':
      return { caption: 'the ' + M + ' column names', ink: 'darkorchid',
        cell: none, head: () => tint('orchid', 130), fade: () => true };
    default:
      return { caption: 'one data type per column', ink: 'chocolate',
        cell: (r, c) => tint(TYPE_COLORS[DTYPES[c]], 45), head: c => tint(TYPE_COLORS[DTYPES[c]], 110), fade: never };
  }
}

// The DataFrame being inspected, with the cells the command used highlighted.
function drawDataFrame(x, y, w, h, m, n, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = narrow ? 12 : 16, pad = narrow ? 8 : 12;
  const titleH = narrow ? 20 : 32, capH = narrow ? 0 : 30;
  const hl = highlight(m.id, n);

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
  text(narrow ? 'highlighted:' : 'the DataFrame you are inspecting', x + pad + dfW + 8, y + titleH / 2 + 2);

  // caption: beside the title when narrow, under the table when wide
  fill(hl.ink);
  textStyle(BOLD);
  if (narrow) {
    text(hl.caption, x + pad + dfW + 8 + textWidth('highlighted: '), y + titleH / 2 + 2);
  } else {
    textSize(w < 330 ? 13 : 15);
    textAlign(CENTER, CENTER);
    text((w < 330 ? '' : 'Highlighted: ') + hl.caption, x + w / 2, y + h - capH / 2 - 4);
  }
  textStyle(NORMAL);

  const rowH = min((h - titleH - capH - 8) / (N + 1), 42);
  const cols = COLUMNS.map((name, c) => ({ head: name, cells: CELLS.map(row => row[c]) }));
  drawTable(x + pad, y + titleH, w - 2 * pad, rowH, ts, INDEX, cols, {
    sheet: true,
    cellFill: hl.cell,
    headFill: hl.head,
    ink: (r, c) => (hl.fade(r, c) ? 'darkgray' : (DATA[r][c] === null ? 'firebrick' : 'black'))
  });
}

// A table of strings with a bold index and bold column names, right-aligned like pandas.
// style.sheet: grid lines, gray label cells, stretched to the full width (the DataFrame view).
// Otherwise: notebook output with striped rows, as wide as its content needs.
function drawTable(x, y, w, rowH, ts, index, cols, style) {
  const rows = index.length;
  // column widths follow the widest entry. The text shrinks, down to 11px, until the table fits.
  let natural, pad;
  for (let size = ts; ; size--) {
    textSize(size);
    textStyle(BOLD);
    natural = [max(index.map(s => textWidth(s)))].concat(cols.map(col => max([textWidth(col.head)].concat(col.cells.map(s => textWidth(s))))));
    pad = (w - natural.reduce((a, b) => a + b, 0)) / natural.length;
    if (pad >= 6 || size <= 11) break;
  }
  pad = style.sheet ? max(pad, 2) : constrain(pad, 2, 26);
  const widths = natural.map(v => v + pad);
  const total = widths.reduce((a, b) => a + b, 0);
  const lefts = [];
  widths.reduce((acc, v, i) => { lefts[i] = acc; return acc + v; }, x);
  const inset = min(pad * 0.45, 10);

  noStroke();
  if (style.sheet) {
    fill('gainsboro');
    rect(x, y, total, rowH);
    rect(x, y, widths[0], rowH * (rows + 1));
  }
  for (let r = 0; r < rows; r++) {
    if (!style.sheet && r % 2 === 0) {
      fill('whitesmoke');
      rect(x, y + (r + 1) * rowH, total, rowH);
    }
  }
  for (let c = 0; c < cols.length; c++) {
    const hf = style.headFill ? style.headFill(c) : null;
    if (hf) { fill(hf); rect(lefts[c + 1], y, widths[c + 1], rowH); }
    for (let r = 0; r < rows; r++) {
      const cf = style.cellFill ? style.cellFill(r, c) : null;
      if (cf) { fill(cf); rect(lefts[c + 1], y + (r + 1) * rowH, widths[c + 1], rowH); }
    }
  }

  stroke(style.sheet ? 'silver' : 'black');
  strokeWeight(1);
  if (style.sheet) {
    noFill();
    rect(x, y, total, rowH * (rows + 1));
    for (let r = 1; r <= rows; r++) line(x, y + r * rowH, x + total, y + r * rowH);
    for (let c = 1; c < widths.length; c++) line(lefts[c], y, lefts[c], y + rowH * (rows + 1));
  } else {
    line(x, y + rowH, x + total, y + rowH);
  }

  noStroke();
  textAlign(RIGHT, CENTER);
  for (let c = 0; c < cols.length; c++) {
    const tx = lefts[c + 1] + widths[c + 1] - inset;
    fill('black');
    textStyle(BOLD);
    text(cols[c].head, tx, y + rowH / 2 + 1);
    textStyle(NORMAL);
    for (let r = 0; r < rows; r++) {
      fill(style.ink ? style.ink(r, c) : 'black');
      text(cols[c].cells[r], tx, y + (r + 1.5) * rowH + 1);
    }
  }
  textStyle(BOLD);
  fill('black');
  for (let r = 0; r < rows; r++) text(index[r], lefts[0] + widths[0] - inset, y + (r + 1.5) * rowH + 1);
  textStyle(NORMAL);
  return y + rowH * (rows + 1);
}

// The command as a notebook input cell, its output, and a note on how to read it.
function drawNotebook(x, y, w, h, m, n, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = narrow ? 12 : constrain(floor(w / 25), 12, 15);      // smaller text in a slim panel
  const promptW = ts < 15 ? 46 : 62;
  const cx = x + promptW, cw = w - promptW - (narrow ? 8 : 12);
  const cellH = ts + (narrow ? 12 : 16);
  let cy = y + (narrow ? 7 : 12);

  // input cell with light syntax coloring: numbers green, as in a notebook
  drawPrompt('In [1]:', 'navy', cx - 6, cy + cellH / 2 + 1, ts);
  fill('whitesmoke');
  stroke('lightgray');
  strokeWeight(1);
  rect(cx, cy, cw, cellH, 3);
  noStroke();
  textSize(ts + 1);
  textStyle(NORMAL);
  textAlign(LEFT, CENTER);
  let tx = cx + 8;
  for (const token of m.label.replace('(n)', '(' + n + ')').split(/(\d+)/)) {
    fill(/^\d+$/.test(token) ? 'green' : 'black');
    text(token, tx, cy + cellH / 2 + 1);
    tx += textWidth(token);
  }
  cy += cellH + (narrow ? 6 : 10);

  // output. info() prints its report, so it has no Out[ ] prompt.
  const rowH = narrow ? 16 : ts + 11, lh = narrow ? 15 : ts + 7;
  if (m.id !== 'info') drawPrompt('Out[1]:', 'firebrick', cx - 6, cy + (m.id === 'head' || m.id === 'tail' || m.id === 'describe' ? rowH : lh) / 2 + 1, ts);
  const ox = cx + 4, ow = cw - 4;
  let bottomY;
  noStroke();
  fill('black');
  textSize(ts);
  textStyle(NORMAL);
  if (m.id === 'head' || m.id === 'tail') {
    const start = m.id === 'head' ? 0 : N - n;
    const cols = COLUMNS.map((name, c) => ({ head: name, cells: CELLS.slice(start, start + n).map(row => row[c]) }));
    bottomY = drawTable(ox, cy, ow, rowH, ts, INDEX.slice(start, start + n), cols,
      { ink: (r, c) => (DATA[start + r][c] === null ? 'firebrick' : 'black') });
  } else if (m.id === 'describe') {
    const cols = NUMERIC.map(c => ({ head: COLUMNS[c], cells: describeColumn(columnValues(c)).map(v => v.toFixed(6)) }));
    bottomY = drawTable(ox, cy, ow, rowH, ts, STAT_NAMES, cols, {});
  } else if (m.id === 'shape') {
    bottomY = drawShape(ox, cy, ts, lh);
  } else if (m.id === 'info') {
    bottomY = drawInfo(ox, cy, ts, lh);
  } else if (m.id === 'columns') {
    const str = 'Index([' + COLUMNS.map(name => "'" + name + "'").join(', ') + "], dtype='object')";
    const lines = countLines(str, ow);
    textAlign(LEFT, TOP);
    textWrap(WORD);
    textLeading(lh);
    text(str, ox, cy + 2, ow, lines * lh + 4);
    bottomY = cy + lines * lh;
  } else {
    // dtypes: a Series with the column names as its index
    const nameW = max(COLUMNS.map(name => textWidth(name))) + 16;
    const typeW = max(DTYPES.map(t => textWidth(t)));
    for (let c = 0; c < M; c++) {
      fill('black');
      textAlign(LEFT, TOP);
      text(COLUMNS[c], ox, cy + 2 + c * lh);
      fill(TYPE_COLORS[DTYPES[c]]);
      textAlign(RIGHT, TOP);
      text(DTYPES[c], ox + nameW + typeW, cy + 2 + c * lh);
    }
    fill('black');
    textAlign(LEFT, TOP);
    text('dtype: object', ox, cy + 2 + M * lh);
    bottomY = cy + (M + 1) * lh;
  }

  // note on how to read the output
  const note = noteFor(m.id, n);
  const np = narrow ? 8 : 12;
  const nlh = narrow ? 16 : ts + 6;
  textSize(ts);
  textStyle(NORMAL);
  const noteH = countLines(note, w - 2 * np - 18) * nlh + (narrow ? 10 : 14);
  const ny = min(bottomY + (narrow ? 8 : 14), y + h - noteH - (narrow ? 6 : 10));
  fill('lightyellow');
  stroke('khaki');
  strokeWeight(1);
  rect(x + np, ny, w - 2 * np, noteH, 8);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textWrap(WORD);
  textLeading(nlh);
  text(note, x + np + 9, ny + (narrow ? 5 : 7), w - 2 * np - 18, noteH);
}

function drawPrompt(label, col, rightX, cy, ts) {
  noStroke();
  fill(col);
  textSize(ts - 1);
  textStyle(NORMAL);
  textAlign(RIGHT, CENTER);
  text(label, rightX, cy);
}

// (rows, columns), with each number labeled
function drawShape(x, y, ts, lh) {
  const parts = ['(', String(N), ', ', String(M), ')'];
  const labels = { 1: 'rows', 3: 'columns' };
  textSize(ts + 3);
  textStyle(NORMAL);
  textAlign(LEFT, TOP);
  let tx = x;
  const centers = {};
  for (let i = 0; i < parts.length; i++) {
    noStroke();
    fill('black');
    text(parts[i], tx, y + 1);
    if (labels[i]) centers[i] = tx + textWidth(parts[i]) / 2;
    tx += textWidth(parts[i]);
  }
  // the two labels sit under the tuple, spread apart so they do not collide
  textSize(ts - 1);
  const ly = y + ts + 22;
  const spots = { 1: x - 2, 3: x + textWidth('rows') + 16 };
  for (const i of [1, 3]) {
    const lw = textWidth(labels[i]);
    stroke('royalblue');
    strokeWeight(1.2);
    line(centers[i], y + ts + 8, spots[i] + lw / 2, ly - 2);
    noStroke();
    fill('royalblue');
    textAlign(LEFT, TOP);
    text(labels[i], spots[i], ly);
  }
  return ly + ts + 2;
}

// The printed report of df.info(). Columns with missing values stand out.
function drawInfo(x, y, ts, lh) {
  noStroke();
  fill('black');
  textSize(ts);
  textStyle(NORMAL);
  textAlign(LEFT, TOP);
  let cy = y + 2;
  for (const str of ["<class 'pandas.core.frame.DataFrame'>", 'RangeIndex: ' + N + ' entries, 0 to ' + (N - 1),
    'Data columns (total ' + M + ' columns):']) {
    text(str, x, cy);
    cy += lh;
  }
  const heads = ['#', 'Column', 'Non-Null Count', 'Dtype'];
  const rows = COLUMNS.map((name, c) => [String(c), name, NON_NULL[c] + ' non-null', DTYPES[c]]);
  const lefts = [x + 6];
  for (let k = 0; k < heads.length - 1; k++) {
    lefts.push(lefts[k] + max([textWidth(heads[k])].concat(rows.map(row => textWidth(row[k])))) + 16);
  }
  for (let k = 0; k < heads.length; k++) {
    noStroke();
    fill('black');
    text(heads[k], lefts[k], cy);
    stroke('gray');
    strokeWeight(1);
    line(lefts[k], cy + lh + 1, lefts[k] + textWidth(heads[k]), cy + lh + 1);     // the --- rule line
  }
  cy += lh + 6;
  noStroke();
  for (let c = 0; c < M; c++) {
    const missing = NON_NULL[c] < N;
    for (let k = 0; k < heads.length; k++) {
      fill(missing && k === 2 ? 'firebrick' : 'black');
      textStyle(missing && k === 2 ? BOLD : NORMAL);
      text(rows[c][k], lefts[k], cy);
    }
    cy += lh;
  }
  textStyle(NORMAL);
  fill('black');
  const counts = {};
  DTYPES.forEach(t => { counts[t] = (counts[t] || 0) + 1; });
  text('dtypes: ' + Object.keys(counts).sort().map(t => t + '(' + counts[t] + ')').join(', '), x, cy);
  cy += lh;
  const bytes = 132 + N * DTYPES.reduce((a, t) => a + (t === 'bool' ? 1 : 8), 0);
  text('memory usage: ' + bytes.toFixed(1) + '+ bytes', x, cy);
  return cy + lh;
}

function noteFor(id, n) {
  const withMissing = COLUMNS.filter((name, c) => NON_NULL[c] < N);
  switch (id) {
    case 'head':
      return n < N
        ? 'Notice: only shows ' + n + ' of the ' + N + ' rows! head() with no number shows the first 5.'
        : 'With n = ' + N + ' you see every row, because df has only ' + N + '. On a real dataset head() is a quick peek at the top.';
    case 'tail':
      return 'The last ' + n + (n === 1 ? ' row' : ' rows') + ', still carrying ' + (n === 1 ? 'its' : 'their') +
        ' original index (' + (n === 1 ? N - 1 : (N - n) + ' to ' + (N - 1)) + '). Use it to check that the whole file loaded.';
    case 'shape':
      return '(rows, columns): easy to remember! shape is an attribute, not a method, so it has no parentheses.';
    case 'info':
      return 'Look for non-null counts to find missing data. Here ' +
        withMissing.map(name => name + ' has ' + NON_NULL[COLUMNS.indexOf(name)] + ' of ' + N).join(' and ') + '.';
    case 'describe':
      return 'Only numeric columns are shown by default, so ' + COLUMNS.filter((name, c) => !NUMERIC.includes(c)).join(', ') +
        ' are left out. count is the number of non-missing values.';
    case 'columns':
      return 'The column names, in order. Like shape, columns is an attribute, so there are no parentheses.';
    default: {
      const forced = COLUMNS.filter((name, c) => DTYPES[c] === 'float64' && NON_NULL[c] < N &&
        columnValues(c).every(v => v === null || Number.isInteger(v)));
      return 'One data type per column. object means text.' +
        (forced.length ? ' ' + forced.join(', ') + ' holds whole numbers but is float64, because its missing value (NaN) is a float.' : '');
    }
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

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  nSlider.size(canvasWidth - sliderLeftMargin - margin);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
