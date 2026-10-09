// One-Hot Encoding Visualizer
// CANVAS_HEIGHT: 620
// Bloom L2-L3 (Understand, Apply): students read one categorical column before and after
// pd.get_dummies, switch drop_first and dtype, add a new category, and hover over a row to trace
// which new column receives its True.
//
// The encoding follows pandas (checked against pandas 2.3):
//   - the category labels are sorted and one column named prefix_label is made for each label
//   - each cell is the boolean (value == label); dtype=int stores the same answers as 1 and 0
//   - drop_first=True removes the column of the first label in sorted order
// With all k columns every row sums to 1, which duplicates the intercept column of a linear model
// (the dummy variable trap). With k - 1 columns the dropped label becomes the reference category
// and can still be recovered: first = 1 - (sum of the other columns).

let containerWidth;
let canvasWidth = 400;
let drawHeight = 540;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const SETS = [
  { col: 'color', values: ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green'], extra: 'Yellow' },
  { col: 'neighborhood', values: ['Downtown', 'Suburbs', 'Rural', 'Downtown', 'Rural', 'Suburbs'], extra: 'Uptown' },
  { col: 'fuel', values: ['Gas', 'Diesel', 'Electric', 'Hybrid', 'Gas', 'Electric'], extra: 'Hydrogen' }
];
const PALETTE = ['lightcoral', 'lightskyblue', 'lightgreen', 'khaki', 'plum'];   // by order of first appearance
const MAX_ROWS = 7;

let columnSelect, addButton, resetButton, dropBox, intBox;
let set = SETS[0], values = [];
let enc = {};                          // result of encode() for the rows on screen
let T = {};                            // table metrics for this frame: row height, header height, text size, index width
let locked = -1, sel = -1, hits = [];  // clicked row, row shown this frame, row rectangles

// What pd.get_dummies(df, columns=[col], drop_first=dropFirst) produces for one column
function encode(vals, col, dropFirst) {
  const levels = [...new Set(vals)].sort();              // pandas sorts the category labels
  const kept = dropFirst ? levels.slice(1) : levels;
  return { levels, kept, ref: dropFirst ? levels[0] : null, names: kept.map(v => col + '_' + v),
    cells: vals.map(v => kept.map(k => k === v)) };
}

function loadSet() {
  set = SETS.find(s => s.col === columnSelect.selected());
  values = set.values.slice();
  locked = -1;
  addButton.removeAttribute('disabled');
}

function addCategory() {
  if (values.length > set.values.length) return;
  values.push(set.extra);                                // a new row whose value has not been seen before
  locked = values.length - 1;
  addButton.attribute('disabled', '');
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  columnSelect = createSelect();
  columnSelect.parent(mainElement);
  columnSelect.position(75, drawHeight + 10);
  SETS.forEach(s => columnSelect.option(s.col));
  columnSelect.changed(loadSet);

  addButton = createButton('Add a Category');
  addButton.parent(mainElement);
  addButton.position(205, drawHeight + 10);
  addButton.mousePressed(addCategory);
  resetButton = createButton('Reset');
  resetButton.parent(mainElement);
  resetButton.position(325, drawHeight + 10);
  resetButton.mousePressed(loadSet);

  dropBox = createCheckbox(' drop_first=True', false);
  intBox = createCheckbox(' dtype=int', false);
  [dropBox, intBox].forEach((c, i) => {
    c.parent(mainElement);
    c.position(10 + 170 * i, drawHeight + 45);
    c.style('font-size', '16px');
  });
  loadSet();

  describe('A small DataFrame column of categories on the left and the columns that pandas get_dummies makes from it ' +
    'on the right, one true or false column per category. Controls choose the column, switch drop_first and integer ' +
    'output, and add a row with a new category. A note explains the dummy variable trap or the reference category, ' +
    'and hovering over a row traces it through both tables.', LABEL);
}

const colorOf = v => PALETTE[[...new Set(values)].indexOf(v)];
const cellText = on => intBox.checked() ? (on ? '1' : '0') : (on ? 'True' : 'False');

function rowAt(x, y) {
  for (const h of hits) if (x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h) return h.i;
  return -1;
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600, w = canvasWidth - 2 * margin;
  enc = encode(values, set.col, dropBox.checked());
  T = narrow ? { rh: 17, hh: 30, ts: 12, iw: 20 } : { rh: 28, hh: 40, ts: 15, iw: 28 };
  const blockH = 18 + T.hh + MAX_ROWS * T.rh;            // label, header, and rows of one table
  const hov = rowAt(mouseX, mouseY);
  sel = Math.min(hov >= 0 ? hov : locked, values.length - 1);
  cursor(hov >= 0 ? HAND : ARROW);
  hits = [];
  textWrap(WORD);

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 19 : 24);
  text('One-Hot Encoding Visualizer', canvasWidth / 2, 8);
  textAlign(LEFT, CENTER);
  textSize(narrow ? 14 : defaultTextSize);
  text('Column:', 10, drawHeight + 22);

  if (narrow) {
    drawCode(margin, 36, w, 46, 12);
    drawOriginal(margin, 88, 118);
    drawCallout(margin + 126, 88, w - 126, blockH);
    drawEncoded(margin, 94 + blockH, w, true);
    drawNote(margin, 100 + 2 * blockH, w, drawHeight - 108 - 2 * blockH);
  } else {
    const ow = 170, y0 = 98, y1 = y0 + blockH + 10, noteW = Math.round(w * 0.42);
    drawCode(margin, 44, w, 44, 15);
    drawOriginal(margin, y0, ow);
    drawEncoded(margin + ow + 46, y0, w - ow - 46, false);
    // arrow from the original column to the new columns, at the selected row when there is one
    const ay = y0 + 18 + T.hh + (sel >= 0 ? sel + 0.5 : values.length / 2) * T.rh, ax = margin + ow + 6;
    stroke('dimgray');
    strokeWeight(2.5);
    line(ax, ay, ax + 28, ay);
    noStroke();
    fill('dimgray');
    triangle(ax + 36, ay, ax + 26, ay - 6, ax + 26, ay + 6);
    drawNote(margin, y1, noteW, drawHeight - y1 - 8);
    drawCallout(margin + noteW + 10, y1, w - noteW - 10, drawHeight - y1 - 8);
  }
}

// The pandas call that the two checkboxes describe
function drawCode(x, y, w, h, ts) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 8);
  noStroke();
  fill('black');
  textSize(ts);
  textAlign(LEFT, CENTER);
  text("df_encoded = pd.get_dummies(df, columns=['" + set.col + "']" + (dropBox.checked() ? ', drop_first=True' : '') +
    (intBox.checked() ? ', dtype=int' : '') + ')', x + 10, y + 2, w - 20, h - 4);
}

function tableLabel(bold, rest, x, y) {
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textSize(T.ts);
  textStyle(BOLD);
  text(bold, x, y);
  const bw = textWidth(bold);
  textStyle(NORMAL);
  fill('dimgray');
  text(rest, x + bw, y);
}

function cell(x, y, w, h, bg, str, col, style) {
  fill(bg);
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h);
  noStroke();
  fill(col);
  textStyle(style || NORMAL);
  textAlign(CENTER, CENTER);
  text(str, x + w / 2, y + h / 2 + 1);
  textStyle(NORMAL);
}

function rowFrame(x, y, w) {              // outline of the selected row, and the hover rectangles of all rows
  values.forEach((v, i) => hits.push({ i, x, y: y + i * T.rh, w, h: T.rh }));
  if (sel < 0) return;
  noFill();
  stroke('black');
  strokeWeight(2.5);
  rect(x, y + sel * T.rh, w, T.rh);
}

function drawOriginal(x, y, w) {
  tableLabel('df', '  (original)', x, y);
  const top = y + 18, vw = w - T.iw;
  textSize(T.ts);
  cell(x, top, T.iw, T.hh, 'gainsboro', '', 'black');
  cell(x + T.iw, top, vw, T.hh, 'gainsboro', set.col, 'black', BOLD);
  values.forEach((v, i) => {
    const ry = top + T.hh + i * T.rh;
    cell(x, ry, T.iw, T.rh, 'white', i, 'dimgray');
    cell(x + T.iw, ry, vw, T.rh, colorOf(v), v, 'black');
  });
  rowFrame(x, top + T.hh, w);
}

function drawEncoded(x, y, w, narrow) {
  const k = enc.levels.length, sumW = narrow ? 50 : 66, cw = (w - T.iw - 8 - sumW) / k, top = y + 18, small = Math.max(11, T.ts - 2);
  tableLabel((narrow ? '↓ ' : '') + 'df_encoded', '  (' + enc.kept.length + ' new columns, dtype ' + (intBox.checked() ? 'int64' : 'bool') + ')', x, y);
  cell(x, top, T.iw, T.hh, 'gainsboro', '', 'black');
  enc.levels.forEach((lv, j) => {
    const cx = x + T.iw + j * cw, ghost = lv === enc.ref, tint = color(colorOf(lv));
    tint.setAlpha(110);
    // header: the column name prefix_label on two lines; a dropped column is drawn faded and crossed out
    cell(cx, top, cw, T.hh, ghost ? 'whitesmoke' : tint, '', 'black');
    textSize(small);
    fill(ghost ? 'firebrick' : 'dimgray');
    text(ghost ? 'dropped' : set.col + '_', cx + cw / 2, top + T.hh * 0.28);
    textSize(T.ts);
    textStyle(BOLD);
    fill(ghost ? 'darkgray' : 'black');
    text(lv, cx + cw / 2, top + T.hh * 0.7);
    if (ghost) {
      stroke('darkgray');
      strokeWeight(1.5);
      line(cx + cw / 2 - textWidth(lv) / 2 - 2, top + T.hh * 0.7, cx + cw / 2 + textWidth(lv) / 2 + 2, top + T.hh * 0.7);
    }
    textStyle(NORMAL);
    values.forEach((v, i) => {
      const on = v === lv, ry = top + T.hh + i * T.rh;
      if (ghost) cell(cx, ry, cw, T.rh, 'whitesmoke', cellText(on), 'silver');
      else cell(cx, ry, cw, T.rh, on ? colorOf(lv) : 'white', cellText(on), on ? 'black' : 'darkgray', on ? BOLD : NORMAL);
    });
  });
  values.forEach((v, i) => cell(x, top + T.hh + i * T.rh, T.iw, T.rh, 'white', i, 'dimgray'));

  // row sums of the kept columns: a check drawn beside the table, not a column of the DataFrame
  const sx = x + w - sumW, trap = !enc.ref;
  textSize(small);
  drawingContext.setLineDash([4, 3]);
  cell(sx, top, sumW, T.hh, 'white', 'row\nsum', trap ? 'firebrick' : 'darkgreen', ITALIC);
  textSize(T.ts);
  values.forEach((v, i) => cell(sx, top + T.hh + i * T.rh, sumW, T.rh, trap ? 'mistyrose' : 'honeydew',
    enc.kept.includes(v) ? 1 : 0, trap ? 'firebrick' : 'darkgreen', BOLD));
  drawingContext.setLineDash([]);
  rowFrame(x, top + T.hh, w - sumW - 8);
}

// How the first (alphabetical) column can be rebuilt from the others
function rebuildFormula() {
  return set.col + '_' + enc.levels[0] + ' = 1 − ' + enc.levels.slice(1).map(v => set.col + '_' + v).join(' − ');
}

function drawCallout(x, y, w, h) {
  const trap = !enc.ref, ts = T.ts, off = cellText(false);
  fill(trap ? 'mistyrose' : 'honeydew');
  stroke(trap ? 'indianred' : 'mediumseagreen');
  strokeWeight(1.5);
  rect(x, y, w, h, 10);
  noStroke();
  fill(trap ? 'firebrick' : 'darkgreen');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text(trap ? 'Warning: dummy variable trap' : 'Reference category: ' + enc.ref, x + 10, y + 8, w - 20, 40);
  textStyle(NORMAL);
  fill('black');
  textSize(ts);
  text(trap ? 'Every row sums to 1, the same as the column of ones a linear model uses for its intercept, so one column is ' +
    'redundant:\n' + rebuildFormula() + '\nThat is perfect multicollinearity. The coefficients are not uniquely determined.'
    : 'pandas drops the first label in sorted order. An all-' + off + ' row can only be ' + enc.ref + ':\n' + rebuildFormula() +
    '\nRow sums now vary, so the trap is gone. Each coefficient is a difference from ' + enc.ref + '.',
    x + 10, y + ts + 16, w - 20, h - ts - 14);
}

function drawNote(x, y, w, h) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = T.ts, k = enc.levels.length, on = cellText(true), off = cellText(false);
  let head, body;
  if (sel >= 0) {
    const v = values[sel];
    head = 'Row ' + sel + ':  ' + set.col + ' = \'' + v + '\'';
    body = v === enc.ref ? v + ' is the reference category. Its column was dropped, so this row is ' + off + ' in every new column.'
      : set.col + '_' + v + ' is ' + on + ' because this row\'s ' + set.col + ' is ' + v + '. Every other new column is ' + off + '.';
    if (sel >= set.values.length) body += ' This value had not appeared before, so it needed a column of its own.';
  } else if (enc.ref) {
    head = k + ' categories → k − 1 = ' + (k - 1) + ' new columns';
    body = 'The column ' + set.col + '_' + enc.ref + ' is left out (drawn faded). The ' + (k - 1) + ' columns that remain still tell ' +
      'every category apart. Hover over a row to trace it, or click to keep it selected.';
  } else {
    head = k + ' categories → ' + k + ' new columns';
    body = 'Each new column answers a yes or no question, such as "is ' + set.col + ' ' + enc.levels[0] + '?", so every row has exactly one ' +
      on + '. pandas sorts the labels, so the columns are alphabetical. Hover over a row to trace it.';
  }
  noStroke();
  fill('darkslateblue');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text(head, x + 10, y + 8);
  textStyle(NORMAL);
  fill('black');
  textSize(ts);
  text(body, x + 10, y + ts + 16, w - 20, h - ts - 20);
}

function mousePressed() {
  if (mouseY < 0 || mouseY > drawHeight || mouseX < 0 || mouseX > canvasWidth) return;
  const r = rowAt(mouseX, mouseY);
  locked = r === locked ? -1 : r;
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
