// DataFrame Anatomy
// CANVAS_HEIGHT: 505
// Bloom L1 (Remember): students identify and name the six parts of a pandas DataFrame: index,
// columns, row, column, cell, and values. Pointing at a part of the table (or at its callout)
// highlights it, and clicking keeps it selected. The panel names the part, says what it is,
// and, with Show code on, gives the pandas expression that returns it and the result.
//
// The table is the chapter's 4 x 3 example. Every result in the panel is built from ROWS,
// COLS, and VALUES below and laid out the way pandas 2.x prints it.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 460;
let controlHeight = 45;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const ROWS = ['row_0', 'row_1', 'row_2', 'row_3'];
const COLS = ['Col_A', 'Col_B', 'Col_C'];
const VALUES = [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]];
const R = 1, C = 1;           // the example row (row_1) and the example column (Col_B)

const quoted = list => '[' + list.map(s => "'" + s + "'").join(', ') + ']';

// The six parts, in callout order. `result` describes what the code returns.
const PARTS = [
  { name: 'Index', color: 'royalblue', brief: 'row labels',
    text: 'The row labels: a name tag for each row. Labels can be numbers, strings, or dates.',
    tip: 'Every DataFrame has an index. If you do not supply one, pandas numbers the rows 0, 1, 2, ...',
    code: ['df.index'],
    result: { kind: 'text', str: 'Index(' + quoted(ROWS) + ", dtype='object')" } },
  { name: 'Columns', color: 'seagreen', brief: 'column names',
    text: 'The column names, usually strings. Each one names a variable in the data.',
    tip: 'Index labels the rows and columns labels the columns. Together they let you ask for data by name.',
    code: ['df.columns'],
    result: { kind: 'text', str: 'Index(' + quoted(COLS) + ", dtype='object')" } },
  { name: 'Row', color: 'darkorange', brief: 'one observation',
    text: 'One observation (record): a horizontal slice with one value for every column.',
    tip: 'loc looks a row up by its label. iloc looks it up by its position, counting from 0.',
    code: ["df.loc['" + ROWS[R] + "']", 'df.iloc[' + R + ']'],
    result: { kind: 'series', labels: COLS, values: VALUES[R], footer: 'Name: ' + ROWS[R] + ', dtype: int64' } },
  { name: 'Column', color: 'darkorchid', brief: 'one variable',
    text: 'One variable (feature): a vertical slice with one value for every row.',
    tip: 'A single column is a Series. A DataFrame is a set of Series that share one index.',
    code: ["df['" + COLS[C] + "']"],
    result: { kind: 'series', labels: ROWS, values: VALUES.map(row => row[C]), footer: 'Name: ' + COLS[C] + ', dtype: int64' } },
  { name: 'Cell', color: 'crimson', brief: 'one value',
    text: 'A single value, found where one row and one column cross.',
    tip: 'Give the row label first and the column label second, the same order as (rows, columns).',
    code: ["df.loc['" + ROWS[R] + "', '" + COLS[C] + "']"],
    result: { kind: 'text', str: String(VALUES[R][C]) } },
  { name: 'Values', color: 'dimgray', brief: 'the data, no labels',
    text: 'All of the data without the row and column labels. Underneath, it is a NumPy array.',
    tip: ROWS.length + ' rows × ' + COLS.length + ' columns = ' + (ROWS.length * COLS.length) + ' values. The labels are not part of the array.',
    code: ['df.values'],
    result: { kind: 'array', rows: VALUES } }
];

let partSelect, codeCheckbox;
let pinned = null;            // index of the part kept selected by a click, or null
let hitBoxes = [];            // clickable regions, rebuilt every frame: { x, y, w, h, part }

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  partSelect = createSelect();
  partSelect.parent(mainElement);
  partSelect.position(86, drawHeight + 10);
  partSelect.option('All parts', '-1');
  PARTS.forEach((p, i) => partSelect.option((i + 1) + '  ' + p.name, String(i)));
  partSelect.style('font-size', '15px');
  partSelect.changed(() => {
    const v = int(partSelect.value());
    pinned = v < 0 ? null : v;
  });

  codeCheckbox = createCheckbox(' Show code', true);
  codeCheckbox.parent(mainElement);
  codeCheckbox.position(220, drawHeight + 12);
  codeCheckbox.style('font-size', '16px');

  describe('A labeled diagram of a pandas DataFrame with four rows and three columns. Six numbered callouts name ' +
    'its parts: index, columns, row, column, cell, and values. Pointing at or clicking a part highlights it, and a ' +
    'panel explains the part and shows the pandas code that returns it together with the result.', LABEL);
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
  const showCode = codeCheckbox.checked();

  // the part under the mouse wins over the pinned part
  const hovered = partAt(mouseX, mouseY);
  const active = hovered !== null ? hovered : pinned;
  cursor(hovered !== null ? HAND : ARROW);
  hitBoxes = [];

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textStyle(NORMAL);
  textSize(narrow ? 19 : 24);
  text('DataFrame Anatomy', canvasWidth / 2, 8);

  // layout: table on the left and panel on the right, or stacked when narrow
  const top = narrow ? 36 : 44;
  const regionW = narrow ? canvasWidth - 2 * margin : canvasWidth * 0.55 - margin;
  const g = tableGeometry(narrow, margin, regionW, top, drawHeight - top - 8);
  drawTable(g, active, narrow);
  drawCallouts(g, active, narrow);

  if (narrow) {
    const py = g.ty + g.h + 40;
    drawInfoPanel(margin, py, canvasWidth - 2 * margin, drawHeight - 8 - py, active, narrow, showCode);
  } else {
    const px = margin + regionW + 10;
    drawInfoPanel(px, top, canvasWidth - margin - px, drawHeight - top - 8, active, narrow, showCode);
  }

  // control label
  noStroke();
  fill('black');
  textStyle(NORMAL);
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Highlight:', 10, drawHeight + 22);
}

// Size and place the table so that it and its callouts fit the region.
function tableGeometry(narrow, regionX, regionW, regionY, regionH) {
  const calloutW = narrow ? 88 : 112;                 // room for the callouts right of the table
  const unit = constrain((regionW - calloutW - 6) / 4.15, 44, narrow ? 64 : 76);
  const idxW = unit * 1.15, cw = unit, ch = narrow ? 27 : 42;
  const w = idxW + COLS.length * cw, h = (ROWS.length + 1) * ch;
  const above = narrow ? 26 : 34, below = narrow ? 34 : 42;
  const tx = regionX + max(0, (regionW - w - calloutW) / 2);
  const ty = narrow ? regionY + above : regionY + (regionH - h - above - below) / 2 + above;
  return { tx, ty, idxW, cw, ch, w, h, vx: tx + idxW, vy: ty + ch };
}

// The rectangle each part covers in the table: [x, y, w, h]
function partRect(i, g) {
  const nR = ROWS.length, nC = COLS.length;
  return [
    [g.tx, g.vy, g.idxW, nR * g.ch],                          // index: the row labels
    [g.vx, g.ty, nC * g.cw, g.ch],                            // columns: the column names
    [g.tx, g.vy + R * g.ch, g.w, g.ch],                       // one row, label included
    [g.vx + C * g.cw, g.ty, g.cw, g.h],                       // one column, name included
    [g.vx + C * g.cw, g.vy + R * g.ch, g.cw, g.ch],           // one cell
    [g.vx, g.vy, nC * g.cw, nR * g.ch]                        // values: the data block
  ][i];
}

function drawTable(g, active, narrow) {
  const nR = ROWS.length, nC = COLS.length;

  // sheet, with gray label cells like a notebook table
  stroke('silver');
  strokeWeight(1);
  fill('white');
  rect(g.tx, g.ty, g.w, g.h);
  noStroke();
  fill('gainsboro');
  rect(g.tx, g.ty, g.w, g.ch);
  rect(g.tx, g.ty, g.idxW, g.h);

  // color tints: every part lightly in the overview, one part strongly when it is active
  const overviewAlpha = [60, 60, 55, 55, 150, 0];
  for (let i = 0; i < PARTS.length; i++) {
    if (active !== null && active !== i) continue;
    const [x, y, w, h] = partRect(i, g);
    const c = color(PARTS[i].color);
    c.setAlpha(active === i ? (i === 5 ? 60 : 95) : overviewAlpha[i]);
    noStroke();
    fill(c);
    rect(x, y, w, h);
  }

  // grid lines
  stroke('silver');
  strokeWeight(1);
  for (let r = 1; r <= nR; r++) line(g.tx, g.ty + r * g.ch, g.tx + g.w, g.ty + r * g.ch);
  for (let c = 0; c < nC; c++) line(g.vx + c * g.cw, g.ty, g.vx + c * g.cw, g.ty + g.h);

  // labels and numbers, a little smaller when the columns are tight
  noStroke();
  fill('black');
  textSize(constrain(floor(g.cw * 0.25), 12, 16));
  textAlign(CENTER, CENTER);
  textStyle(BOLD);
  for (let c = 0; c < nC; c++) text(COLS[c], g.vx + (c + 0.5) * g.cw, g.ty + g.ch / 2);
  for (let r = 0; r < nR; r++) text(ROWS[r], g.tx + g.idxW / 2, g.vy + (r + 0.5) * g.ch);
  textStyle(NORMAL);
  for (let r = 0; r < nR; r++) {
    for (let c = 0; c < nC; c++) text(VALUES[r][c], g.vx + (c + 0.5) * g.cw, g.vy + (r + 0.5) * g.ch);
  }

  // outlines: the active part, or the values block (dashed) and the example cell in the overview
  noFill();
  for (let i = 0; i < PARTS.length; i++) {
    const isActive = active === i;
    if (!isActive && !(active === null && i >= 4)) continue;
    const [x, y, w, h] = partRect(i, g);
    stroke(PARTS[i].color);
    strokeWeight(isActive ? 3 : 2);
    if (i === 5 && !isActive) drawingContext.setLineDash([5, 4]);
    rect(x + 1, y + 1, w - 2, h - 2, 3);
    drawingContext.setLineDash([]);
  }

  // hit regions for the table itself. Order matters: the first match wins, so the example
  // cell beats its row and column, and the label cells belong to Index and Columns.
  const cell = partRect(4, g), row = partRect(2, g), col = partRect(3, g);
  const order = [[4, cell], [0, partRect(0, g)], [1, partRect(1, g)],
    [2, [g.vx, row[1], nC * g.cw, row[3]]], [3, [col[0], g.vy, col[2], nR * g.ch]], [5, partRect(5, g)]];
  for (const [part, [x, y, w, h]] of order) hitBoxes.push({ x, y, w, h, part });
}

// Numbered callouts around the table with leader lines or brackets to the parts they name.
function drawCallouts(g, active, narrow) {
  const ts = narrow ? 13 : 16, bd = narrow ? 18 : 22;
  const rx = g.tx + g.w + (narrow ? 14 : 22);         // left edge of the callouts on the right
  const topY = g.ty - (narrow ? 14 : 19);
  const botY = g.ty + g.h + (narrow ? 21 : 27);
  const right = g.tx + g.w;
  const spots = [
    { side: 'below', x: g.tx + g.idxW / 2, y: botY, span: [g.tx, g.vx] },
    { side: 'right', y: g.ty + g.ch / 2, path: [[right, g.ty + g.ch / 2]] },
    { side: 'right', y: g.vy + (R + 0.5) * g.ch, path: [[right, g.vy + (R + 0.5) * g.ch]] },
    { side: 'above', x: g.vx + (C + 0.5) * g.cw, y: topY },
    // the cell's leader runs along the grid line above the row so it crosses no numbers
    { side: 'right', y: g.vy + (R - 0.5) * g.ch, path: [[g.vx + (C + 1) * g.cw, g.vy + R * g.ch], [right, g.vy + R * g.ch]] },
    { side: 'below', x: g.vx + COLS.length * g.cw / 2, y: botY, span: [g.vx, right] }
  ];

  textSize(ts);
  textStyle(BOLD);
  for (let i = 0; i < PARTS.length; i++) {
    const s = spots[i], p = PARTS[i];
    const c = color(p.color);
    if (active !== null && active !== i) c.setAlpha(70);
    const lw = bd + 5 + textWidth(p.name);
    const lx = s.side === 'right' ? rx : s.x - lw / 2;

    stroke(c);
    strokeWeight(active === i ? 2.5 : 1.5);
    noFill();
    if (s.side === 'right') {
      beginShape();
      for (const pt of s.path) vertex(pt[0], pt[1]);
      vertex(lx - 4, s.y);
      endShape();
      fill(c);
      noStroke();
      circle(s.path[0][0], s.path[0][1], 6);
    } else if (s.side === 'above') {
      line(s.x, s.y + bd / 2 + 2, s.x, g.ty);
    } else {
      // a bracket under the table shows how far the part extends
      const by = g.ty + g.h + 6;
      line(s.span[0] + 3, by, s.span[1] - 3, by);
      line(s.span[0] + 3, by, s.span[0] + 3, by - 4);
      line(s.span[1] - 3, by, s.span[1] - 3, by - 4);
      line(s.x, by, s.x, s.y - bd / 2 - 2);
    }

    drawBadge(i, lx + bd / 2, s.y, bd, c);
    noStroke();
    fill(c);
    textSize(ts);
    textStyle(BOLD);
    textAlign(LEFT, CENTER);
    text(p.name, lx + bd + 5, s.y + 1);
    hitBoxes.unshift({ x: lx - 4, y: s.y - bd / 2 - 4, w: lw + 8, h: bd + 8, part: i });
  }
  textStyle(NORMAL);
}

function drawBadge(i, cx, cy, d, c) {
  noStroke();
  fill(c);
  circle(cx, cy, d);
  fill('white');
  textStyle(BOLD);
  textSize(max(11, d * 0.62));
  textAlign(CENTER, CENTER);
  text(i + 1, cx, cy + 1);
}

// A dark chip holding one line of code. Returns the chip's width.
function codeChip(str, x, y, ts) {
  textSize(ts);
  textStyle(NORMAL);
  const w = textWidth(str) + 14;
  noStroke();
  fill('darkslategray');
  rect(x, y, w, ts + 9, 5);
  fill('white');
  textAlign(LEFT, CENTER);
  text(str, x + 7, y + (ts + 9) / 2 + 1);
  return w;
}

function drawInfoPanel(x, y, w, h, active, narrow, showCode) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = narrow ? 13 : 15, lh = ts + 6, pad = narrow ? 10 : 14;
  const ix = x + pad, iw = w - 2 * pad;
  if (active === null) drawOverview(ix, y, iw, h, ts, lh, narrow, showCode);
  else drawDetail(ix, y, iw, h, ts, lh, narrow, showCode, active);
  textStyle(NORMAL);
}

// The list of all six parts, shown while nothing is selected.
function drawOverview(ix, y, iw, h, ts, lh, narrow, showCode) {
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text('The six parts of a DataFrame', ix, y + 9);

  const bd = narrow ? 17 : 21;
  const twoLine = !narrow && showCode;
  const rowH = narrow ? 24 : (twoLine ? 48 : 36);
  const nameW = narrow ? 66 : 76;
  let ry = y + 9 + lh + (narrow ? 5 : 10);
  for (let i = 0; i < PARTS.length; i++) {
    const p = PARTS[i];
    drawBadge(i, ix + bd / 2, ry + bd / 2, bd, color(p.color));
    noStroke();
    textSize(ts);
    textAlign(LEFT, CENTER);
    textStyle(BOLD);
    fill(p.color);
    text(p.name, ix + bd + 6, ry + bd / 2 + 1);
    textStyle(NORMAL);
    fill('black');
    text(p.brief, ix + bd + 6 + nameW, ry + bd / 2 + 1);
    if (showCode && narrow) {
      fill('dimgray');
      textAlign(RIGHT, CENTER);
      text(p.code[0], ix + iw, ry + bd / 2 + 1);
    } else if (showCode) {
      codeChip(p.code[0], ix + bd + 6, ry + bd + 2, ts - 2);
    }
    hitBoxes.unshift({ x: ix - 4, y: ry - 2, w: iw + 8, h: rowH - 2, part: i });
    ry += rowH;
  }

  noStroke();
  fill('dimgray');
  textStyle(ITALIC);
  textSize(ts - (narrow ? 0 : 1));
  textAlign(LEFT, TOP);
  textWrap(WORD);
  text('Point at a part of the table to highlight it. Click to keep it selected.', ix, ry + (narrow ? 0 : 4), iw, y + h - ry);
}

// Name, meaning, code, and result for one part.
function drawDetail(ix, y, iw, h, ts, lh, narrow, showCode, active) {
  const p = PARTS[active];
  const bd = narrow ? 20 : 26;
  drawBadge(active, ix + bd / 2, y + 10 + bd / 2, bd, color(p.color));
  noStroke();
  fill(p.color);
  textStyle(BOLD);
  textSize(ts + 4);
  textAlign(LEFT, CENTER);
  text(p.name, ix + bd + 7, y + 10 + bd / 2 + 1);
  const nameW = textWidth(p.name);
  fill('dimgray');
  textStyle(NORMAL);
  textSize(ts);
  text('(' + p.brief + ')', ix + bd + 15 + nameW, y + 10 + bd / 2 + 2);

  // meaning
  let cy = y + 10 + bd + (narrow ? 6 : 10);
  fill('black');
  textAlign(LEFT, TOP);
  textWrap(WORD);
  textLeading(lh);
  const lines = countLines(p.text, iw);
  text(p.text, ix, cy, iw, lines * lh + 4);
  cy += lines * lh + (narrow ? 5 : 10);

  if (!showCode) {
    fill('dimgray');
    textStyle(ITALIC);
    text('Turn on Show code to see the pandas expression that returns this part.', ix, cy, iw, 3 * lh);
    textStyle(NORMAL);
  } else {
    // code: one chip per way of writing it, wrapped onto a new line when it does not fit
    const labelW = narrow ? 58 : 72;
    fill('black');
    textStyle(BOLD);
    textAlign(LEFT, CENTER);
    text('Code', ix, cy + (ts + 9) / 2 + 1);
    let cx = ix + labelW;
    for (let k = 0; k < p.code.length; k++) {
      textSize(ts);
      textStyle(NORMAL);
      const cwid = textWidth(p.code[k]) + 14;
      const orW = k > 0 ? textWidth('or') + 10 : 0;
      if (cx + orW + cwid > ix + iw && cx > ix + labelW) { cx = ix + labelW; cy += ts + 13; }
      if (k > 0) {
        fill('dimgray');
        textAlign(LEFT, CENTER);
        text('or', cx, cy + (ts + 9) / 2 + 1);
        cx += orW;
      }
      codeChip(p.code[k], cx, cy, ts);
      cx += cwid + 8;
    }
    cy += ts + 9 + (narrow ? 7 : 12);

    fill('black');
    textStyle(BOLD);
    textSize(ts);
    textAlign(LEFT, TOP);
    text('Returns', ix, cy);
    textStyle(NORMAL);
    cy = drawResult(p.result, ix + labelW, cy, iw - labelW, ts, narrow ? ts + 3 : ts + 5);
  }

  // memory tip at the foot of the panel, where there is room for it
  if (!narrow) {
    const tip = 'Remember: ' + p.tip;
    textSize(ts - 1);
    textStyle(NORMAL);
    const tipH = countLines(tip, iw - 18) * lh + 14;
    const tipY = y + h - tipH - 12;
    if (tipY > cy + 6) {
      fill('lightyellow');
      stroke('khaki');
      strokeWeight(1);
      rect(ix, tipY, iw, tipH, 8);
      noStroke();
      fill('black');
      textAlign(LEFT, TOP);
      textLeading(lh);
      text(tip, ix + 9, tipY + 8, iw - 18, tipH);
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

// Draw what the code returns, with pandas' column alignment. Returns the bottom y.
function drawResult(result, x, y, w, ts, lh) {
  noStroke();
  fill('midnightblue');
  textSize(ts);
  textStyle(NORMAL);
  textAlign(LEFT, TOP);
  if (result.kind === 'text') {
    const lines = countLines(result.str, w);
    textLeading(lh);
    text(result.str, x, y, w, lines * lh + 4);
    return y + lines * lh;
  }
  if (result.kind === 'series') {
    // labels on the left, values right-aligned, then the Series footer
    const labelW = max(result.labels.map(s => textWidth(s))) + 14;
    const valueW = max(result.values.map(v => textWidth(String(v))));
    for (let i = 0; i < result.labels.length; i++) {
      textAlign(LEFT, TOP);
      text(result.labels[i], x, y + i * lh);
      textAlign(RIGHT, TOP);
      text(result.values[i], x + labelW + valueW, y + i * lh);
    }
    textAlign(LEFT, TOP);
    text(result.footer, x, y + result.labels.length * lh);
    return y + (result.labels.length + 1) * lh;
  }
  // a 2D NumPy array: array([[ ... ], [ ... ]])
  const open = 'array([';
  const ow = textWidth(open), bw = textWidth('[');
  const numW = max(result.rows.map(row => max(row.map(v => textWidth(v + ','))))) + 6;
  for (let i = 0; i < result.rows.length; i++) {
    const row = result.rows[i], last = i === result.rows.length - 1;
    textAlign(LEFT, TOP);
    if (i === 0) text(open, x, y);
    text('[', x + ow, y + i * lh);
    textAlign(RIGHT, TOP);
    for (let j = 0; j < row.length; j++) {
      text(row[j] + (j < row.length - 1 ? ',' : ''), x + ow + bw + (j + 1) * numW, y + i * lh);
    }
    textAlign(LEFT, TOP);
    text(last ? ']])' : '],', x + ow + bw + row.length * numW + 1, y + i * lh);
  }
  return y + result.rows.length * lh;
}

function partAt(px, py) {
  if (py < 0 || py > drawHeight) return null;
  for (const b of hitBoxes) {
    if (px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h) return b.part;
  }
  return null;
}

// Click a part to keep it selected. Click it again, or click empty space, to clear.
function mousePressed() {
  if (mouseX < 0 || mouseX > canvasWidth || mouseY < 0 || mouseY > drawHeight) return;
  const hit = partAt(mouseX, mouseY);
  pinned = hit === pinned ? null : hit;
  partSelect.selected(pinned === null ? '-1' : String(pinned));
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
