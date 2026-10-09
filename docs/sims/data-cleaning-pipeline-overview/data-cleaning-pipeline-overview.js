// Data Cleaning Pipeline Overview
// CANVAS_HEIGHT: 593
// Bloom L2 (Understand): students step through the eight stages of a data cleaning pipeline
// with Next and Previous (or by clicking a stage) and explain what each stage changes in a
// small messy table. Nothing moves on its own.
//
// Model: the 7-row table in RAW is really cleaned here, stage by stage, the way pandas does it.
//   missing values:  the code -999 becomes NaN, then each NaN is filled with its column median
//   duplicates:      a row equal to an earlier row is dropped (drop_duplicates, keep="first")
//   outliers:        values outside Q1 - 1.5 IQR .. Q3 + 1.5 IQR are replaced by the column
//                    median (quartiles use linear interpolation, like Series.quantile)
//   data types:      age float64 -> int64, joined object -> datetime64[ns]
//   validation:      rows that break the rule "score between 0 and 100" are removed
//   transformation:  scaled = (score - min) / (max - min)
// Every count, fence, fill value, and log entry on screen is computed from the table.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 548;
let controlHeight = 45;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

// name, age, score, joined. null is a missing value (NaN).
const RAW = [
  ['Alice', 25, 80, '2024-01-15'],
  ['Bob', null, 88, '2024-02-03'],
  ['Cara', 22, null, '2024-02-10'],
  ['Bob', -999, 88, '2024-02-03'],
  ['Dev', 230, 72, '2024-03-01'],
  ['Eve', 27, 96, '2024-03-12'],
  ['Finn', 19, 104, '2024-03-20']
];
const SENTINEL = -999;            // a "no value" code that pandas does not treat as missing
const NUMERIC = [1, 2];           // positions of the age and score columns
const SCORE_RULE = [0, 100];      // business rule checked in the validation stage

// ink: the text color that is readable on top of the stage color
const STAGES = [
  { name: 'Raw Data', short: 'Raw Data', color: 'crimson', ink: 'white', hover: 'Data as received, full of problems' },
  { name: 'Missing Values', short: 'Missing', color: 'darkorange', ink: 'black', hover: 'Identify and handle NaN, None, and hidden codes' },
  { name: 'Duplicates', short: 'Duplicates', color: 'gold', ink: 'black', hover: 'Find and remove duplicate records' },
  { name: 'Outliers', short: 'Outliers', color: 'yellowgreen', ink: 'black', hover: 'Detect extreme values and decide how to handle them' },
  { name: 'Data Types', short: 'Data Types', color: 'seagreen', ink: 'white', hover: 'Convert columns to appropriate types' },
  { name: 'Validation', short: 'Validation', color: 'royalblue', ink: 'white', hover: 'Verify that the data meets business rules' },
  { name: 'Transformation', short: 'Transform', color: 'mediumpurple', ink: 'white', hover: 'Scale, normalize, and prepare for analysis' },
  { name: 'Clean Data', short: 'Clean Data', color: 'goldenrod', ink: 'black', hover: 'Analysis-ready dataset!' }
];

// Quantile with linear interpolation between sorted values (the pandas default)
function quantile(values, q) {
  const s = values.slice().sort((a, b) => a - b);
  const pos = (s.length - 1) * q, lo = Math.floor(pos);
  return s[lo] + (pos - lo) * (s[Math.min(lo + 1, s.length - 1)] - s[lo]);
}
const plural = (n, word) => n + ' ' + word + (n === 1 ? '' : 's');

// Run the pipeline once and keep a snapshot of the table after every stage.
// hi: cells changed in the stage ("rowId,col", or "d,col" for a dtype). gone: rows it removes.
const STATES = [];
function buildStates() {
  let rows = RAW.map((v, i) => ({ id: i, v: v.slice() }));
  const cols = ['name', 'age', 'score', 'joined'];
  const dtypes = ['object', 'float64', 'float64', 'object'];
  const log = [];
  const column = c => rows.map(r => r.v[c]).filter(v => v !== null);
  const snap = (extra) => STATES.push(Object.assign({ rows: rows.map(r => ({ id: r.id, v: r.v.slice() })),
    cols: cols.slice(), dtypes: dtypes.slice(), hi: [], gone: [], log: log.slice(), code: [] }, extra));
  const nIn = rows.length, cIn = cols.length;

  // 1. raw data
  snap({ code: ['df = pd.read_csv("members.csv")'],
    text: 'The table as it was loaded: ' + nIn + ' rows and ' + cIn + ' columns. Each stage of the pipeline catches a ' +
      'different kind of problem. Before you press Next, look closely. How many problems can you find?' });

  // 2. missing values: reveal the hidden code, then fill with the median
  let hi = [], hidden = 0;
  rows.forEach(r => NUMERIC.forEach(c => { if (r.v[c] === SENTINEL) { r.v[c] = null; hidden++; } }));
  const fills = NUMERIC.map(c => {
    const m = quantile(column(c), 0.5);
    rows.forEach(r => { if (r.v[c] === null) { r.v[c] = m; hi.push(r.id + ',' + c); } });
    return cols[c] + ' ' + m;
  });
  log.push(plural(hi.length, 'missing value') + ' (' + hidden + ' hidden as ' + SENTINEL + ') filled with medians');
  snap({ hi, logged: log[log.length - 1],
    code: ['df = df.replace(' + SENTINEL + ', np.nan)', 'df = df.fillna(df.median(numeric_only=True))'],
    text: 'isnull() finds NaN, but the code ' + SENTINEL + ' also means "no value". Once it is replaced by NaN, ' +
      plural(hi.length, 'cell') + ' are missing. Each one is filled with the median of its column (' + fills.join(', ') + ').' });

  // 3. duplicates: a row equal to an earlier row
  const seen = new Set();
  let gone = [];
  rows.forEach(r => { const k = JSON.stringify(r.v); if (seen.has(k)) gone.push(r.id); else seen.add(k); });
  log.push(plural(gone.length, 'duplicate row') + ' removed');
  snap({ gone, logged: log[log.length - 1],
    code: ['df.duplicated().sum()     →  ' + gone.length, 'df = df.drop_duplicates()'],
    text: 'Row ' + gone.join(', ') + ' is now an exact copy of an earlier row, so drop_duplicates() removes it. Order ' +
      'matters: before the missing values were handled these two rows did not match, because one held NaN and the ' +
      'other ' + SENTINEL + '.' });
  rows = rows.filter(r => !gone.includes(r.id));

  // 4. outliers: the 1.5 x IQR rule on each numeric column
  hi = [];
  const found = [];
  NUMERIC.forEach(c => {
    const vals = column(c), q1 = quantile(vals, 0.25), q3 = quantile(vals, 0.75);
    const lo = q1 - 1.5 * (q3 - q1), up = q3 + 1.5 * (q3 - q1), m = quantile(vals, 0.5);
    rows.forEach(r => {
      if (r.v[c] < lo || r.v[c] > up) {
        found.push('an ' + cols[c] + ' of ' + r.v[c] + ' (the fences are ' + lo.toFixed(1) + ' and ' + up.toFixed(1) + ')');
        r.v[c] = m;
        hi.push(r.id + ',' + c);
      }
    });
  });
  log.push(plural(hi.length, 'outlier') + ' replaced by the column median');
  snap({ hi, logged: log[log.length - 1],
    code: ['low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR', 'df.loc[df["age"] > high, "age"] = df["age"].median()'],
    text: 'The IQR rule flags values far outside the middle half of a column. It finds ' + found.join(' and ') +
      '. That cannot be real, so it is replaced by the median. A genuine extreme value would be kept.' });

  // 5. data types
  dtypes[1] = 'int64';
  dtypes[3] = 'datetime64[ns]';
  log.push('age converted to int64, joined to datetime64');
  snap({ hi: ['d,1', 'd,3'], logged: log[log.length - 1],
    code: ['df["age"] = df["age"].astype(int)', 'df["joined"] = pd.to_datetime(df["joined"])'],
    text: 'Each column should have the type that fits its meaning. age was float64 only because NaN is a float. With ' +
      'no NaN left it can be int64. joined was text (object). As datetime64 it can be sorted and subtracted.' });

  // 6. validation against a business rule
  const bad = rows.filter(r => r.v[2] < SCORE_RULE[0] || r.v[2] > SCORE_RULE[1]);
  gone = bad.map(r => r.id);
  log.push(plural(gone.length, 'row') + ' removed, score outside ' + SCORE_RULE[0] + ' to ' + SCORE_RULE[1]);
  snap({ gone, logged: log[log.length - 1],
    code: ['ok = df["score"].between(' + SCORE_RULE[0] + ', ' + SCORE_RULE[1] + ')', 'df = df[ok]'],
    text: 'Business rules say what is possible. A score must be between ' + SCORE_RULE[0] + ' and ' + SCORE_RULE[1] +
      '. Row ' + gone.join(', ') + ' has a score of ' + bad.map(r => r.v[2]).join(', ') + '. It was not extreme enough ' +
      'to be flagged as an outlier, but it breaks the rule, so the row is removed.' });
  rows = rows.filter(r => !gone.includes(r.id));

  // 7. transformation: min-max scaling into a new column
  const lowest = Math.min(...column(2)), highest = Math.max(...column(2));
  cols.push('scaled');
  dtypes.push('float64');
  rows.forEach(r => r.v.push((r.v[2] - lowest) / (highest - lowest)));
  log.push('scaled column added (min-max of score)');
  snap({ hi: rows.map(r => r.id + ',' + (cols.length - 1)), logged: log[log.length - 1],
    code: ['lo, hi = df["score"].min(), df["score"].max()', 'df["scaled"] = ((df["score"] - lo) / (hi - lo)).round(2)'],
    text: 'Models compare columns more fairly when the columns share a scale. Min-max scaling maps the lowest score (' +
      lowest + ') to 0 and the highest (' + highest + ') to 1 in a new column.' });

  // 8. clean data
  snap({ report: true,
    text: nIn + ' rows and ' + cIn + ' columns came in. ' + rows.length + ' rows and ' + cols.length + ' columns go on ' +
      'to analysis. Always document your cleaning decisions.' });
}
buildStates();

let step = 0;
let prevButton, nextButton, resetButton;
let hitBoxes = [];                // clickable pipeline stages: { x, y, w, h, stage }

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  prevButton = createButton('Previous');
  prevButton.parent(mainElement);
  prevButton.position(10, drawHeight + 10);
  prevButton.mousePressed(() => goToStep(step - 1));

  nextButton = createButton('Next');
  nextButton.parent(mainElement);
  nextButton.position(86, drawHeight + 10);
  nextButton.mousePressed(() => goToStep(step + 1));

  resetButton = createButton('Reset');
  resetButton.parent(mainElement);
  resetButton.position(138, drawHeight + 10);
  resetButton.mousePressed(() => goToStep(0));

  goToStep(0);

  describe('An eight-stage data cleaning pipeline: raw data, missing values, duplicates, outliers, data types, ' +
    'validation, transformation, and clean data. Next and Previous buttons step through the stages. A small table ' +
    'shows the data after each stage with the changed cells highlighted, and a panel explains the stage, shows the ' +
    'pandas code, and lists what was written to the data quality report.', LABEL);
}

// Move to a stage. A button that has nowhere to go is disabled.
function goToStep(n) {
  step = constrain(n, 0, STATES.length - 1);
  if (step === 0) prevButton.attribute('disabled', ''); else prevButton.removeAttribute('disabled');
  if (step === STATES.length - 1) nextButton.attribute('disabled', ''); else nextButton.removeAttribute('disabled');
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
  const st = STATES[step], stage = STAGES[step];
  textWrap(WORD);

  noStroke();
  fill('black');
  textStyle(NORMAL);
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Data Cleaning Pipeline', canvasWidth / 2, 8);

  const flowBottom = drawFlow(narrow ? 34 : 44, narrow);
  cursor(hitBoxes.some(mouseOver) ? HAND : ARROW);

  // two panels: the table after this stage, and what the stage does
  const top = flowBottom + (narrow ? 6 : 12), bottom = drawHeight - 8, fullW = canvasWidth - 2 * margin;
  if (narrow) {
    const tableH = 196;
    drawTable(margin, top, fullW, tableH, st, stage, narrow);
    drawExplain(margin, top + tableH + 6, fullW, bottom - top - tableH - 6, st, stage, narrow);
  } else {
    const leftW = fullW * 0.5 - 5;
    drawTable(margin, top, leftW, bottom - top, st, stage, narrow);
    drawExplain(margin + leftW + 10, top, fullW - leftW - 10, bottom - top, st, stage, narrow);
  }
}

// The eight stages as numbered boxes: one row when wide, two rows when narrow.
// Returns the y coordinate of the bottom of the flowchart.
function drawFlow(y, narrow) {
  const perRow = narrow ? 4 : 8, gap = narrow ? 8 : 14, h = narrow ? 34 : 54;
  const bw = (canvasWidth - 2 * margin - (perRow - 1) * gap) / perRow;
  hitBoxes = [];
  for (let i = 0; i < STAGES.length; i++) {
    const s = STAGES[i], current = i === step;
    const x = margin + (i % perRow) * (bw + gap), by = y + Math.floor(i / perRow) * (h + 6);
    const c = color(s.color);
    c.setAlpha(current ? 90 : i < step ? 40 : 12);
    fill(c);
    stroke(current ? 'black' : s.color);
    strokeWeight(current ? 2.5 : 1.2);
    rect(x, by, bw, h, 8);

    // number badge and name: stacked when wide, side by side when narrow
    const d = narrow ? 17 : 22;
    const bx = narrow ? x + 13 : x + bw / 2, bcy = narrow ? by + h / 2 : by + 16;
    noStroke();
    fill(s.color);
    circle(bx, bcy, d);
    fill(s.ink);
    textStyle(BOLD);
    textAlign(CENTER, CENTER);
    textSize(narrow ? 11 : 13);
    text(i + 1, bx, bcy + 1);
    fill('black');
    if (narrow) {
      textAlign(LEFT, CENTER);
      text(s.short, x + 25, by + h / 2 + 1);
    } else {
      text(s.short, x + bw / 2, by + 39);
    }
    textStyle(NORMAL);

    // arrow to the next stage in the same row
    if (i % perRow < perRow - 1) {
      const ax = x + bw + gap / 2, ay = by + h / 2, a = narrow ? 3 : 5;
      noStroke();
      fill(i < step ? 'dimgray' : 'silver');
      triangle(ax + a, ay, ax - a, ay - a - 1, ax - a, ay + a + 1);
    }
    hitBoxes.push({ x, y: by, w: bw, h, stage: i });
  }
  return y + (narrow ? 2 * h + 6 : h);
}

// How pandas prints a value of the given column
function cellText(v, c, st) {
  if (v === null) return 'NaN';
  if (typeof v === 'string') return v;
  if (st.cols[c] === 'scaled') return v.toFixed(2);
  return st.dtypes[c] === 'float64' ? v.toFixed(1) : String(v);
}

// The table after this stage. Changed cells are tinted, removed rows are struck out.
function drawTable(x, y, w, h, st, stage, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = narrow ? 11 : 14, pad = narrow ? 8 : 12;
  const headH = narrow ? 22 : 34, footH = narrow ? 18 : 30;

  noStroke();
  fill('black');
  textStyle(BOLD);
  textSize(ts + 1);
  textAlign(LEFT, CENTER);
  text('DataFrame df', x + pad, y + headH / 2 + 1);
  textStyle(NORMAL);
  fill('dimgray');
  textAlign(RIGHT, CENTER);
  text('df.shape  →  (' + (st.rows.length - st.gone.length) + ', ' + st.cols.length + ')', x + w - pad, y + headH / 2 + 1);

  // header row, dtype row, and room for the 7 raw rows
  const rowH = min((h - headH - footH) / (RAW.length + 2), 36);
  const idxW = narrow ? 18 : 28, gx = x + pad, gw = w - 2 * pad, gy = y + headH;
  // column widths in proportion to the widest entry of each column
  textSize(ts);
  textStyle(BOLD);
  const natural = st.cols.map((name, c) =>
    max([textWidth(name), textWidth(st.dtypes[c])].concat(st.rows.map(r => textWidth(cellText(r.v[c], c, st))))) + 14);
  const scale = (gw - idxW) / natural.reduce((a, b) => a + b, 0);
  const colW = natural.map(n => n * scale), colX = [];
  colW.reduce((acc, cw, c) => { colX[c] = acc; return acc + cw; }, gx + idxW);

  noStroke();
  fill('gainsboro');
  rect(gx, gy, gw, rowH);
  fill('whitesmoke');
  rect(gx, gy + rowH, gw, rowH);
  const tint = color(stage.color);
  tint.setAlpha(110);

  for (let c = 0; c < st.cols.length; c++) {
    const numeric = typeof st.rows[0].v[c] === 'number';
    const tx = numeric ? colX[c] + colW[c] - 7 : colX[c] + 7;
    const typeChanged = st.hi.includes('d,' + c);
    if (typeChanged) {
      fill(tint);
      rect(colX[c], gy + rowH, colW[c], rowH);
    }
    textAlign(numeric ? RIGHT : LEFT, CENTER);
    fill('black');
    textStyle(BOLD);
    textSize(ts);
    text(st.cols[c], tx, gy + rowH / 2 + 1);
    textStyle(typeChanged ? BOLD : NORMAL);
    fill(typeChanged ? 'black' : 'dimgray');
    textSize(ts - (narrow ? 0 : 2));
    text(st.dtypes[c], tx, gy + rowH * 1.5 + 1);
    textSize(ts);
    for (let k = 0; k < st.rows.length; k++) {
      const r = st.rows[k], changed = st.hi.includes(r.id + ',' + c);
      if (changed) {
        fill(tint);
        rect(colX[c], gy + (k + 2) * rowH, colW[c], rowH);
      }
      const v = r.v[c];
      fill(v === null ? 'firebrick' : 'black');
      textStyle(changed ? BOLD : NORMAL);
      text(cellText(v, c, st), tx, gy + (k + 2.5) * rowH + 1);
    }
  }

  // index labels, grid lines, and the strike through removed rows
  for (let k = 0; k < st.rows.length; k++) {
    const ry = gy + (k + 2) * rowH, removed = st.gone.includes(st.rows[k].id);
    noStroke();
    if (removed) {
      fill(220, 20, 60, 45);
      rect(gx, ry, gw, rowH);
    }
    fill('dimgray');
    textStyle(BOLD);
    textAlign(CENTER, CENTER);
    text(st.rows[k].id, gx + idxW / 2, ry + rowH / 2 + 1);
    stroke(removed ? 'crimson' : 'gainsboro');
    strokeWeight(removed ? 1.5 : 1);
    if (removed) line(gx + idxW, ry + rowH / 2, gx + gw - 4, ry + rowH / 2);
    else line(gx, ry + rowH, gx + gw, ry + rowH);
  }
  noFill();
  stroke('silver');
  strokeWeight(1);
  rect(gx, gy, gw, (st.rows.length + 2) * rowH);

  // what the marks mean
  const key = st.gone.length > 0 ? 'The struck-out row is removed in this stage.' :
    st.hi.length > 0 ? 'Tinted cells were changed in this stage.' :
    st.report ? 'This table is ready for analysis.' : 'Nothing has been changed yet.';
  noStroke();
  fill('dimgray');
  textStyle(ITALIC);
  textSize(narrow ? 11 : 13);
  textAlign(LEFT, BOTTOM);
  text(key, x + pad, y + h - (narrow ? 4 : 9));
  textStyle(NORMAL);
}

// A dark chip holding one line of code. The text shrinks, down to 11px, to fit the width.
function codeChip(str, x, y, w, ts) {
  textStyle(NORMAL);
  textSize(ts);
  if (textWidth(str) > w - 14) textSize(max(11, ts * (w - 14) / textWidth(str)));
  noStroke();
  fill('darkslategray');
  rect(x, y, min(w, textWidth(str) + 14), ts + 9, 5);
  fill('white');
  textAlign(LEFT, CENTER);
  text(str, x + 7, y + (ts + 9) / 2 + 1);
}

// What the stage does: explanation on top, code and the report entry anchored at the bottom.
function drawExplain(x, y, w, h, st, stage, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const pad = narrow ? 9 : 14, ix = x + pad, iw = w - 2 * pad;
  const ts = narrow ? 12 : 15, lh = ts + (narrow ? 3 : 6), pillH = ts + 8;

  // step pill, stage name, and the one-line summary of the stage
  const pill = 'Step ' + (step + 1) + ' of ' + STATES.length;
  noStroke();
  textStyle(BOLD);
  textSize(ts - 1);
  const pillW = textWidth(pill) + 16;
  fill(stage.color);
  rect(ix, y + pad, pillW, pillH, pillH / 2);
  fill(stage.ink);
  textAlign(CENTER, CENTER);
  text(pill, ix + pillW / 2, y + pad + pillH / 2 + 1);
  fill('black');
  textSize(ts + 1);
  textAlign(LEFT, CENTER);
  text(stage.name, ix + pillW + 8, y + pad + pillH / 2 + 1);
  textStyle(ITALIC);
  textSize(ts);
  fill('dimgray');
  textAlign(LEFT, TOP);
  let cy = y + pad + pillH + (narrow ? 4 : 8);
  text(stage.hover, ix, cy);
  cy += lh + (narrow ? 2 : 6);
  textStyle(NORMAL);

  // bottom up: the entry written to the data quality report, then the code
  let by = y + h - pad;
  const entries = st.report ? st.log : st.logged ? [st.logged] : [];
  if (entries.length > 0) {
    const boxH = (st.report ? (narrow ? 7 : 9) : 2) * lh + 12;
    by -= boxH;
    fill('lightyellow');
    stroke('goldenrod');
    strokeWeight(1);
    drawingContext.setLineDash([4, 3]);
    rect(ix, by, iw, boxH, 6);
    drawingContext.setLineDash([]);
    noStroke();
    fill('black');
    textSize(ts);
    textLeading(lh);
    textAlign(LEFT, TOP);
    const body = st.report ? 'Data quality report\n' + entries.map((e, i) => 'Step ' + (i + 2) + ': ' + e).join('\n')
      : 'Logged in the data quality report: ' + entries[0];
    text(body, ix + 8, by + 6, iw - 16, boxH - 6);
    by -= narrow ? 5 : 10;
  }
  const chipH = ts + 9 + (narrow ? 3 : 6);
  by -= st.code.length * chipH;
  st.code.forEach((line, i) => codeChip(line, ix, by + i * chipH, iw, ts - (narrow ? 1 : 2)));

  noStroke();
  fill('black');
  textStyle(NORMAL);
  textSize(ts);
  textLeading(lh);
  textAlign(LEFT, TOP);
  text(st.text, ix, cy, iw, by - cy);
}

// Clicking a stage of the pipeline jumps to it.
const mouseOver = b => mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h;
function mousePressed() {
  const box = hitBoxes.find(mouseOver);
  if (box) goToStep(box.stage);
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
