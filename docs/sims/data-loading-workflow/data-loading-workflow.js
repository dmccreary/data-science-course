// Data Loading Workflow: From CSV File to DataFrame
// CANVAS_HEIGHT: 525
// Bloom L2 (Understand): students step through what pd.read_csv() does to a small CSV file,
// one stage at a time, and explain how raw text becomes a typed table in memory. Nothing moves
// on its own: the student presses Next and Previous, or clicks a stage of the flowchart.
//
// Model: the file text in CSV_TEXT is really parsed here, the way read_csv parses it with its
// default settings. Lines are split at the comma, the first line gives the column names, each
// column gets one dtype (all whole numbers: int64; numbers with a blank or a decimal: float64;
// anything else: object), and an empty field becomes NaN. Every count, type, and result shown
// is computed from that text. It is the chapter's students.csv with one score left blank so
// that the missing-value step has something to handle.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 480;
let controlHeight = 45;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const FILE_NAME = 'students.csv';
const CSV_TEXT = 'name,age,city,score\nAlice,25,New York,85\nBob,30,Los Angeles,92\nCharlie,22,Chicago,\nDiana,28,Houston,95';
const DELIM = ',';                                    // read_csv's default separator

// ---- parse the text ----
const LINES = CSV_TEXT.split('\n');
const GRID = LINES.map(line => line.split(DELIM));
const HEADER = GRID[0], BODY = GRID.slice(1);
const N_FIELDS = HEADER.length, N_ROWS = BODY.length;

function inferType(values) {
  const present = values.filter(v => v !== '');
  if (present.length > 0 && present.every(v => /^-?\d+$/.test(v))) {
    return present.length === values.length ? 'int64' : 'float64';   // NaN forces float
  }
  if (present.length > 0 && present.every(v => !isNaN(Number(v)))) return 'float64';
  return 'object';
}
const DTYPES = HEADER.map((name, c) => inferType(BODY.map(row => row[c])));
const TYPE_COLORS = { object: 'slateblue', int64: 'seagreen', float64: 'chocolate' };
const namesOfType = t => HEADER.filter((name, c) => DTYPES[c] === t).join(', ');

const MISSING = [];                                   // [row, column] of each empty field
BODY.forEach((row, r) => row.forEach((v, c) => { if (v === '') MISSING.push([r, c]); }));
const [missRow, missCol] = MISSING[0];
const firstPresent = BODY.map(row => row[missCol]).find(v => v !== '');

// two analysis results for the last step. Like pandas, the mean skips NaN.
const scoreCol = HEADER.indexOf('score'), ageCol = HEADER.indexOf('age');
const scores = BODY.map(row => row[scoreCol]).filter(v => v !== '').map(Number);
const MEAN_SCORE = scores.reduce((a, b) => a + b, 0) / scores.length;
const OLDER = BODY.filter(row => Number(row[ageCol]) > 25).map(row => row[0]);

// color: box outline and icon. ink: a darker shade for text and for white-on-color labels.
const STAGES = [
  { name: 'CSV file on disk', short: 'CSV file', color: 'dimgray', ink: 'dimgray' },
  { name: 'pd.read_csv()', short: 'read_csv()', color: 'darkorange', ink: 'chocolate' },
  { name: 'Parsing', short: 'Parsing', color: 'royalblue', ink: 'royalblue' },
  { name: 'DataFrame created', short: 'DataFrame', color: 'seagreen', ink: 'seagreen' },
  { name: 'Ready for analysis', short: 'Analysis', color: 'goldenrod', ink: 'darkgoldenrod' }
];

// Eight steps over the five stages. Parsing has four steps of its own.
const STEPS = [
  { stage: 0, title: 'A CSV file on disk', view: 'file',
    caption: LINES.length + ' lines of plain text',
    text: 'A CSV file is raw text with comma-separated values. Each line is one row, and the first line usually ' +
      'holds the column names. This file has ' + LINES.length + ' lines. A real one could have 100 rows or 100 million.',
    error: ['Not really a CSV', 'An Excel workbook (.xlsx) is not plain text, even if you rename it to .csv. ' +
      'Open the file in a text editor first: you should see readable lines like these.'] },
  { stage: 1, title: 'Call pd.read_csv()', view: 'file',
    caption: 'pandas opens the file and reads the text',
    text: 'One line of Python hands the file to pandas. Pandas looks for "' + FILE_NAME + '" in the current working ' +
      'directory, unless you give a longer path. Then it reads and parses the text.',
    error: ['FileNotFoundError', 'Check your file path! The name is misspelled, or the file is in a different folder ' +
      'from your notebook.'] },
  { stage: 2, title: 'Parsing: detect the delimiter', view: 'delim',
    caption: 'Split at each "' + DELIM + '"  →  ' + N_FIELDS + ' fields on every line',
    text: 'The delimiter is the character that separates one value from the next. read_csv expects a comma. For a ' +
      'tab-separated file you would pass sep="\\t". Splitting at the commas gives ' + N_FIELDS + ' fields on every line.',
    error: ['ParserError', 'Check the file format and delimiter. One extra comma breaks the pattern: "Expected ' +
      N_FIELDS + ' fields in line 3, saw ' + (N_FIELDS + 1) + '".'] },
  { stage: 2, title: 'Parsing: read the header row', view: 'header',
    caption: 'Column names: ' + HEADER.join(', '),
    text: 'The first line becomes the column names. The other ' + N_ROWS + ' lines are the data rows. At this point ' +
      'every value is still a piece of text, even the numbers.',
    error: ['No header in the file', 'If the first line is data, pandas still uses it for the column names and you ' +
      'lose a row. Fix it with header=None and names=[...].'] },
  { stage: 2, title: 'Parsing: infer the data types', view: 'types',
    caption: 'One data type (dtype) per column',
    text: 'Pandas reads down each column and picks one type for it. Text columns (' + namesOfType('object') +
      ') become object. Whole-number columns (' + namesOfType('int64') + ') become int64. Numeric columns with a ' +
      'blank or a decimal (' + namesOfType('float64') + ') become float64.',
    error: ['Numbers stored as text', 'One stray entry such as "85%" or "unknown" turns a whole numeric column into ' +
      'object, and then mean() fails. Check with df.dtypes.'] },
  { stage: 2, title: 'Parsing: handle missing values', view: 'missing',
    caption: MISSING.length + ' empty field  →  NaN',
    text: 'The empty field in the row for ' + BODY[missRow][0] + ' has no value, so pandas stores NaN (Not a Number) ' +
      'there. NaN is a float, so the whole ' + HEADER[missCol] + ' column is stored as floats: ' + firstPresent +
      ' becomes ' + Number(firstPresent).toFixed(1) + '.',
    error: ['Missing values in disguise', 'A blank becomes NaN, but a placeholder such as "?" or -999 does not. ' +
      'Name it with na_values=[...], then count with df.isnull().sum().'] },
  { stage: 3, title: 'The DataFrame is created', view: 'frame',
    caption: 'df.shape  →  (' + N_ROWS + ', ' + N_FIELDS + ')',
    text: 'Parsing is finished. The data now lives in computer memory (RAM) as a DataFrame with ' + N_ROWS +
      ' rows and ' + N_FIELDS + ' columns. The file had no row labels, so pandas added the index 0 to ' +
      (N_ROWS - 1) + ' on the left.',
    error: ['MemoryError', 'The whole table has to fit in RAM. For a file that is too big, load part of it with ' +
      'nrows= or usecols=.'] },
  { stage: 4, title: 'Ready for analysis', view: 'frame',
    caption: 'df.shape  →  (' + N_ROWS + ', ' + N_FIELDS + ')',
    text: 'This is where the fun begins: filter, analyze, visualize! Each question is now one line of code.',
    examples: [['df["score"].mean()', MEAN_SCORE.toFixed(2) + '  (NaN is skipped)'],
      ['df[df["age"] > 25]', OLDER.length + ' rows: ' + OLDER.join(', ')]],
    error: ['KeyError', 'Column names are case-sensitive. df["Score"] fails because the column is "score". ' +
      'Check the exact names with df.columns.'] }
];

let step = 0;
let prevButton, nextButton, resetButton, errorCheckbox;
let hitBoxes = [];            // clickable flowchart stages: { x, y, w, h, stage }

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

  errorCheckbox = createCheckbox(' Show common errors', true);
  errorCheckbox.parent(mainElement);
  errorCheckbox.position(204, drawHeight + 11);
  errorCheckbox.style('font-size', '16px');

  goToStep(0);

  describe('A five-stage flowchart of loading a CSV file with pandas: CSV file on disk, pd.read_csv, parsing, ' +
    'DataFrame created, and ready for analysis. Next and Previous buttons step through eight steps. For each step a ' +
    'panel shows the file text or the table as it looks at that moment, and a second panel explains what happens ' +
    'and what commonly goes wrong.', LABEL);
}

// Move to a step. A button that has nowhere to go is dimmed.
function goToStep(n) {
  step = constrain(n, 0, STEPS.length - 1);
  prevButton.style('opacity', step === 0 ? '0.45' : '1');
  nextButton.style('opacity', step === STEPS.length - 1 ? '0.45' : '1');
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
  const s = STEPS[step];
  const showErrors = errorCheckbox.checked();

  noStroke();
  fill('black');
  textStyle(NORMAL);
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Data Loading Workflow: CSV File to DataFrame', canvasWidth / 2, 8);

  // flowchart
  const flowY = narrow ? 36 : 42, flowH = narrow ? 68 : 88;
  hitBoxes = [];
  drawFlow(flowY, flowH, s.stage, narrow, showErrors);
  const overStage = hitBoxes.some(b => mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h);
  cursor(overStage ? HAND : ARROW);

  // step caption
  const capY = flowY + flowH + (narrow ? 6 : 9);
  const ts = narrow ? 13 : 16;
  const pill = 'Step ' + (step + 1) + ' of ' + STEPS.length;
  textSize(ts - 1);
  textStyle(BOLD);
  const pillW = textWidth(pill) + 16;
  noStroke();
  fill(STAGES[s.stage].ink);
  rect(margin, capY, pillW, ts + 8, (ts + 8) / 2);
  fill('white');
  textAlign(CENTER, CENTER);
  text(pill, margin + pillW / 2, capY + (ts + 8) / 2 + 1);
  fill('black');
  textSize(ts);
  textAlign(LEFT, CENTER);
  text(s.title, margin + pillW + 8, capY + (ts + 8) / 2 + 1);
  textStyle(NORMAL);

  // two panels: what the data looks like now, and what is happening
  const panelY = capY + ts + (narrow ? 14 : 18);
  const bottom = drawHeight - 8;
  if (narrow) {
    const dataH = 158;
    drawDataPanel(margin, panelY, canvasWidth - 2 * margin, dataH, s, narrow);
    drawExplainPanel(margin, panelY + dataH + 6, canvasWidth - 2 * margin, bottom - panelY - dataH - 6, s, narrow, showErrors);
  } else {
    const leftW = (canvasWidth - 2 * margin - 10) * 0.5;
    drawDataPanel(margin, panelY, leftW, bottom - panelY, s, narrow);
    drawExplainPanel(margin + leftW + 10, panelY, canvasWidth - 2 * margin - leftW - 10, bottom - panelY, s, narrow, showErrors);
  }
}

// The five stages as a row of boxes joined by arrows. The current stage is emphasized.
function drawFlow(y, h, stageNow, narrow, showErrors) {
  const n = STAGES.length;
  const gap = narrow ? 10 : 24;
  const bw = (canvasWidth - 2 * margin - (n - 1) * gap) / n;
  for (let i = 0; i < n; i++) {
    const st = STAGES[i];
    const x = margin + i * (bw + gap);
    const current = i === stageNow;
    const c = color(st.color);
    c.setAlpha(current ? 70 : 22);
    fill('white');
    noStroke();
    rect(x, y, bw, h, 10);
    fill(c);
    stroke(st.color);
    strokeWeight(current ? 3 : 1.2);
    rect(x, y, bw, h, 10);

    const iconS = narrow ? 22 : 30;
    drawStageIcon(i, x + bw / 2, y + (narrow ? 21 : 26), iconS, st.color);

    // parsing shows its four steps as dots: filled when reached
    if (i === 2) {
      const first = STEPS.findIndex(sp => sp.stage === 2);
      const count = STEPS.filter(sp => sp.stage === 2).length;
      const dy = y + (narrow ? 41 : 52), dd = narrow ? 6 : 8, dg = narrow ? 5 : 7;
      for (let k = 0; k < count; k++) {
        const dx = x + bw / 2 + (k - (count - 1) / 2) * (dd + dg);
        stroke(st.color);
        strokeWeight(1.2);
        fill(step >= first + k ? st.color : 'white');
        circle(dx, dy, dd);
        if (step === first + k) {
          noFill();
          circle(dx, dy, dd + 5);
        }
      }
    }

    // stage number and name
    noStroke();
    fill(st.ink);
    circle(x + (narrow ? 11 : 15), y + (narrow ? 11 : 15), narrow ? 15 : 20);
    fill('white');
    textStyle(BOLD);
    textAlign(CENTER, CENTER);
    textSize(narrow ? 11 : 13);
    text(i + 1, x + (narrow ? 11 : 15), y + (narrow ? 12 : 16));
    fill('black');
    textSize(narrow ? 11 : 14);
    text(narrow || bw < 132 ? st.short : st.name, x + bw / 2, y + h - (narrow ? 11 : 15));
    textStyle(NORMAL);

    // a red mark on the current stage ties it to the error box
    if (current && showErrors) {
      const mx = x + bw - (narrow ? 11 : 15), my = y + (narrow ? 11 : 15), md = narrow ? 15 : 20;
      fill('firebrick');
      circle(mx, my, md);
      fill('white');
      textStyle(BOLD);
      textSize(narrow ? 11 : 14);
      text('!', mx, my + 1);
      textStyle(NORMAL);
    }

    // arrow to the next stage
    if (i < n - 1) {
      const ax = x + bw, ay = y + h / 2;
      const reached = i < stageNow;
      stroke(reached ? 'dimgray' : 'silver');
      strokeWeight(2);
      if (!narrow) line(ax + 4, ay, ax + gap - 9, ay);
      noStroke();
      fill(reached ? 'dimgray' : 'silver');
      triangle(ax + gap - 2, ay, ax + gap - (narrow ? 8 : 10), ay - 5, ax + gap - (narrow ? 8 : 10), ay + 5);
    }
    hitBoxes.push({ x, y, w: bw, h, stage: i });
  }
}

// Small line icons for the stages: document, function call, gears, table, bar chart.
function drawStageIcon(i, cx, cy, s, col) {
  push();
  translate(cx, cy);
  stroke(col);
  strokeWeight(1.5);
  fill('white');
  if (i === 0) {
    const w = s * 0.74, h = s, f = s * 0.26;
    beginShape();
    vertex(-w / 2, -h / 2); vertex(w / 2 - f, -h / 2); vertex(w / 2, -h / 2 + f); vertex(w / 2, h / 2); vertex(-w / 2, h / 2);
    endShape(CLOSE);
    line(w / 2 - f, -h / 2, w / 2 - f, -h / 2 + f);
    line(w / 2 - f, -h / 2 + f, w / 2, -h / 2 + f);
    strokeWeight(1);
    for (let k = 0; k < 3; k++) line(-w / 2 + 4, -h / 2 + s * 0.46 + k * s * 0.18, w / 2 - 4, -h / 2 + s * 0.46 + k * s * 0.18);
  } else if (i === 1) {
    rect(-s * 0.75, -s * 0.42, s * 1.5, s * 0.84, 6);
    noStroke();
    fill(col);
    textStyle(BOLD);
    textSize(s * 0.5);
    textAlign(CENTER, CENTER);
    text('pd.( )', 0, 1);
  } else if (i === 2) {
    drawGear(-s * 0.2, s * 0.1, s * 0.34, 8, col);
    drawGear(s * 0.42, -s * 0.2, s * 0.21, 6, col);
  } else if (i === 3) {
    const w = s * 1.2, h = s * 0.92;
    rect(-w / 2, -h / 2, w, h, 2);
    noStroke();
    fill(col);
    rect(-w / 2, -h / 2, w, h / 3, 2);
    stroke(col);
    strokeWeight(1);
    line(-w / 2, h / 6, w / 2, h / 6);
    line(-w / 6, -h / 2, -w / 6, h / 2);
    line(w / 6, -h / 2, w / 6, h / 2);
  } else {
    noStroke();
    fill(col);
    const bw = s * 0.22, base = s * 0.46;
    [0.35, 0.62, 0.9].forEach((frac, k) => rect(-s * 0.5 + k * (bw + 3), base - frac * s, bw, frac * s));
    // four-point sparkle
    const sx = s * 0.52, sy = -s * 0.3, r = s * 0.24;
    beginShape();
    for (let k = 0; k < 8; k++) {
      const rr = k % 2 === 0 ? r : r * 0.32;
      vertex(sx + rr * cos(k * QUARTER_PI), sy + rr * sin(k * QUARTER_PI));
    }
    endShape(CLOSE);
  }
  pop();
}

function drawGear(x, y, r, teeth, col) {
  push();
  translate(x, y);
  noStroke();
  fill(col);
  for (let k = 0; k < teeth; k++) {
    rotate(TWO_PI / teeth);
    rect(-r * 0.2, -r * 1.3, r * 0.4, r * 0.5, 1);
  }
  circle(0, 0, r * 2);
  fill('white');
  circle(0, 0, r * 0.8);
  pop();
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

// What the data looks like at this step: file text, a grid of fields, or the DataFrame.
function drawDataPanel(x, y, w, h, s, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = narrow ? 12 : 15, pad = narrow ? 8 : 12;
  const headH = narrow ? 28 : 40, footH = narrow ? 20 : 30;

  // header strip: the file name before the call, the call from then on
  noStroke();
  if (step === 0) {
    fill('black');
    textStyle(BOLD);
    textSize(ts);
    textAlign(LEFT, CENTER);
    text(FILE_NAME, x + pad, y + headH / 2 + 2);
    const nameW = textWidth(FILE_NAME);
    textStyle(NORMAL);
    fill('dimgray');
    text('a text file on disk', x + pad + nameW + 8, y + headH / 2 + 2);
  } else {
    codeChip('df = pd.read_csv("' + FILE_NAME + '")', x + pad, y + (headH - ts - 9) / 2 + 2, ts);
  }

  const bodyY = y + headH, bodyH = h - headH - footH - 4;
  const rowH = min(bodyH / (N_ROWS + 2), 34);         // header, data rows, and a dtype row
  if (s.view === 'file' || s.view === 'delim') drawFileText(x + pad, bodyY, w - 2 * pad, rowH, s.view === 'delim', ts);
  else drawGrid(x + pad, bodyY, w - 2 * pad, rowH, s.view, ts, narrow);

  // caption, a little smaller if the panel is slim
  noStroke();
  fill(STAGES[s.stage].ink);
  textStyle(BOLD);
  textSize(ts);
  if (textWidth(s.caption) > w - 2 * pad) textSize(max(11, ts * (w - 2 * pad) / textWidth(s.caption)));
  textAlign(CENTER, CENTER);
  text(s.caption, x + w / 2, y + h - footH / 2 - 3);
  textStyle(NORMAL);
}

// The raw lines of the file. With showDelims on, every comma is marked.
function drawFileText(x, y, w, rowH, showDelims, ts) {
  const h = rowH * LINES.length + 8;
  fill('ivory');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 4);
  textSize(ts + 1);
  textStyle(NORMAL);
  textAlign(LEFT, CENTER);
  for (let r = 0; r < LINES.length; r++) {
    const cy = y + 4 + (r + 0.5) * rowH;
    let cx = x + 10;
    if (!showDelims) {
      noStroke();
      fill('black');
      text(LINES[r], cx, cy);
      continue;
    }
    const fields = GRID[r];
    for (let c = 0; c < fields.length; c++) {
      noStroke();
      fill('black');
      text(fields[c], cx, cy);
      cx += textWidth(fields[c]);
      if (c < fields.length - 1) {
        const dw = textWidth(DELIM) + 8;
        fill('orange');
        rect(cx + 2, cy - rowH * 0.36, dw, rowH * 0.72, 4);
        fill('black');
        textStyle(BOLD);
        text(DELIM, cx + 6, cy);
        textStyle(NORMAL);
        cx += dw + 4;
      }
    }
  }
}

// The value pandas holds for a field once parsing has reached the given view
function cellText(r, c, view) {
  const raw = BODY[r][c];
  if (view === 'header' || view === 'types') return raw;         // still the text from the file
  if (raw === '') return 'NaN';
  return DTYPES[c] === 'float64' ? Number(raw).toFixed(1) : raw;
}

// The fields as a table. The left column is kept free for the index pandas adds at the end.
function drawGrid(x, y, w, rowH, view, ts, narrow) {
  const isFrame = view === 'frame';
  const idxW = narrow ? 24 : 34;
  // column widths in proportion to the widest entry of each column. The text shrinks, down
  // to 11px, until the columns fit.
  let natural, scale;
  for (; ; ts--) {
    textSize(ts);
    textStyle(BOLD);
    natural = HEADER.map((name, c) => max([textWidth(name), textWidth('float64')].concat(BODY.map(row => textWidth(row[c])))) + 18);
    scale = (w - idxW) / natural.reduce((a, b) => a + b, 0);
    if (scale >= 0.86 || ts <= 11) break;     // the 18px of padding can give a little
  }
  const colW = natural.map(v => v * scale);
  const colX = [];
  colW.reduce((acc, v, c) => { colX[c] = acc; return acc + v; }, x + idxW);
  const gridW = w - idxW, gridH = rowH * (N_ROWS + 1);

  // header row
  noStroke();
  fill(view === 'header' ? 'lightgreen' : 'gainsboro');
  rect(x + idxW, y, gridW, rowH);
  fill('white');
  rect(x + idxW, y + rowH, gridW, rowH * N_ROWS);

  // dtype tints and the dtype row under the table
  if (view === 'types' || view === 'missing') {
    for (let c = 0; c < N_FIELDS; c++) {
      const tc = color(TYPE_COLORS[DTYPES[c]]);
      tc.setAlpha(view === 'types' ? 45 : 22);
      noStroke();
      fill(tc);
      rect(colX[c], y + rowH, colW[c], rowH * N_ROWS);
      fill(TYPE_COLORS[DTYPES[c]]);
      textStyle(BOLD);
      textSize(ts - 1);
      textAlign(CENTER, CENTER);
      text(DTYPES[c], colX[c] + colW[c] / 2, y + gridH + rowH / 2 + 1);
    }
  }

  // index column, once the DataFrame exists
  if (isFrame) {
    const ic = color('royalblue');
    ic.setAlpha(STEPS[step].stage === 3 ? 60 : 0);
    noStroke();
    fill('gainsboro');
    rect(x, y, idxW, gridH);
    fill(ic);
    rect(x, y + rowH, idxW, rowH * N_ROWS);
    fill('black');
    textStyle(BOLD);
    textSize(ts);
    textAlign(CENTER, CENTER);
    for (let r = 0; r < N_ROWS; r++) text(r, x + idxW / 2, y + (r + 1.5) * rowH + 1);
  }

  // grid lines
  stroke('silver');
  strokeWeight(1);
  noFill();
  const gx = isFrame ? x : x + idxW;
  rect(gx, y, x + w - gx, gridH);
  for (let r = 1; r <= N_ROWS; r++) line(gx, y + r * rowH, x + w, y + r * rowH);
  for (let c = isFrame ? 0 : 1; c < N_FIELDS; c++) line(colX[c], y, colX[c], y + gridH);

  // column names and values. Before the types are known everything is text and sits left.
  const typed = view !== 'header';
  for (let c = 0; c < N_FIELDS; c++) {
    const numeric = typed && DTYPES[c] !== 'object';
    const tx = numeric ? colX[c] + colW[c] - 8 : colX[c] + 8;
    textAlign(numeric ? RIGHT : LEFT, CENTER);
    noStroke();
    fill('black');
    textStyle(BOLD);
    textSize(ts);
    text(HEADER[c], tx, y + rowH / 2 + 1);
    textStyle(NORMAL);
    for (let r = 0; r < N_ROWS; r++) {
      const missing = BODY[r][c] === '';
      const cy = y + (r + 1.5) * rowH + 1;
      if (missing && !isFrame) {
        // the empty field: outlined while it is still blank, red once it is NaN
        const mc = color('crimson');
        mc.setAlpha(view === 'missing' ? 60 : 0);
        fill(mc);
        stroke(view === 'missing' ? 'crimson' : 'gray');
        strokeWeight(view === 'missing' ? 2 : 1);
        if (view !== 'missing') drawingContext.setLineDash([3, 3]);
        rect(colX[c] + 3, y + (r + 1) * rowH + 3, colW[c] - 6, rowH - 6, 3);
        drawingContext.setLineDash([]);
        noStroke();
      }
      fill(missing && view === 'missing' ? 'firebrick' : 'black');
      textStyle(missing && view === 'missing' ? BOLD : NORMAL);
      text(cellText(r, c, view), tx, cy);
    }
  }
  textStyle(NORMAL);
}

// What happens at this step, and what commonly goes wrong.
function drawExplainPanel(x, y, w, h, s, narrow, showErrors) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const pad = narrow ? 10 : 14;
  const ix = x + pad, iw = w - 2 * pad;
  const examples = s.examples || [];
  let cy = y + (narrow ? 7 : 10);

  // the text shrinks, down to 12px, until the explanation and the error box both fit
  const errorTitle = 'What can go wrong: ' + s.error[0];
  let ts = narrow ? 12 : 15, lh, n, m, titleLines;
  for (; ; ts--) {
    lh = narrow ? ts + 4 : ts + 6;
    textSize(ts);
    textStyle(NORMAL);
    n = countLines(s.text, iw);
    m = countLines(s.error[1], iw - 18);
    textStyle(BOLD);
    titleLines = countLines(errorTitle, iw - 18);
    const need = (narrow ? 7 : lh + 14) + n * lh + 8 + examples.length * (ts + 17) +
      (showErrors ? (m + titleLines) * lh + 26 : 0);
    if (need <= h || ts <= 12) break;
  }

  noStroke();
  textAlign(LEFT, TOP);
  textWrap(WORD);
  textLeading(lh);
  if (!narrow) {
    fill('black');
    textStyle(BOLD);
    textSize(ts + 1);
    text('What happens', ix, cy);
    cy += lh + 4;
  }
  textStyle(NORMAL);
  textSize(ts);
  fill('black');
  text(s.text, ix, cy, iw, n * lh + 4);
  cy += n * lh + (narrow ? 4 : 8);

  // example one-liners with their results (last step)
  for (const [code, result] of examples) {
    const cw = codeChip(code, ix, cy, ts);
    fill('black');
    textAlign(LEFT, CENTER);
    text('→  ' + result, ix + cw + 8, cy + (ts + 9) / 2 + 1);
    cy += ts + (narrow ? 13 : 17);
  }

  if (showErrors) {
    const eh = (m + titleLines) * lh + (narrow ? 10 : 14);
    const ey = y + h - eh - (narrow ? 7 : 12);
    fill('mistyrose');
    stroke('indianred');
    strokeWeight(1);
    rect(ix, ey, iw, eh, 8);
    noStroke();
    fill('firebrick');
    textSize(ts);
    textStyle(BOLD);
    textAlign(LEFT, TOP);
    textLeading(lh);
    text(errorTitle, ix + 9, ey + (narrow ? 5 : 7), iw - 18, titleLines * lh + 4);
    textStyle(NORMAL);
    fill('black');
    text(s.error[1], ix + 9, ey + (narrow ? 5 : 7) + titleLines * lh, iw - 18, m * lh + 4);
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

// Clicking a stage of the flowchart jumps to its first step.
function mousePressed() {
  for (const b of hitBoxes) {
    if (mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h) {
      goToStep(STEPS.findIndex(sp => sp.stage === b.stage));
      return;
    }
  }
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
