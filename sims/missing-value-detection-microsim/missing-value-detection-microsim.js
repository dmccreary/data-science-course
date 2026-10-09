// Missing Value Detective
// CANVAS_HEIGHT: 550
// Bloom L3 (Apply): students apply df.isnull() to a small DataFrame. They count the missing
// values pandas will find in one column, type the count, and check it, then switch on the
// isnull() mask and the hidden missing values to see what pandas did and did not detect.
//
// Model: pandas treats NaN and None as missing, so isnull() is True for them. An empty string
// and a placeholder code such as -999 are ordinary values, so isnull() is False for them even
// though no real value is there. Each scenario removes cells from the same clean 8 x 5 table
// at positions chosen by a seeded generator. Every count and percentage is computed from the grid.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 470;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const COLS = ['name', 'age', 'city', 'score', 'hours'];
const IS_TEXT = [true, false, true, false, false];
const BASE = [
  ['Alice', 25, 'Austin', 85, 3.5],
  ['Bob', 31, 'Boston', 92, 1.0],
  ['Cara', 22, 'Denver', 78, 4.5],
  ['Dev', 28, 'Miami', 64, 2.0],
  ['Eve', 35, 'Omaha', 88, 5.5],
  ['Finn', 19, 'Tampa', 71, 0.5],
  ['Gia', 27, 'Reno', 95, 6.0],
  ['Hank', 40, 'Dallas', 59, 2.5]
];
const CODE = -999;                // placeholder code used by the "hidden" scenario
// missing: cells set to NaN (None in text columns when none is true). hidden: cells set to ""
// (text columns) or -999 (numeric columns). column: put every missing cell in this column.
const SCENARIOS = [
  { name: '1. NaN only', seed: 11, missing: 6 },
  { name: '2. NaN and None', seed: 23, missing: 8, none: true },
  { name: '3. Hidden missing', seed: 37, missing: 3, hidden: 6 },
  { name: '4. All in one column', seed: 5, missing: 5, column: 3 },
  { name: '5. Sparse data', seed: 41, missing: 22 }
];
const TINTS = { present: [60, 179, 113, 50], nan: [220, 20, 60, 100], none: [255, 140, 0, 120], hidden: [255, 215, 0, 170] };

let scenarioSelect, answerInput, checkButton, isnullCheckbox, hiddenCheckbox;
let grid = [];                    // grid[row][col] = { v, kind }: ok, nan, none, empty, or code
let scenarioIndex = 0;
let quizCol = 0;                  // the column the question asks about
let feedback = '', feedbackColor = 'dimgray';
let solved = {};                  // "scenario,column" for each question answered correctly
let headerBoxes = [];             // clickable column headers

const isNull = cell => cell.kind === 'nan' || cell.kind === 'none';
const isHidden = cell => cell.kind === 'empty' || cell.kind === 'code';
const countCol = (c, test) => grid.filter(row => test(row[c])).length;
const countAll = test => COLS.reduce((sum, name, c) => sum + countCol(c, test), 0);

// Build the grid for a scenario with a small linear congruential generator, so the same
// scenario always removes the same cells.
function loadScenario(k) {
  scenarioIndex = k;
  const sc = SCENARIOS[k];
  let seed = sc.seed;
  const pick = n => { seed = (seed * 1664525 + 1013904223) % 4294967296; return Math.floor(seed / 4294967296 * n); };
  grid = BASE.map(row => row.map(v => ({ v, kind: 'ok' })));
  const place = (count, kindOf) => {
    while (count > 0) {
      const r = pick(BASE.length), c = sc.column !== undefined ? sc.column : pick(COLS.length);
      if (grid[r][c].kind !== 'ok') continue;
      grid[r][c].kind = kindOf(c);
      count--;
    }
  };
  place(sc.missing, c => sc.none && IS_TEXT[c] ? 'none' : 'nan');
  place(sc.hidden || 0, c => IS_TEXT[c] ? 'empty' : 'code');
  // start the quiz on the column with the most to find
  let best = -1;
  COLS.forEach((name, c) => {
    const weight = countCol(c, isHidden) * 10 + countCol(c, isNull);
    if (weight > best) { best = weight; quizCol = c; }
  });
  clearQuiz();
}

function clearQuiz() {
  feedback = '';
  if (answerInput) answerInput.value('');
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  scenarioSelect = createSelect();
  scenarioSelect.parent(mainElement);
  scenarioSelect.position(10, drawHeight + 10);
  SCENARIOS.forEach(sc => scenarioSelect.option(sc.name));
  scenarioSelect.style('font-size', '15px');
  // a new scenario is a new challenge: the answers are hidden again
  scenarioSelect.changed(() => {
    loadScenario(SCENARIOS.findIndex(sc => sc.name === scenarioSelect.value()));
    isnullCheckbox.checked(false);
    hiddenCheckbox.checked(false);
  });

  answerInput = createInput('', 'number');
  answerInput.parent(mainElement);
  answerInput.position(196, drawHeight + 10);
  answerInput.size(58);
  answerInput.attribute('placeholder', 'count');
  answerInput.elt.addEventListener('keydown', e => { if (e.key === 'Enter') checkAnswer(); });

  checkButton = createButton('Check');
  checkButton.parent(mainElement);
  checkButton.position(270, drawHeight + 10);
  checkButton.mousePressed(checkAnswer);

  // scenario 1 opens with the mask on, as a worked example
  isnullCheckbox = createCheckbox(' Show isnull()', true);
  isnullCheckbox.parent(mainElement);
  isnullCheckbox.position(10, drawHeight + 46);
  isnullCheckbox.style('font-size', '16px');

  hiddenCheckbox = createCheckbox(' Find hidden missing', false);
  hiddenCheckbox.parent(mainElement);
  hiddenCheckbox.position(150, drawHeight + 46);
  hiddenCheckbox.style('font-size', '16px');

  loadScenario(0);

  describe('A table of 8 rows and 5 columns with some values missing. A menu chooses one of five scenarios. ' +
    'The student types how many missing values isnull() will count in one column and presses Check for feedback. ' +
    'One checkbox colors the cells that isnull() reports and shows the count for each column. A second checkbox ' +
    'marks hidden missing values, which are empty strings and the code -999.', LABEL);
}

// Compare the typed count with what isnull().sum() gives for the quiz column
function checkAnswer() {
  const typed = answerInput.value().trim(), guess = Number(typed);
  const nulls = countCol(quizCol, isNull), hid = countCol(quizCol, isHidden), name = COLS[quizCol];
  if (typed === '' || !Number.isInteger(guess)) {
    feedback = 'Type a whole number in the box, then press Check.';
    feedbackColor = 'dimgray';
  } else if (guess === nulls) {
    solved[scenarioIndex + ',' + quizCol] = true;
    feedback = 'Correct! isnull() counts ' + nulls + ' in ' + name + '.' +
      (hid > 0 ? ' But ' + hid + ' more are hidden as "" or ' + CODE + ', and isnull() cannot see them.' : '');
    feedbackColor = 'darkgreen';
  } else if (hid > 0 && guess === nulls + hid) {
    feedback = 'Not what pandas reports. You also counted the hidden ones. To pandas, "" and ' + CODE +
      ' are real values, so isnull() is False for them.';
    feedbackColor = 'firebrick';
  } else {
    feedback = 'Not yet. Hint: go down the ' + name + ' column and count only the cells that show NaN or None.';
    feedbackColor = 'firebrick';
  }
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
  const showNull = isnullCheckbox.checked(), showHidden = hiddenCheckbox.checked();
  textWrap(WORD);

  noStroke();
  fill('black');
  textStyle(NORMAL);
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Missing Value Detective', canvasWidth / 2, 8);

  // table on the left and panels on the right, or stacked when narrow
  const top = narrow ? 32 : 44, fullW = canvasWidth - 2 * margin;
  const gridW = narrow ? fullW : fullW * 0.57;
  const gridH = narrow ? 190 : drawHeight - top - 36;
  drawGrid(margin, top, gridW, gridH, narrow, showNull, showHidden);
  drawLegend(margin, top + gridH + (narrow ? 11 : 17), narrow);
  if (narrow) drawPanels(margin, top + gridH + 24, fullW, drawHeight - top - gridH - 32, narrow, showNull, showHidden);
  else drawPanels(margin + gridW + 10, top, fullW - gridW - 10, drawHeight - top - 8, narrow, showNull, showHidden);
  cursor(headerBoxes.some(mouseOver) ? HAND : ARROW);
}

// How pandas prints a cell. A numeric column that holds NaN is stored as floats.
function cellText(cell, c, showHidden) {
  if (cell.kind === 'nan') return 'NaN';
  if (cell.kind === 'none') return 'None';
  if (cell.kind === 'empty') return showHidden ? '""' : '';
  const v = cell.kind === 'code' ? CODE : cell.v;
  if (IS_TEXT[c]) return v;
  return c === 4 || grid.some(row => row[c].kind === 'nan') ? v.toFixed(1) : String(v);
}

// The DataFrame, with a last row for df.isnull().sum()
function drawGrid(x, y, w, h, narrow, showNull, showHidden) {
  const ts = narrow ? 12 : 15, idxW = narrow ? 20 : 30;
  const nRows = BASE.length, rowH = h / (nRows + 2), colW = (w - idxW) / COLS.length;
  headerBoxes = [];
  noStroke();
  fill('white');
  rect(x, y, w, h);
  fill('gainsboro');
  rect(x, y, w, rowH);
  fill('whitesmoke');
  rect(x, y + (nRows + 1) * rowH, w, rowH);
  textSize(ts);

  for (let c = 0; c < COLS.length; c++) {
    const cx = x + idxW + c * colW, tx = IS_TEXT[c] ? cx + 7 : cx + colW - 7;
    // header: the quiz column is blue. Click a header to ask about that column.
    noStroke();
    if (c === quizCol) {
      fill('royalblue');
      rect(cx, y, colW, rowH);
    }
    fill(c === quizCol ? 'white' : 'black');
    textStyle(BOLD);
    textAlign(IS_TEXT[c] ? LEFT : RIGHT, CENTER);
    text(COLS[c], tx, y + rowH / 2 + 1);
    headerBoxes.push({ x: cx, y, w: colW, h: rowH, col: c });

    for (let r = 0; r < nRows; r++) {
      const cell = grid[r][c], cy = y + (r + 1) * rowH;
      let tint = null;
      if (showHidden && isHidden(cell)) tint = TINTS.hidden;
      else if (showNull) tint = TINTS[isNull(cell) ? cell.kind : 'present'];
      if (tint) {
        fill(tint[0], tint[1], tint[2], tint[3]);
        rect(cx, cy, colW, rowH);
      }
      fill(isNull(cell) ? 'firebrick' : 'black');
      textStyle(isNull(cell) ? BOLD : NORMAL);
      text(cellText(cell, c, showHidden), tx, cy + rowH / 2 + 1);
    }

    // count row: a bar and the number of missing values pandas finds in this column
    const nulls = countCol(c, isNull), hid = countCol(c, isHidden), sy = y + (nRows + 1) * rowH;
    textAlign(CENTER, CENTER);
    textStyle(BOLD);
    if (showNull) {
      fill(220, 20, 60, 100);
      rect(cx + 3, sy + 3, (colW - 6) * nulls / nRows, rowH - 6, 3);
    }
    fill(showNull ? 'black' : 'gray');
    text((showNull ? nulls : '?') + (showHidden && hid > 0 ? '  +' + hid : ''), cx + colW / 2, sy + rowH / 2 + 1);
  }

  // index labels and grid lines
  textStyle(BOLD);
  for (let r = 0; r <= nRows; r++) {
    noStroke();
    fill('dimgray');
    textAlign(CENTER, CENTER);
    text(r < nRows ? r : 'Σ', x + idxW / 2, y + (r + 1.5) * rowH + 1);
    stroke('gainsboro');
    strokeWeight(1);
    line(x, y + (r + 1) * rowH, x + w, y + (r + 1) * rowH);
  }
  for (let c = 0; c < COLS.length; c++) line(x + idxW + c * colW, y, x + idxW + c * colW, y + h);
  noFill();
  stroke('silver');
  rect(x, y, w, h);
  textStyle(NORMAL);
}

// Color key for the cell tints
function drawLegend(x, y, narrow) {
  const items = [['present', 'present'], ['nan', 'NaN'], ['none', 'None'], ['hidden', 'hidden: "" or ' + CODE]];
  const s = narrow ? 11 : 13;
  textSize(narrow ? 11 : 13);
  textStyle(NORMAL);
  textAlign(LEFT, CENTER);
  for (const [key, label] of items) {
    const t = TINTS[key];
    stroke('silver');
    strokeWeight(1);
    fill(t[0], t[1], t[2], t[3] + 60);
    rect(x, y - s / 2, s, s, 2);
    noStroke();
    fill('black');
    text(label, x + s + 4, y + 1);
    x += s + 4 + textWidth(label) + (narrow ? 12 : 16);
  }
}

// Right side: what the detection code returns, and the quiz
function drawPanels(x, y, w, h, narrow, showNull, showHidden) {
  const ts = narrow ? 12 : 16, lh = ts + (narrow ? 4 : 10), pad = narrow ? 8 : 12;
  const nulls = countAll(isNull), cells = BASE.length * COLS.length;
  const lines = [
    ['df.isnull().sum().sum()', showNull ? String(nulls) : '?'],
    ['... / df.size * 100', showNull ? (100 * nulls / cells).toFixed(1) + '%' : '?'],
    ['(df == "").sum().sum()', showHidden ? String(countAll(cell => cell.kind === 'empty')) : '?'],
    ['(df == ' + CODE + ').sum().sum()', showHidden ? String(countAll(cell => cell.kind === 'code')) : '?']
  ];
  const codeH = pad + lh * (lines.length + 1) + (narrow ? 2 : 8);
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, codeH, 10);
  rect(x, y + codeH + 6, w, h - codeH - 6, 10);

  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text('Detection code', x + pad, y + pad - 2);
  text('Quiz', x + pad, y + codeH + 6 + pad - 2);
  textStyle(NORMAL);
  textSize(ts);
  fill('dimgray');
  textAlign(RIGHT, TOP);
  text('solved ' + Object.keys(solved).length + ' of ' + SCENARIOS.length * COLS.length, x + w - pad, y + codeH + 6 + pad - 1);
  lines.forEach(([code, value], i) => {
    const ly = y + pad + lh * (i + 1);
    fill(i < 2 ? 'firebrick' : 'darkgoldenrod');
    textAlign(LEFT, TOP);
    textStyle(NORMAL);
    text(code, x + pad, ly);
    fill('black');
    textAlign(RIGHT, TOP);
    textStyle(BOLD);
    text(value, x + w - pad, ly);
  });

  // question for the selected column, then the feedback on the last answer
  const qy = y + codeH + 6 + pad + lh;
  const qh = narrow ? 2 * (ts + 3) : 3 * (ts + 5);
  textStyle(NORMAL);
  textAlign(LEFT, TOP);
  textLeading(narrow ? ts + 3 : ts + 5);
  fill('black');
  text('How many missing values will df["' + COLS[quizCol] + '"].isnull().sum() count? Click a column name to change column.',
    x + pad, qy, w - 2 * pad, qh + 4);
  fill(feedback ? feedbackColor : 'dimgray');
  textStyle(feedback ? BOLD : ITALIC);
  text(feedback || 'Count first, type your answer below, and press Check. Then use the checkboxes to see why.',
    x + pad, qy + qh + (narrow ? 4 : 8), w - 2 * pad, y + h - qy - qh - 8);
  textStyle(NORMAL);
}

// Clicking a column name moves the question to that column.
const mouseOver = b => mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h;
function mousePressed() {
  const box = headerBoxes.find(mouseOver);
  if (box && box.col !== quizCol) {
    quizCol = box.col;
    clearQuiz();
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
