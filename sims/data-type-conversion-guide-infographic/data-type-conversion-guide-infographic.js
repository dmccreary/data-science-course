// Data Type Conversion Guide
// CANVAS_HEIGHT: 630
// Bloom L1 (Remember): a reference card of nine pandas type conversions in four color-coded
// groups. Students click a method, or step with Next and Previous, to recall what it does,
// when to use it, and what to watch for. "Quiz me" hides the result so they can recall it first.
//
// Model: every example column is really converted here, following the pandas rules.
//   pd.to_numeric      whole numbers give int64, anything else float64. With errors="coerce"
//                      text that is not a number becomes NaN (a float).
//   astype(float)      always float64
//   pd.to_datetime     a date written with slashes is read month/day/year
//   astype("category") the categories are the sorted unique values, and each row stores the
//                      position (code) of its value in that list
//   pd.Categorical     with categories=[...] the codes follow the order you give
//   astype(str), "{:.2f}".format   text, as Python writes the number
// Text columns are labeled object, as pandas 2 reports them.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 585;
let controlHeight = 45;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const GROUPS = [
  { title: 'Converting TO Numeric', color: 'royalblue', ink: 'mediumblue' },
  { title: 'Converting TO Datetime', color: 'seagreen', ink: 'darkgreen' },
  { title: 'Converting TO Categorical', color: 'mediumpurple', ink: 'indigo' },
  { title: 'Converting TO String', color: 'darkorange', ink: 'saddlebrown' }
];

// ---- the conversions. Each returns { dtype, values (as printed), extra: [code, result] } ----
const quote = s => '"' + s + '"';
const floatText = v => Number.isNaN(v) ? 'NaN' : Number.isInteger(v) ? v.toFixed(1) : String(v);
const parseNumber = s => /^-?\d+(\.\d+)?$/.test(s) ? Number(s) : NaN;
const total = a => a.reduce((sum, v) => sum + v, 0);

function toNumeric(m) {
  const nums = m.input.map(parseNumber), whole = nums.every(Number.isInteger);
  const missing = nums.filter(Number.isNaN).length;
  return { dtype: whole ? 'int64' : 'float64', values: nums.map(v => whole ? String(v) : floatText(v)),
    extra: missing > 0 ? ['col.isnull().sum()', String(missing)] : ['col.sum()', String(total(nums))] };
}
function toFloat(m) {
  const nums = m.input.map(Number);
  return { dtype: 'float64', values: nums.map(floatText), extra: ['col.mean()', (total(nums) / nums.length).toFixed(2)] };
}
function toDatetime(m) {
  const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
    'November', 'December'];
  const DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const dates = m.input.map(s => {
    const p = s.split(/[-\/]/).map(Number);
    return s.includes('/') ? new Date(Date.UTC(p[2], p[0] - 1, p[1])) : new Date(Date.UTC(p[0], p[1] - 1, p[2]));
  });
  return { dtype: 'datetime64[ns]', values: dates.map(d => d.toISOString().slice(0, 10)),
    extra: m.input[0].includes('/') ? ['col.dt.month_name()', dates.map(d => MONTHS[d.getUTCMonth()]).join(', ')]
      : ['col.dt.day_name()', dates.map(d => DAYS[d.getUTCDay()]).join(', ')] };
}
function toCategory(m) {
  const cats = m.categories || Array.from(new Set(m.input)).sort();
  const codes = m.input.map(v => cats.indexOf(v));
  const lowest = cats[Math.min(...codes)];
  return { dtype: 'category', values: m.input.map((v, i) => v + '   (code ' + codes[i] + ')'),
    extra: m.categories ? ['col.min()', quote(lowest)] : ['col.cat.categories', cats.join(', ')] };
}
function toText(m) {
  const texts = m.input.map(v => m.decimals === undefined ? floatText(v) : v.toFixed(m.decimals));
  return { dtype: 'object', values: texts.map(quote),
    extra: m.decimals === undefined ? ['col.str.len()', texts.map(s => s.length).join(', ')]
      : ['"$" + col', texts.map(s => quote('$' + s)).join(', ')] };
}

const METHODS = [
  { group: 0, code: 'pd.to_numeric(col)', label: 'Basic conversion', input: ['42', '7', '19'], run: toNumeric,
    use: 'The first choice for numbers stored as text. pandas picks int64 or float64 for you.',
    watch: 'One value that is not a number, such as "seven", stops it with a ValueError.' },
  { group: 0, code: 'pd.to_numeric(col, errors="coerce")', label: 'Invalid → NaN', input: ['42', 'seven', '3.5'], run: toNumeric,
    use: 'For messy columns. Text that cannot be read as a number becomes NaN instead of raising an error.',
    watch: 'NaN makes the column float64. Whole numbers with NaN need astype("Int64"), with a capital I.' },
  { group: 0, code: 'col.astype(float)', label: "When you're sure it's clean", input: ['42', '7', '19'], run: toFloat,
    use: 'When you are sure every value is clean and you want one specific type.',
    watch: 'Bad text raises a ValueError. astype(int) also fails if the column holds NaN.' },
  { group: 1, code: 'pd.to_datetime(col)', label: 'Smart parsing', input: ['03/15/2024', '12/01/2024'], run: toDatetime,
    use: 'pandas works out the date format for you. Real dates sort correctly and can be subtracted.',
    watch: '03/04/2024 is read month first, as March 4. Parsing can be slow on large datasets.' },
  { group: 1, code: 'pd.to_datetime(col, format="%Y-%m-%d")', label: 'Specific format', input: ['2024-03-15', '2024-12-01'],
    run: toDatetime,
    use: 'When you know the format. Naming it is faster on large datasets and leaves nothing to guess.',
    watch: 'A value that does not match the format raises an error unless you add errors="coerce".' },
  { group: 2, code: 'col.astype("category")', label: 'Basic categorical', input: ['B', 'A', 'B', 'C'], run: toCategory,
    use: 'For text with few distinct values. Each value is stored once and every row keeps a small code.',
    watch: 'It saves memory but changes behavior. A value outside the categories cannot be assigned.' },
  { group: 2, code: 'pd.Categorical(col, categories=[...], ordered=True)', label: 'Ordered', input: ['B', 'A', 'C', 'B'],
    categories: ['F', 'D', 'C', 'B', 'A'], run: toCategory,
    use: 'When categories have a rank, like grades or sizes. Sorting and comparisons follow your order.',
    watch: 'A value that is missing from the categories list silently becomes NaN.' },
  { group: 3, code: 'col.astype(str)', label: 'Simple conversion', input: [3.14159, 42.0], run: toText,
    use: 'For numbers that are really labels, such as ZIP codes and IDs, or before joining text.',
    watch: 'Math stops working: "2" + "2" is "22". Convert back with pd.to_numeric before calculating.' },
  { group: 3, code: 'col.map("{:.2f}".format)', label: 'With formatting', input: [3.14159, 42.0], decimals: 2, run: toText,
    use: 'To control how numbers look in a report, here with exactly two decimal places.',
    watch: 'The result is text for display only. Keep the numeric column for calculations.' }
];
const beforeText = v => typeof v === 'string' ? quote(v) : floatText(v);

let selected = 0;
let prevButton, nextButton, quizCheckbox;
let rowBoxes = [];                // clickable method rows: { x, y, w, h, index }

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  prevButton = createButton('Previous');
  prevButton.parent(mainElement);
  prevButton.position(10, drawHeight + 10);
  prevButton.mousePressed(() => { selected = (selected + METHODS.length - 1) % METHODS.length; });

  nextButton = createButton('Next');
  nextButton.parent(mainElement);
  nextButton.position(86, drawHeight + 10);
  nextButton.mousePressed(() => { selected = (selected + 1) % METHODS.length; });

  quizCheckbox = createCheckbox(' Quiz me (hide the result)', false);
  quizCheckbox.parent(mainElement);
  quizCheckbox.position(140, drawHeight + 11);
  quizCheckbox.style('font-size', '16px');

  describe('A reference card with four color-coded groups of pandas type conversions: to numeric, to datetime, ' +
    'to categorical, and to string. Clicking one of the nine methods shows when to use it, what to watch out for, ' +
    'and a small example column before and after the conversion. A checkbox hides the result for self-testing.', LABEL);
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
  textWrap(WORD);
  noStroke();
  fill('black');
  textStyle(NORMAL);
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Data Type Conversion Guide', canvasWidth / 2, 8);

  // the four cards: a 2 x 2 grid when wide, one column when narrow
  const cols = narrow ? 1 : 2, gap = narrow ? 5 : 10, fullW = canvasWidth - 2 * margin;
  const cardW = (fullW - (cols - 1) * gap) / cols;
  const headH = narrow ? 20 : 28, rowH = narrow ? 19 : 38;
  let y = narrow ? 32 : 44;
  rowBoxes = [];
  for (let g = 0; g < GROUPS.length; g += cols) {
    // cards that share a grid row share the height of the taller one
    const rows = max(GROUPS.slice(g, g + cols).map((grp, k) => METHODS.filter(m => m.group === g + k).length));
    const cardH = headH + rows * rowH + 6;
    for (let k = 0; k < cols; k++) drawCard(g + k, margin + k * (cardW + gap), y, cardW, cardH, headH, rowH, narrow);
    y += cardH + gap;
  }
  cursor(rowBoxes.some(mouseOver) ? HAND : ARROW);
  drawDetail(margin, y, fullW, drawHeight - 8 - y, narrow);
}

// One group: a colored header with a before → after sample, then its methods
function drawCard(g, x, y, w, h, headH, rowH, narrow) {
  const grp = GROUPS[g], mine = METHODS.filter(m => m.group === g);
  fill('white');
  stroke(grp.color);
  strokeWeight(1.5);
  rect(x, y, w, h, 8);
  noStroke();
  fill(grp.color);
  rect(x, y, w, headH, 8, 8, 0, 0);
  fill('white');
  textStyle(BOLD);
  textSize(narrow ? 12 : 15);
  textAlign(LEFT, CENTER);
  text(grp.title, x + 9, y + headH / 2 + 1);
  // sample taken from the group's first example
  const sample = mine[0].run(mine[0]);
  textStyle(NORMAL);
  textSize(narrow ? 11 : 13);
  textAlign(RIGHT, CENTER);
  text(beforeText(mine[0].input[0]) + '  →  ' + sample.values[0].replace('   (', ' ('), x + w - 9, y + headH / 2 + 1);

  mine.forEach((m, k) => {
    const i = METHODS.indexOf(m), ry = y + headH + 3 + k * rowH;
    const box = { x: x + 3, y: ry, w: w - 6, h: rowH - 1, index: i };
    if (i === selected || mouseOver(box)) {
      const c = color(grp.color);
      c.setAlpha(i === selected ? 70 : 25);
      fill(c);
      rect(box.x, box.y, box.w, box.h, 5);
    }
    fill(grp.ink);
    textStyle(BOLD);
    textSize(narrow ? 11 : 14);
    textAlign(LEFT, CENTER);
    text(m.code, x + 10, ry + (narrow ? rowH / 2 : 12));
    if (!narrow) {
      fill('dimgray');
      textStyle(NORMAL);
      textSize(12);
      text(m.label, x + 10, ry + 28);
    }
    rowBoxes.push(box);
  });
  textStyle(NORMAL);
}

// The selected method: when to use it, what to watch for, and a worked example
function drawDetail(x, y, w, h, narrow) {
  const m = METHODS[selected], grp = GROUPS[m.group];
  fill('white');
  stroke(grp.color);
  strokeWeight(2);
  rect(x, y, w, h, 10);
  const pad = narrow ? 8 : 14, ix = x + pad, iw = w - 2 * pad;
  const ts = narrow ? 12 : 15, lh = ts + (narrow ? 3 : 6);
  const textW = narrow ? iw : iw * 0.56;
  const say = (str, sx, sy, sw, lines, col, style) => {
    noStroke();
    fill(col);
    textStyle(style);
    textSize(ts);
    textLeading(lh);
    textAlign(LEFT, TOP);
    text(str, sx, sy, sw, lines * lh + 3);
  };

  let cy = y + pad;
  textSize(ts + 1);
  noStroke();
  fill(grp.ink);
  textStyle(BOLD);
  textAlign(LEFT, TOP);
  text(m.code, ix, cy);
  cy += lh + 2;
  say(m.label + '   (method ' + (selected + 1) + ' of ' + METHODS.length + ')', ix, cy, textW, 1, 'dimgray', NORMAL);
  cy += lh + (narrow ? 4 : 10);
  const lines = 2;
  say('When to use it', ix, cy, textW, 1, 'black', BOLD);
  say(m.use, ix, cy + lh, textW, lines, 'black', NORMAL);
  cy += (lines + 1) * lh + (narrow ? 3 : 8);
  say('Watch out', ix, cy, textW, 1, 'firebrick', BOLD);
  say(m.watch, ix, cy + lh, textW, lines, 'black', NORMAL);
  cy += (lines + 1) * lh + (narrow ? 5 : 8);

  // example: beside the text when wide, under it when narrow
  const ex = narrow ? ix : ix + textW + 18, ew = narrow ? iw : iw - textW - 18;
  const ey = narrow ? cy : y + pad;
  const out = m.run(m), quiz = quizCheckbox.checked();
  const rowH = narrow ? 15 : 23, boxW = (ew - 26) / 2, boxH = rowH * 5 + 4;
  const sides = [['Before', typeof m.input[0] === 'string' ? 'object' : 'float64', m.input.map(beforeText)],
    ['After', quiz ? '?' : out.dtype, quiz ? [] : out.values]];
  sides.forEach(([title, dtype, values], k) => {
    const bx = ex + k * (boxW + 26);
    fill(k === 0 ? 'whitesmoke' : 'white');
    stroke(k === 0 ? 'silver' : grp.color);
    strokeWeight(k === 0 ? 1 : 2);
    rect(bx, ey, boxW, boxH, 6);
    noStroke();
    textSize(ts - 1);
    textAlign(LEFT, CENTER);
    textStyle(BOLD);
    fill('black');
    text(title, bx + 8, ey + rowH / 2 + 3);
    textAlign(RIGHT, CENTER);
    textStyle(NORMAL);
    fill('dimgray');
    text(dtype, bx + boxW - 8, ey + rowH / 2 + 3);
    textAlign(LEFT, CENTER);
    fill(k === 0 ? 'black' : grp.ink);
    textSize(ts);
    values.forEach((v, r) => text(v, bx + 8, ey + (r + 1.5) * rowH + 3));
    if (k === 1 && quiz) say('What will the values and the dtype be? Untick Quiz me to check.', bx + 8, ey + rowH + 5,
      boxW - 16, 4, 'dimgray', ITALIC);
  });
  noStroke();
  fill(grp.color);
  textAlign(CENTER, CENTER);
  textStyle(BOLD);
  textSize(ts + 4);
  text('→', ex + boxW + 13, ey + boxH / 2);
  say('Now you can:  ' + out.extra[0] + '  →  ' + (quiz ? '?' : out.extra[1]), ex, ey + boxH + (narrow ? 4 : 8), ew, 2,
    'black', NORMAL);
  textStyle(NORMAL);
}

// Clicking a method selects it.
function mouseOver(b) {
  return mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h;
}
function mousePressed() {
  const box = rowBoxes.find(mouseOver);
  if (box) selected = box.index;
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
