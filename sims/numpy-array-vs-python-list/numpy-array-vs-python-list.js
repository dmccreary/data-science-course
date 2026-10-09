// NumPy Array vs Python List
// CANVAS_HEIGHT: 614
// Bloom L2 (Understand): students step through doubling five numbers and compare how a Python
// list and a NumPy array store the same numbers and how each one is processed.
//
// Model (CPython on a 64-bit machine, the numbers the chapter's sys.getsizeof code prints):
//   list of n ints   56 bytes for the list + 8 bytes per pointer + 28 bytes per int object
//   int64 array      8 bytes per value in one block (arr.nbytes)
// [x * 2 for x in nums] makes one pass through the interpreter per element. arr * 2 is one
// call whose loop runs in compiled C. The addresses on screen are made up for illustration.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 534;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let sliderLeftMargin = 200;
let defaultTextSize = 16;

const VALUES = [1, 2, 3, 4, 5];
const N = VALUES.length;
// where each int object is drawn in the heap region (fractions of the region) and its address
const OBJ_POS = [[0.56, 0.06], [0.02, 0.8], [0.98, 0.55], [0.24, 0], [0.6, 1]];
const OBJ_ADDR = [5216, 2840, 7392, 3464, 6128];
const ARRAY_ADDR = 1000;                          // the array's values sit at 1000, 1008, 1016, ...
const LIST_BYTES = 56, POINTER_BYTES = 8, INT_BYTES = 28, VALUE_BYTES = 8;
const SIZES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000];

let step = 0;                                     // 0 = storage only, 1..N = list element being doubled
let prevButton, nextButton, resetButton, addressCheckbox, sizeSlider;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  prevButton = createButton('Previous');
  prevButton.parent(mainElement);
  prevButton.position(10, drawHeight + 10);
  prevButton.mousePressed(() => { step = max(0, step - 1); });

  nextButton = createButton('Next');
  nextButton.parent(mainElement);
  nextButton.position(86, drawHeight + 10);
  nextButton.mousePressed(() => { step = min(N, step + 1); });

  resetButton = createButton('Reset');
  resetButton.parent(mainElement);
  resetButton.position(138, drawHeight + 10);
  resetButton.mousePressed(() => { step = 0; });

  addressCheckbox = createCheckbox(' Show addresses', true);
  addressCheckbox.parent(mainElement);
  addressCheckbox.position(200, drawHeight + 11);
  addressCheckbox.style('font-size', '16px');

  sizeSlider = createSlider(0, SIZES.length - 1, SIZES.length - 1, 1);
  sizeSlider.parent(mainElement);
  sizeSlider.position(sliderLeftMargin, drawHeight + 45);
  sizeSlider.size(canvasWidth - sliderLeftMargin - margin);

  describe('A side by side comparison of a Python list and a NumPy array holding the numbers 1 to 5. ' +
    'The list is drawn as five pointers to separate int objects scattered through memory. The array is drawn as ' +
    'one block of five values side by side. Stepping forward doubles the numbers, one element per step for the list ' +
    'and the whole block at once for the array. A slider sets an array size for a memory comparison.', LABEL);
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
  textStyle(NORMAL);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('NumPy Array vs Python List', canvasWidth / 2, 8);
  textSize(narrow ? 12 : 15);
  fill('dimgray');
  text(step === 0 ? 'The same five numbers, stored two ways. Press Next to double them.'
    : 'Doubling every number: step ' + step + ' of ' + N, canvasWidth / 2, narrow ? 34 : 40);

  prevButton.elt.disabled = step === 0;
  nextButton.elt.disabled = step === N;

  const gap = narrow ? 6 : 12, py = 62, ph = 326;
  const pw = (canvasWidth - 2 * margin - gap) / 2;
  drawListPanel(margin, py, pw, ph, narrow);
  drawArrayPanel(margin + pw + gap, py, pw, ph, narrow);
  drawScalePanel(margin, py + ph + 8, canvasWidth - 2 * margin, drawHeight - py - ph - 16, narrow);

  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(narrow ? 14 : defaultTextSize);
  text('Array size n: ' + SIZES[sizeSlider.value()].toLocaleString('en-US'), 10, drawHeight + 56);
}

// Shared panel frame: title, the code that made the object, and the gray memory region
function panelFrame(x, y, w, h, title, code, ink, narrow) {
  fill('white');
  stroke(ink);
  strokeWeight(1.5);
  rect(x, y, w, h, 10);
  noStroke();
  fill(ink);
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(narrow ? 13 : 16);
  text(title, x + 8, y + 7);
  textStyle(NORMAL);
  fill('black');
  textSize(narrow ? 11 : 14);
  text(code, x + 8, y + 28);
  fill('gainsboro');
  rect(x + 6, y + 102, w - 12, 128, 6);
}

// One memory cell with a value in it
function cell(x, y, w, h, label, bg, edge, ts) {
  fill(bg);
  stroke(edge);
  strokeWeight(edge === 'silver' ? 1 : 2);
  rect(x, y, w, h, 3);
  noStroke();
  fill('black');
  textAlign(CENTER, CENTER);
  textSize(ts);
  text(label, x + w / 2, y + h / 2 + 1);
}

function note(str, x, y, w, lines, ts, ink) {
  noStroke();
  fill(ink);
  textAlign(LEFT, TOP);
  textSize(ts);
  textLeading(ts + 3);
  text(str, x, y, w, lines * (ts + 3) + 3);
}

// State colors: the element being worked on now, elements already done, untouched
function stateFill(done, current) { return current ? 'gold' : done ? 'palegreen' : 'white'; }
function stateEdge(done, current) { return current ? 'darkorange' : done ? 'seagreen' : 'silver'; }

// Left: the list object is a row of pointers, and every number is its own object elsewhere
function drawListPanel(x, y, w, h, narrow) {
  panelFrame(x, y, w, h, 'Python list', 'nums = [1, 2, 3, 4, 5]', 'darkorange', narrow);
  const ts = narrow ? 11 : 13, showAddr = addressCheckbox.checked();
  const sw = min(54, (w - 16) / N - 4), rowW = N * (sw + 4) - 4, sx = x + (w - rowW) / 2;
  note('list object: ' + N + ' pointers, 8 bytes each', x + 8, y + 48, w - 12, 1, ts, 'dimgray');

  // heap region and the position of each int object in it
  const hx = x + 10, hy = y + 108, hw = w - 20, hh = 104, ow = sw + 4, oh = 30;
  const objs = OBJ_POS.map(p => [hx + p[0] * (hw - ow), hy + p[1] * (hh - oh - 14)]);
  // pointers first, so the object boxes cover the line ends
  for (let i = 0; i < N; i++) {
    const current = i === step - 1, done = i < step - 1;
    stroke(current ? 'darkorange' : done ? 'seagreen' : 'gray');
    strokeWeight(current ? 3 : 1.2);
    line(sx + i * (sw + 4) + sw / 2, y + 92, objs[i][0] + ow / 2, objs[i][1]);
  }
  for (let i = 0; i < N; i++) {
    const current = i === step - 1, done = i < step - 1;
    cell(sx + i * (sw + 4), y + 64, sw, 28, showAddr ? OBJ_ADDR[i] : '●', stateFill(done, current),
      stateEdge(done, current), showAddr ? ts : 11);
    cell(objs[i][0], objs[i][1], ow, oh, 'int ' + VALUES[i], stateFill(done, current), stateEdge(done, current), ts + 1);
    if (showAddr) {
      noStroke();
      fill('dimgray');
      textSize(11);
      textAlign(CENTER, TOP);
      text('@' + OBJ_ADDR[i], objs[i][0] + ow / 2, objs[i][1] + oh + 1);
    }
  }
  note('each number is a separate object', x + 10, y + 214, w - 16, 1, ts, 'dimgray');

  note('[x * 2 for x in nums]', x + 8, y + 236, w - 12, 1, ts + 1, 'black');
  for (let i = 0; i < N; i++) {
    cell(sx + i * (sw + 4), y + 254, sw, 24, i < step ? VALUES[i] * 2 : '', i < step ? 'palegreen' : 'whitesmoke',
      i < step ? 'seagreen' : 'silver', ts + 1);
  }
  const say = step === 0 ? 'The list stores pointers, not numbers. Each pointer leads to an object somewhere else in memory.'
    : 'Pass ' + step + ' of ' + N + ': follow the pointer to int ' + VALUES[step - 1] + ', check its type, multiply, ' +
      'then store a pointer to the result.';
  note(say, x + 8, y + 284, w - 14, 3, ts, 'black');
}

// Right: a small header, then the numbers themselves side by side in one block
function drawArrayPanel(x, y, w, h, narrow) {
  panelFrame(x, y, w, h, 'NumPy array', 'arr = np.array([1, 2, 3, 4, 5])', 'royalblue', narrow);
  const ts = narrow ? 11 : 13, showAddr = addressCheckbox.checked();
  const sw = min(54, (w - 16) / N - 4), rowW = N * sw, sx = x + (w - rowW) / 2;
  const done = step > 1, current = step === 1;
  note('array header', x + 8, y + 48, w - 12, 1, ts, 'dimgray');
  cell(x + 8, y + 64, w - 16, 28, narrow ? 'int64 · shape (5,) · data ↓' : 'dtype int64 · shape (5,) · pointer to data ↓',
    'lavender', 'silver', ts);
  stroke('gray');
  strokeWeight(1.2);
  line(sx, y + 92, sx, y + 130);

  // the data buffer: no gaps between the cells
  for (let i = 0; i < N; i++) {
    cell(sx + i * sw, y + 130, sw, 34, VALUES[i], current ? 'gold' : done ? 'palegreen' : 'lightskyblue', 'royalblue', ts + 3);
    if (showAddr) {
      noStroke();
      fill('dimgray');
      textSize(11);
      textAlign(CENTER, TOP);
      text((narrow ? '' : '@') + (ARRAY_ADDR + VALUE_BYTES * i), sx + i * sw + sw / 2, y + 167);
    }
  }
  if (step >= 1) {
    noFill();
    stroke(current ? 'darkorange' : 'seagreen');
    strokeWeight(3);
    rect(sx - 3, y + 127, rowW + 6, 40, 4);
  }
  note(showAddr ? 'address = ' + ARRAY_ADDR + ' + 8 × index' : 'values only, no pointers', x + 10, y + 188, w - 16, 1, ts, 'dimgray');
  note('one block: ' + N + ' values × 8 bytes' + (narrow ? '' : ', side by side'), x + 10, y + 214, w - 16, 1, ts, 'dimgray');

  note('arr * 2', x + 8, y + 236, w - 12, 1, ts + 1, 'black');
  for (let i = 0; i < N; i++) {
    cell(sx + i * sw, y + 254, sw, 24, step >= 1 ? VALUES[i] * 2 : '', step >= 1 ? 'palegreen' : 'whitesmoke',
      step >= 1 ? 'seagreen' : 'silver', ts + 1);
  }
  const say = step === 0 ? 'The array stores the numbers themselves, all the same type, packed into one block.'
    : step === 1 ? 'One call. Compiled C code loops straight along the block: no pointers to follow, no types to check.'
      : 'Finished at step 1. The list is ' + (step < N ? 'still on pass ' + step + ' of ' + N + '.' : 'only now done.');
  note(say, x + 8, y + 284, w - 14, 3, ts, 'black');
}

function formatBytes(b) {
  return b >= 1e6 ? (b / 1e6).toFixed(1) + ' MB' : b >= 1e3 ? (b / 1e3).toFixed(1) + ' kB' : b + ' bytes';
}

// Bottom: the same comparison for n numbers, computed from the byte sizes in the model
function drawScalePanel(x, y, w, h, narrow) {
  const n = SIZES[sizeSlider.value()], ts = narrow ? 11 : 14;
  const pointerBytes = LIST_BYTES + POINTER_BYTES * n, objectBytes = INT_BYTES * n;
  const listBytes = pointerBytes + objectBytes, arrayBytes = VALUE_BYTES * n;
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text('At scale: ' + n.toLocaleString('en-US') + ' whole numbers', x + 10, y + 7);
  textStyle(NORMAL);

  const labelW = narrow ? 74 : 100, valueW = narrow ? 58 : 76;
  const bx = x + 10 + labelW, bw = w - 20 - labelW - valueW, scale = bw / listBytes;
  const rows = [['Python list', [[pointerBytes, 'sandybrown', 'pointers'], [objectBytes, 'darkorange', 'int objects']], listBytes],
    ['NumPy array', [[arrayBytes, 'royalblue', 'values']], arrayBytes]];
  rows.forEach(([label, parts, total], r) => {
    const ry = y + 28 + r * 22;
    noStroke();
    fill('black');
    textSize(ts);
    textAlign(LEFT, CENTER);
    text(label, x + 10, ry + 9);
    let px = bx;
    for (const [bytes, col, name] of parts) {
      fill(col);
      rect(px, ry, bytes * scale, 18);
      fill('white');
      textSize(11);
      if (bytes * scale > 64) text(name, px + 5, ry + 9);
      px += bytes * scale;
    }
    fill('black');
    textSize(ts);
    textAlign(RIGHT, CENTER);
    text(formatBytes(total), x + w - 10, ry + 9);
  });
  note('Memory: the list needs ' + (listBytes / arrayBytes).toFixed(1) + ' times as much. Work: the list takes ' +
    n.toLocaleString('en-US') + ' passes through the interpreter, the array takes 1 call. On large arrays that ' +
    'typically makes NumPy about 10 to 100 times faster (a rough range, not a measurement). On tiny arrays the ' +
    'fixed cost of the call cancels the gain.', x + 10, y + 75, w - 20, 4, ts, 'black');
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  sizeSlider.size(canvasWidth - sliderLeftMargin - margin);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
