// Matrix Multiplication Visualizer
// CANVAS_HEIGHT: 470
// Bloom L2-L3 (Understand, Apply): students step through the cells of C = A @ B, see each one
// built as a row of A times a column of B, and compare it with the element-wise product A * B.
//
// Model: for A of shape (m, n) and B of shape (n, p), C = A @ B has shape (m, p) and
//   C[i, j] = A[i, 0] * B[0, j] + A[i, 1] * B[1, j] + ... + A[i, n-1] * B[n-1, j].
// The columns of A must equal the rows of B, otherwise NumPy raises the matmul ValueError
// quoted on screen. A * B multiplies matching positions and needs shapes that broadcast,
// which for the shapes offered here means identical shapes.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 390;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const SHAPES = [[3, 2], [2, 3], [2, 2], [3, 3]];
const MARKS = ['①', '②', '③'];        // pair up the numbers that get multiplied together

let aSelect, bSelect, opSelect, prevButton, nextButton;
let cellIndex = 0;                // which cell of C is being computed, counted row by row
let lastKey = '';
let cellBoxes = [];               // clickable cells of C: { x, y, w, h, index }

// ---- NumPy behaviour ----
// A holds 1, 2, 3, ... and B continues from 7, which gives the chapter's example for (3, 2) @ (2, 3)
function makeMatrix(shape, first) {
  return Array.from({ length: shape[0] }, (r, i) => Array.from({ length: shape[1] }, (c, j) => first + i * shape[1] + j));
}
function matmul(A, B) {
  if (A[0].length !== B.length) return null;
  return A.map(row => B[0].map((unused, j) => row.reduce((sum, a, k) => sum + a * B[k][j], 0)));
}
function elementwise(A, B) {
  if (A.length !== B.length || A[0].length !== B[0].length) return null;
  return A.map((row, i) => row.map((a, j) => a * B[i][j]));
}
function errorText(op, sa, sb) {
  return op === '@' ? 'matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature ' +
    '(n?,k),(k,m?)->(n?,m?) (size ' + sb[0] + ' is different from ' + sa[1] + ')'
    : 'operands could not be broadcast together with shapes (' + sa.join(',') + ') (' + sb.join(',') + ')';
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  aSelect = createSelect();
  aSelect.parent(mainElement);
  aSelect.position(30, drawHeight + 10);
  opSelect = createSelect();
  opSelect.parent(mainElement);
  opSelect.position(108, drawHeight + 10);
  opSelect.option('@  matrix product', '@');
  opSelect.option('*  element-wise', '*');
  bSelect = createSelect();
  bSelect.parent(mainElement);
  bSelect.position(278, drawHeight + 10);
  SHAPES.forEach(s => { aSelect.option('(' + s.join(', ') + ')'); bSelect.option('(' + s.join(', ') + ')'); });
  bSelect.elt.selectedIndex = 1;

  prevButton = createButton('Previous');
  prevButton.parent(mainElement);
  prevButton.position(10, drawHeight + 45);
  prevButton.mousePressed(() => { cellIndex--; });
  nextButton = createButton('Next');
  nextButton.parent(mainElement);
  nextButton.position(86, drawHeight + 45);
  nextButton.mousePressed(() => { cellIndex++; });

  describe('Three grids of numbers: matrix A, matrix B, and the result C. One row of A, one column of B and one ' +
    'cell of C are highlighted, and a panel writes out the products and the sum that give that cell. Stepping ' +
    'forward fills C one cell at a time. Menus change the two shapes and switch between the matrix product and ' +
    'element-wise multiplication, and the NumPy error message appears when the shapes do not fit.', LABEL);
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
  const sa = SHAPES[aSelect.elt.selectedIndex], sb = SHAPES[bSelect.elt.selectedIndex], op = opSelect.value();
  const A = makeMatrix(sa, 1), B = makeMatrix(sb, 7), C = op === '@' ? matmul(A, B) : elementwise(A, B);
  const key = sa + '|' + sb + '|' + op;
  if (key !== lastKey) { cellIndex = 0; lastKey = key; }
  const total = C ? C.length * C[0].length : 0;
  cellIndex = constrain(cellIndex, 0, max(total - 1, 0));
  const ci = C ? floor(cellIndex / C[0].length) : 0, cj = C ? cellIndex % C[0].length : 0;
  prevButton.elt.disabled = !C || cellIndex === 0;
  nextButton.elt.disabled = !C || cellIndex === total - 1;
  cellBoxes = [];

  textWrap(WORD);
  textStyle(NORMAL);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Matrix Multiplication Visualizer', canvasWidth / 2, 8);

  // the shape rule, with the sizes that have to agree in color
  const ts = narrow ? 12 : 15, good = C ? 'seagreen' : 'firebrick';
  textSize(ts + 1);
  const outcome = C ? ['   →   C.shape (' + C.length + ', ' + C[0].length + ')', 'black'] : ['   →   error', 'firebrick'];
  drawTokens(op === '@' ? [['A.shape (' + sa[0] + ', ', 'black'], [sa[1], good], [')  @  B.shape (', 'black'], [sb[0], good],
    [', ' + sb[1] + ')', 'black'], outcome]
    : [['A.shape ', 'black'], ['(' + sa.join(', ') + ')', good], ['  *  B.shape ', 'black'], ['(' + sb.join(', ') + ')', good], outcome],
  canvasWidth / 2, narrow ? 36 : 44, true);
  noStroke();
  fill('dimgray');
  textSize(ts);
  textAlign(CENTER, TOP);
  text(op === '@' ? (C ? 'The inner sizes match, ' + sa[1] + ' = ' + sb[0] + '. The outer sizes give C: ' + sa[0] + ' rows and ' + sb[1] + ' columns.'
    : 'The inner sizes differ. The columns of A must equal the rows of B.')
    : C ? 'The shapes are the same, so * multiplies matching positions. This is not the matrix product.'
      : '* multiplies matching positions, so the two shapes must match (or broadcast).',
  margin, narrow ? 56 : 68, canvasWidth - 2 * margin, 36);

  // three grids: A  op  B  =  C
  const opW = narrow ? 18 : 34, top = narrow ? 122 : 134;
  const slotW = (canvasWidth - 2 * margin - 2 * opW) / 3;
  const cw = min(narrow ? 36 : 54, slotW / 3 - 2), ch = narrow ? 30 : 38;
  const centers = [0, 1, 2].map(k => margin + slotW / 2 + k * (slotW + opW));
  // which cells take part: a whole row and column for @, one position for *
  const show = C || op === '@';
  const inA = (i, j) => show && (op === '@' ? i === ci : i === ci && j === cj);
  const inB = (i, j) => show && (op === '@' ? j === cj : i === ci && j === cj);
  drawMatrix(centers[0], top, cw, ch, A, 'A', sa, ts, (i, j) => inA(i, j)
    ? { bg: 'lightskyblue', ink: 'mediumblue', mark: op === '@' ? MARKS[j] : '' } : { bg: 'white', ink: 'black' });
  drawMatrix(centers[1], top, cw, ch, B, 'B', sb, ts, (i, j) => inB(i, j)
    ? { bg: 'lightgreen', ink: 'darkgreen', mark: op === '@' ? MARKS[i] : '' } : { bg: 'white', ink: 'black' });
  if (C) {
    drawMatrix(centers[2], top, cw, ch, C, 'C = A ' + op + ' B', [C.length, C[0].length], ts, (i, j) => {
      const index = i * C[0].length + j;
      return index === cellIndex ? { bg: 'gold', ink: 'black', index } : index < cellIndex ? { bg: 'white', ink: 'black', index }
        : { bg: 'whitesmoke', ink: 'gray', label: '?', index };
    });
  }
  noStroke();
  textAlign(CENTER, CENTER);
  textStyle(BOLD);
  textSize(narrow ? 18 : 24);
  fill('black');
  text(op, margin + slotW + opW / 2, top + ch * 1.5);
  text('=', margin + 2 * slotW + 1.5 * opW, top + ch * 1.5);
  if (!C) {
    fill('firebrick');
    textSize(narrow ? 30 : 40);
    text('✗', centers[2], top + ch * 1.5);
  }
  textStyle(NORMAL);
  cursor(cellBoxes.some(mouseOver) ? HAND : ARROW);

  const panelY = top + 3 * ch + (narrow ? 8 : 12);
  drawCalculation(margin, panelY, canvasWidth - 2 * margin, drawHeight - panelY - 8, A, B, C, op, ci, cj, narrow);

  // control labels
  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('A', 10, drawHeight + 22);
  text('B', 258, drawHeight + 22);
  textSize(narrow ? 14 : defaultTextSize);
  text(C ? 'Cell ' + (cellIndex + 1) + ' of ' + total + ':  C[' + ci + ', ' + cj + ']' : 'No cells to compute', 144, drawHeight + 57);
}

// Text pieces in different colors on one line: [[text, color], ...]
function drawTokens(tokens, x, y, centered) {
  let tx = centered ? x - tokens.reduce((sum, t) => sum + textWidth(String(t[0])), 0) / 2 : x;
  noStroke();
  textAlign(LEFT, TOP);
  for (const [str, ink] of tokens) {
    fill(ink);
    textStyle(ink === 'black' ? NORMAL : BOLD);
    text(str, tx, y);
    tx += textWidth(String(str));
  }
  textStyle(NORMAL);
}

// A matrix centered on cx with its name and shape above. style(i, j) gives each cell its look.
function drawMatrix(cx, top, cw, ch, M, name, shape, ts, style) {
  noStroke();
  textAlign(CENTER, BOTTOM);
  fill('black');
  textStyle(BOLD);
  textSize(ts + 1);
  text(name, cx, top - ts - 8);
  textStyle(NORMAL);
  fill('dimgray');
  textSize(ts);
  text('shape (' + shape.join(', ') + ')', cx, top - 4);
  const gx = cx - M[0].length * cw / 2;
  for (let i = 0; i < M.length; i++) {
    for (let j = 0; j < M[0].length; j++) {
      const s = style(i, j), x = gx + j * cw, y = top + i * ch;
      fill(s.bg);
      stroke('dimgray');
      strokeWeight(1);
      rect(x, y, cw, ch);
      noStroke();
      fill(s.ink);
      textAlign(CENTER, CENTER);
      textStyle(s.bg === 'white' || s.label ? NORMAL : BOLD);
      textSize(min(ts + 1, cw * 0.4));
      text(s.label || M[i][j], x + cw / 2, y + ch / 2 + (s.mark ? 4 : 1));
      if (s.mark) {
        textStyle(NORMAL);
        textSize(11);
        textAlign(LEFT, TOP);
        text(s.mark, x + 2, y + 1);
      }
      if (s.index !== undefined) cellBoxes.push({ x, y, w: cw, h: ch, index: s.index });
    }
  }
  textStyle(NORMAL);
}

// The arithmetic behind the gold cell, or the reason there is no result
function drawCalculation(x, y, w, h, A, B, C, op, ci, cj, narrow) {
  fill('lightyellow');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = narrow ? 12 : 15, lh = narrow ? 18 : 28, tx = x + 12;
  let ty = y + (narrow ? 8 : 10);
  const sa = [A.length, A[0].length], sb = [B.length, B[0].length];
  noStroke();
  textAlign(LEFT, TOP);
  if (!C) {
    fill('firebrick');
    textStyle(BOLD);
    textSize(ts);
    textLeading(ts + 4);
    text('ValueError: ' + errorText(op, sa, sb), tx, ty, w - 24, (narrow ? 4 : 2) * (ts + 4) + 3);
    ty += (narrow ? 4 : 2) * (ts + 4) + 8;
    fill('black');
    textStyle(NORMAL);
    text(op === '@' ? 'In plain words: a row of A has ' + sa[1] + ' numbers and a column of B has ' + sb[0] + ', so they ' +
      'cannot be paired up.' + (sa[1] === sb[1] ? ' A @ B.T would work, because B.T has shape (' + sb[1] + ', ' + sb[0] + ').' : '')
      : 'In plain words: A is ' + sa.join(' by ') + ' and B is ' + sb.join(' by ') + ', so some positions have no partner.',
    tx, ty, w - 24, 4 * (ts + 4));
    return;
  }
  textSize(narrow ? 13 : 18);
  const cell = 'C[' + ci + ', ' + cj + ']';
  if (op === '@') {
    drawTokens([[cell + '  =  ', 'black'], ['row ' + ci + ' of A', 'mediumblue'], ['  ·  ', 'black'], ['column ' + cj + ' of B', 'darkgreen']], tx, ty);
    const terms = A[ci].map((a, k) => [a, B[k][cj]]);
    const products = [['=  ', 'black']];
    terms.forEach(([a, b], k) => products.push([a, 'mediumblue'], ['×', 'black'], [b, 'darkgreen'], [k < terms.length - 1 ? '  +  ' : '', 'black']));
    drawTokens(products, tx, ty + lh);
    drawTokens([['=  ' + terms.map(t => t[0] * t[1]).join('  +  ') + '  =  ', 'black'], [C[ci][cj], 'chocolate']], tx, ty + 2 * lh);
  } else {
    drawTokens([[cell + '  =  ', 'black'], ['A[' + ci + ', ' + cj + ']', 'mediumblue'], ['  ×  ', 'black'], ['B[' + ci + ', ' + cj + ']', 'darkgreen']], tx, ty);
    drawTokens([['=  ', 'black'], [A[ci][cj], 'mediumblue'], ['  ×  ', 'black'], [B[ci][cj], 'darkgreen']], tx, ty + lh);
    drawTokens([['=  ', 'black'], [C[ci][cj], 'chocolate']], tx, ty + 2 * lh);
  }
  noStroke();
  fill('dimgray');
  textSize(ts);
  textLeading(ts + 4);
  textAlign(LEFT, TOP);
  const tip = op === '@' ? 'Each cell of C is a dot product: multiply the pairs with the same circled number, then add.'
    : 'Each cell of C uses only the two numbers in the same position. No rows meet columns.';
  if (narrow) text(tip + ' Press Next, or click a cell of C.', tx, ty + 3 * lh + 2, w - 24, 3 * (ts + 4));
  else text(tip + ' Press Next to fill the next cell, or click a cell of C.', x + w * 0.56, ty + 2, w * 0.44 - 12, 5 * (ts + 4));
}

// Clicking a cell of C jumps to it.
function mouseOver(b) {
  return mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h;
}
function mousePressed() {
  const box = cellBoxes.find(mouseOver);
  if (box) cellIndex = box.index;
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
